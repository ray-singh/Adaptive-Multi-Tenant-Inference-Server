#include "llm_inference.h"
#include <llama.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

LlamaInference::LlamaInference(const std::string& model_path, int n_ctx, int n_threads, int n_gpu_layers) {
    llama_backend_init();

    // Route llama.cpp log output through spdlog instead of raw stderr.
    llama_log_set([](ggml_log_level level, const char* text, void*) {
        std::string msg(text);
        if (!msg.empty() && msg.back() == '\n') msg.pop_back();
        if (msg.empty()) return;
        switch (level) {
            case GGML_LOG_LEVEL_ERROR: spdlog::error("[llama] {}", msg); break;
            case GGML_LOG_LEVEL_WARN:  spdlog::warn ("[llama] {}", msg); break;
            default:                   spdlog::debug("[llama] {}", msg); break;
        }
    }, nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;  // -1 = offload all layers to Metal
    model_ = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model_)
        throw std::runtime_error("Failed to load model: " + model_path);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = static_cast<uint32_t>(n_ctx);
    cp.n_threads = static_cast<uint32_t>(n_threads);
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) {
        llama_model_free(model_);
        throw std::runtime_error("Failed to create llama context");
    }

    spdlog::info("Model loaded: {} (n_ctx={}, n_threads={}, n_gpu_layers={})",
                 model_path, n_ctx, n_threads, n_gpu_layers);
}

LlamaInference::~LlamaInference() {
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
    llama_backend_free();
}

std::string LlamaInference::generate(const std::string& prompt, int max_tokens) {
    std::lock_guard lock(mutex_);

    const std::string formatted = (prompt.rfind("<|im_start|>", 0) == 0)
        ? prompt
        : "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";

    const llama_vocab* vocab = llama_model_get_vocab(model_);

    std::vector<llama_token> tokens(formatted.size() + 16);
    int n_prompt = llama_tokenize(vocab, formatted.c_str(),
                                  static_cast<int32_t>(formatted.size()),
                                  tokens.data(),
                                  static_cast<int32_t>(tokens.size()),
                                  /*add_special=*/true,
                                  /*parse_special=*/true);
    if (n_prompt < 0) {
        tokens.resize(-n_prompt);
        n_prompt = llama_tokenize(vocab, formatted.c_str(),
                                  static_cast<int32_t>(formatted.size()),
                                  tokens.data(),
                                  static_cast<int32_t>(tokens.size()),
                                  true, true);
    }
    if (n_prompt <= 0) return "(tokenization failed)";
    tokens.resize(n_prompt);

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx_, batch) != 0) {
        llama_kv_self_clear(ctx_);
        return "(prompt decode failed)";
    }

    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::string result;
    result.reserve(max_tokens * 4);

    for (int i = 0; i < max_tokens; ++i) {
        llama_token token = llama_sampler_sample(smpl, ctx_, -1);

        if (llama_vocab_is_eog(vocab, token)) break;

        char piece[256];
        int n_piece = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
        if (n_piece > 0) result.append(piece, n_piece);

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx_, next) != 0) break;
    }

    llama_sampler_free(smpl);
    llama_kv_self_clear(ctx_);
    return result;
}

std::vector<std::string> LlamaInference::generate_batch(
    const std::vector<std::string>& prompts, int max_tokens)
{
    std::lock_guard lock(mutex_);
    if (prompts.empty()) return {};

    const int n = static_cast<int>(prompts.size());
    const llama_vocab* vocab = llama_model_get_vocab(model_);

    // Tokenize all prompts, one vector<token> per sequence.
    std::vector<std::vector<llama_token>> seqs(n);
    for (int i = 0; i < n; ++i) {
        const std::string& p = prompts[i];
        const std::string formatted = (p.rfind("<|im_start|>", 0) == 0)
            ? p
            : "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n";

        seqs[i].resize(formatted.size() + 16);
        int n_tok = llama_tokenize(vocab, formatted.c_str(),
                                   static_cast<int32_t>(formatted.size()),
                                   seqs[i].data(), static_cast<int32_t>(seqs[i].size()),
                                   true, true);
        if (n_tok < 0) {
            seqs[i].resize(-n_tok);
            n_tok = llama_tokenize(vocab, formatted.c_str(),
                                   static_cast<int32_t>(formatted.size()),
                                   seqs[i].data(), static_cast<int32_t>(seqs[i].size()),
                                   true, true);
        }
        seqs[i].resize(n_tok > 0 ? n_tok : 0);
    }

    int32_t total_prompt = 0;
    for (auto& s : seqs) total_prompt += static_cast<int32_t>(s.size());
    if (total_prompt == 0) return std::vector<std::string>(n);

    // Build the prompt batch. Each token is tagged with its sequence ID so the
    // KV cache keeps per-sequence state. We request logits only at the last
    // token of each sequence (where we'll sample the first generated token).
    llama_batch prompt_batch = llama_batch_init(total_prompt, 0, 1);
    prompt_batch.n_tokens = 0;

    std::vector<int32_t> last_logit_idx(n, -1); // batch index of each seq's last prompt token
    std::vector<int32_t> cur_pos(n, 0);          // current KV position per sequence

    for (int i = 0; i < n; ++i) {
        const auto& s = seqs[i];
        for (int j = 0; j < static_cast<int>(s.size()); ++j) {
            int idx = prompt_batch.n_tokens++;
            prompt_batch.token[idx]     = s[j];
            prompt_batch.pos[idx]       = j;
            prompt_batch.n_seq_id[idx]  = 1;
            prompt_batch.seq_id[idx][0] = i;
            prompt_batch.logits[idx]    = (j == static_cast<int>(s.size()) - 1) ? 1 : 0;
        }
        last_logit_idx[i] = prompt_batch.n_tokens - 1;
        cur_pos[i]        = static_cast<int32_t>(s.size());
    }

    if (llama_decode(ctx_, prompt_batch) != 0) {
        llama_batch_free(prompt_batch);
        llama_kv_self_clear(ctx_);
        return std::vector<std::string>(n, "(prompt decode failed)");
    }
    llama_batch_free(prompt_batch);

    // One independent greedy sampler per sequence.
    std::vector<llama_sampler*> samplers(n, nullptr);
    for (int i = 0; i < n; ++i) {
        samplers[i] = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(samplers[i], llama_sampler_init_greedy());
    }

    std::vector<std::string> results(n);
    std::vector<bool>        done(n, false);

    // logit_idx[i]: which batch position holds the logits for sequence i after the last decode.
    // Starts as the last-prompt-token position; updated each step.
    std::vector<int32_t> logit_idx = last_logit_idx;

    for (int step = 0; step < max_tokens; ++step) {
        // Sample the next token for every active sequence.
        std::vector<llama_token> next_toks(n, 0);
        bool any_active = false;

        for (int i = 0; i < n; ++i) {
            if (done[i]) continue;
            next_toks[i] = llama_sampler_sample(samplers[i], ctx_, logit_idx[i]);
            if (llama_vocab_is_eog(vocab, next_toks[i])) {
                done[i] = true;
                continue;
            }
            char piece[256];
            int np = llama_token_to_piece(vocab, next_toks[i], piece, sizeof(piece), 0, false);
            if (np > 0) results[i].append(piece, np);
            any_active = true;
        }

        if (!any_active) break;

        // Build a generation batch: one token per still-active sequence.
        int active = 0;
        for (int i = 0; i < n; ++i) if (!done[i]) ++active;

        llama_batch gen_batch = llama_batch_init(active, 0, 1);
        gen_batch.n_tokens = 0;

        for (int i = 0; i < n; ++i) {
            if (done[i]) continue;
            int idx = gen_batch.n_tokens++;
            gen_batch.token[idx]     = next_toks[i];
            gen_batch.pos[idx]       = cur_pos[i]++;
            gen_batch.n_seq_id[idx]  = 1;
            gen_batch.seq_id[idx][0] = i;
            gen_batch.logits[idx]    = 1; // always need logits for sampling
            logit_idx[i] = idx;
        }

        bool ok = (llama_decode(ctx_, gen_batch) == 0);
        llama_batch_free(gen_batch);
        if (!ok) break;
    }

    for (auto* s : samplers) llama_sampler_free(s);
    llama_kv_self_clear(ctx_);
    return results;
}
