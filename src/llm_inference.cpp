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

    // Wrap plain prompts in ChatML so Instruct models generate a response.
    // Skip wrapping if the caller already applied a template.
    const std::string formatted = (prompt.rfind("<|im_start|>", 0) == 0)
        ? prompt
        : "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";

    const llama_vocab* vocab = llama_model_get_vocab(model_);

    // Tokenize — resize buffer if the initial estimate is too small.
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

    // Evaluate the prompt tokens.
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx_, batch) != 0) {
        llama_kv_self_clear(ctx_);
        return "(prompt decode failed)";
    }

    // Greedy sampling loop.
    // llama_sampler_sample internally calls llama_sampler_accept — do not call it again.
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
