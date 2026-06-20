#include "continuous_batch_engine.h"
#include <spdlog/spdlog.h>

ContinuousBatchEngine::ContinuousBatchEngine(const std::string& model_path,
                                             EngineConfig config,
                                             Metrics& metrics)
    : config_(config),
      metrics_(metrics),
      slots_(config.max_slots, [this](int seq_id) {
          llama_kv_self_seq_rm(ctx_, seq_id, 0, -1);
      }),
      samplers_(config.max_slots, nullptr),
      results_(config.max_slots),
      prompt_tokens_(config.max_slots),
      tokens_generated_(config.max_slots, 0),
      in_prefill_(config.max_slots, false)
{
    llama_backend_init();

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
    mp.n_gpu_layers = config_.n_gpu_layers;
    model_ = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model_) throw std::runtime_error("ContinuousBatchEngine: failed to load model: " + model_path);

    // n_ctx sized to hold max_slots sequences of up to (max_tokens + typical prompt) tokens each.
    const int n_ctx = config_.max_slots * (config_.max_tokens + 256);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = static_cast<uint32_t>(n_ctx);
    cp.n_batch    = static_cast<uint32_t>(n_ctx); // allow full context in one decode call
    cp.n_threads  = static_cast<uint32_t>(config_.n_threads);
    cp.n_seq_max  = static_cast<uint32_t>(config_.max_slots);
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) {
        llama_model_free(model_);
        throw std::runtime_error("ContinuousBatchEngine: failed to create context");
    }

    vocab_ = llama_model_get_vocab(model_);

    spdlog::info("ContinuousBatchEngine: model loaded, max_slots={}, n_ctx={}", config_.max_slots, n_ctx);

    engine_thread_ = std::thread([this] { engine_loop(); });
}

ContinuousBatchEngine::~ContinuousBatchEngine() {
    shutdown();
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
    llama_backend_free();
}

void ContinuousBatchEngine::enqueue(Request req) {
    {
        std::lock_guard lock(pending_mutex_);
        pending_.push(std::move(req));
    }
    pending_cv_.notify_one();
}

void ContinuousBatchEngine::shutdown() {
    running_ = false;
    pending_cv_.notify_all();
    if (engine_thread_.joinable()) engine_thread_.join();
}

// --- Private ---

void ContinuousBatchEngine::admit_pending() {
    // Drain under lock (fast: just pointer moves), then tokenize outside the lock.
    std::vector<Request> to_admit;
    {
        std::lock_guard lock(pending_mutex_);
        int can_admit = slots_.free_count();
        while (can_admit > 0 && !pending_.empty()) {
            to_admit.push_back(std::move(pending_.front()));
            pending_.pop();
            --can_admit;
        }
    }

    for (auto& req : to_admit) {
        auto wait = std::chrono::steady_clock::now() - req.enqueue_time;
        metrics_.queue_wait_seconds.Observe(std::chrono::duration<double>(wait).count());

        int sid = slots_.acquire(std::move(req));
        if (sid < 0) {
            spdlog::error("admit_pending: acquire failed unexpectedly");
            continue;
        }

        const std::string& payload = slots_.slot(sid).request.payload;
        const std::string formatted = (payload.rfind("<|im_start|>", 0) == 0)
            ? payload
            : "<|im_start|>user\n" + payload + "<|im_end|>\n<|im_start|>assistant\n";

        auto& toks = prompt_tokens_[sid];
        toks.resize(formatted.size() + 16);
        int n = llama_tokenize(vocab_, formatted.c_str(), (int32_t)formatted.size(),
                               toks.data(), (int32_t)toks.size(), true, true);
        if (n < 0) {
            toks.resize(-n);
            n = llama_tokenize(vocab_, formatted.c_str(), (int32_t)formatted.size(),
                               toks.data(), (int32_t)toks.size(), true, true);
        }
        if (n <= 0) {
            auto& r = slots_.slot(sid).request;
            if (r.on_complete) r.on_complete(R"({"error":"tokenization_failed"})");
            slots_.release(sid);
            continue;
        }
        toks.resize(n);

        in_prefill_[sid]        = true;
        results_[sid].clear();
        tokens_generated_[sid]  = 0;

        if (samplers_[sid]) llama_sampler_free(samplers_[sid]);
        samplers_[sid] = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(samplers_[sid], llama_sampler_init_greedy());
    }
}

void ContinuousBatchEngine::retire_expired(std::chrono::steady_clock::time_point now) {
    for (int sid = 0; sid < slots_.max_slots(); ++sid) {
        if (!slots_.slot(sid).occupied) continue;
        const auto& req = slots_.slot(sid).request;
        if (now > req.enqueue_time + req.deadline) {
            if (req.on_complete) req.on_complete(R"({"error":"deadline_exceeded"})");
            metrics_.requests_deadline_exceeded.Increment();
            metrics_.requests_total.Increment();
            retire_slot(sid);
        }
    }
}

void ContinuousBatchEngine::retire_slot(int sid) {
    if (samplers_[sid]) {
        llama_sampler_free(samplers_[sid]);
        samplers_[sid] = nullptr;
    }
    results_[sid].clear();
    prompt_tokens_[sid].clear();
    tokens_generated_[sid] = 0;
    in_prefill_[sid]        = false;
    slots_.release(sid);
}

void ContinuousBatchEngine::engine_loop() {
    spdlog::info("ContinuousBatchEngine: engine thread started");

    while (running_) {
        // Block only when truly idle — no active sequences and nothing pending.
        if (slots_.free_count() == slots_.max_slots()) {
            std::unique_lock lock(pending_mutex_);
            pending_cv_.wait(lock, [this] {
                return !pending_.empty() || !running_;
            });
            if (!running_) break;
        }

        admit_pending();

        auto now = std::chrono::steady_clock::now();
        retire_expired(now);

        // Collect occupied slots.
        std::vector<int> active;
        active.reserve(slots_.max_slots());
        for (int sid = 0; sid < slots_.max_slots(); ++sid)
            if (slots_.slot(sid).occupied) active.push_back(sid);

        if (active.empty()) continue;

        // Update slot utilization gauge.
        double utilization = static_cast<double>(active.size()) / slots_.max_slots();
        metrics_.kv_slot_utilization.Set(utilization);

        // Count tokens in this decode step.
        int n_tokens = 0;
        for (int sid : active)
            n_tokens += in_prefill_[sid] ? (int)prompt_tokens_[sid].size() : 1;
        if (n_tokens == 0) continue;

        // Build the llama_batch. Prefill sequences contribute all prompt tokens;
        // generating sequences contribute their last sampled token.
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        batch.n_tokens = 0;
        std::vector<int32_t> logit_pos(slots_.max_slots(), -1);

        for (int sid : active) {
            Slot& s = slots_.slot(sid);
            if (in_prefill_[sid]) {
                const auto& toks = prompt_tokens_[sid];
                for (int j = 0; j < (int)toks.size(); ++j) {
                    int idx             = batch.n_tokens++;
                    batch.token[idx]    = toks[j];
                    batch.pos[idx]      = j;
                    batch.n_seq_id[idx] = 1;
                    batch.seq_id[idx][0]= sid;
                    bool last           = (j + 1 == (int)toks.size());
                    batch.logits[idx]   = last ? 1 : 0;
                    if (last) logit_pos[sid] = idx;
                }
                s.kv_pos = (int32_t)toks.size();
            } else {
                int idx             = batch.n_tokens++;
                batch.token[idx]    = s.last_token;
                batch.pos[idx]      = s.kv_pos++;
                batch.n_seq_id[idx] = 1;
                batch.seq_id[idx][0]= sid;
                batch.logits[idx]   = 1;
                logit_pos[sid]      = idx;
            }
        }

        metrics_.requests_inflight.Increment(static_cast<double>(active.size()));
        bool ok = (llama_decode(ctx_, batch) == 0);
        llama_batch_free(batch);

        if (!ok) {
            spdlog::error("llama_decode failed — retiring {} sequences", active.size());
            for (int sid : active) {
                if (!slots_.slot(sid).occupied) continue;
                auto& req = slots_.slot(sid).request;
                if (req.on_complete) req.on_complete(R"({"error":"decode_failed"})");
                metrics_.requests_total.Increment();
                retire_slot(sid);
            }
            metrics_.requests_inflight.Decrement(static_cast<double>(active.size()));
            continue;
        }

        auto end_time = std::chrono::steady_clock::now();

        for (int sid : active) {
            if (!slots_.slot(sid).occupied) continue;  // retired by deadline check above
            Slot& s = slots_.slot(sid);

            llama_token tok = llama_sampler_sample(samplers_[sid], ctx_, logit_pos[sid]);
            bool eos        = llama_vocab_is_eog(vocab_, tok);

            if (!eos) {
                char piece[256];
                int np = llama_token_to_piece(vocab_, tok, piece, sizeof(piece), 0, false);
                if (np > 0) results_[sid].append(piece, np);
                s.last_token     = tok;
                in_prefill_[sid] = false;
                ++tokens_generated_[sid];
            }

            int  limit   = s.request.max_new_tokens > 0 ? s.request.max_new_tokens : config_.max_tokens;
            bool max_hit = (tokens_generated_[sid] >= limit);

            if (eos || max_hit) {
                metrics_.latency_seconds.Observe(
                    std::chrono::duration<double>(end_time - s.request.enqueue_time).count());
                metrics_.requests_total.Increment();
                if (s.request.on_complete)
                    s.request.on_complete(std::move(results_[sid]));
                retire_slot(sid);
            }
        }

        metrics_.requests_inflight.Decrement(static_cast<double>(active.size()));
        metrics_.batch_size.Observe(static_cast<double>(active.size()));
    }

    spdlog::info("ContinuousBatchEngine: engine thread stopped");
}
