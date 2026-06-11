#pragma once
#include "request.h"
#include "kv_slot_manager.h"
#include "metrics.h"
#include <llama.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

struct EngineConfig {
    int max_slots    = 16;   // concurrent sequences; each maps to one KV cache slot (= seq_id)
    int max_tokens   = 128;  // max generated tokens per request before forced retirement
    int n_threads    = 4;
    int n_gpu_layers = -1;   // -1 = offload all layers to Metal/CUDA
};

// Single-threaded decode engine running continuous batching:
// sequences are admitted as slots free rather than waiting for a full batch to drain.
class ContinuousBatchEngine {
public:
    ContinuousBatchEngine(const std::string& model_path, EngineConfig config, Metrics& metrics);
    ~ContinuousBatchEngine();

    ContinuousBatchEngine(const ContinuousBatchEngine&)            = delete;
    ContinuousBatchEngine& operator=(const ContinuousBatchEngine&) = delete;

    // Thread-safe. Called from HTTP handler threads.
    void enqueue(Request req);
    void shutdown();

private:
    void engine_loop();
    void admit_pending();
    void retire_expired(std::chrono::steady_clock::time_point now);
    void retire_slot(int sid);  // free sampler, clear state, release KV slot

    EngineConfig        config_;
    Metrics&            metrics_;
    KVSlotManager       slots_;

    llama_model*        model_ = nullptr;
    llama_context*      ctx_   = nullptr;
    const llama_vocab*  vocab_ = nullptr;

    // Parallel arrays indexed by slot id (= llama seq_id).
    std::vector<llama_sampler*>       samplers_;
    std::vector<std::string>          results_;
    std::vector<std::vector<llama_token>> prompt_tokens_;
    std::vector<int>                  tokens_generated_;
    std::vector<bool>                 in_prefill_;

    std::queue<Request>      pending_;
    std::mutex               pending_mutex_;
    std::condition_variable  pending_cv_;

    std::thread       engine_thread_;
    std::atomic<bool> running_{true};
};
