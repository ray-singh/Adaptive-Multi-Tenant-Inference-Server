#pragma once
#include <mutex>
#include <string>

struct llama_model;
struct llama_context;

// Wraps a llama.cpp model context. Thread-safe: concurrent callers are
// serialized via an internal mutex. One context owns one KV cache, so
// requests cannot be batched at the model level here — that's a future
// upgrade (llama_decode accepts a multi-sequence batch).
class LlamaInference {
public:
    // Loads the GGUF model at model_path. Throws std::runtime_error on failure.
    // n_gpu_layers: number of layers to offload to Metal/GPU (-1 = all).
    LlamaInference(const std::string& model_path, int n_ctx = 512, int n_threads = 4, int n_gpu_layers = -1);
    ~LlamaInference();

    LlamaInference(const LlamaInference&)            = delete;
    LlamaInference& operator=(const LlamaInference&) = delete;

    std::string generate(const std::string& prompt, int max_tokens = 128);

private:
    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;
    std::mutex     mutex_;
};
