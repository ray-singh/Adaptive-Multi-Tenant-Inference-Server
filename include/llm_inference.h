#pragma once
#include <mutex>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;

// Wraps a llama.cpp model context. generate() is serialized via a mutex.
// generate_batch() runs multiple sequences in a single llama_decode call per step,
// giving the GPU real batch utilization instead of N sequential calls.
class LlamaInference {
public:
    // n_ctx should be large enough for all concurrent sequences:
    // n_ctx >= max_batch * (max_prompt_tokens + max_gen_tokens)
    LlamaInference(const std::string& model_path, int n_ctx = 512, int n_threads = 4, int n_gpu_layers = -1);
    ~LlamaInference();

    LlamaInference(const LlamaInference&)            = delete;
    LlamaInference& operator=(const LlamaInference&) = delete;

    std::string generate(const std::string& prompt, int max_tokens = 128);

    // Process multiple prompts in one batched decode per generation step.
    // Returns one result string per input prompt, in the same order.
    std::vector<std::string> generate_batch(const std::vector<std::string>& prompts, int max_tokens = 128);

private:
    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;
    std::mutex     mutex_;
};
