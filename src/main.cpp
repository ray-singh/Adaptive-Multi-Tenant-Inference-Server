#include "server.h"
#include "scheduler.h"
#include "worker_pool.h"
#include "metrics.h"
#include "llm_inference.h"
#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <spdlog/spdlog.h>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <thread>
#include <chrono>

static Server*    g_server    = nullptr;
static Scheduler* g_scheduler = nullptr;

static void on_signal(int) {
    if (g_server)    g_server->stop();
    if (g_scheduler) g_scheduler->stop();
}

int main() {
    // Metrics
    auto registry = std::make_shared<prometheus::Registry>();
    Metrics metrics(registry);
    prometheus::Exposer exposer{"0.0.0.0:9090"};
    exposer.RegisterCollectable(registry);

    // Request queue + scheduler
    RequestQueue queue;
    SchedulerConfig sched_cfg;
    sched_cfg.policy    = SchedulerPolicy::AdaptiveBatch;
    sched_cfg.max_batch = 16;
    sched_cfg.max_wait  = std::chrono::milliseconds{20};
    Scheduler scheduler(queue, sched_cfg);

    // Load real LLM if MODEL_PATH is set, otherwise use a stub.
    std::unique_ptr<LlamaInference> llm;
    std::unique_ptr<WorkerPool>     workers;

    const char* model_path = std::getenv("MODEL_PATH");
    if (model_path && std::filesystem::exists(model_path)) {
        try {
            // n_ctx sized for max_batch sequences of up to 256 tokens each.
            const int n_ctx = sched_cfg.max_batch * 256;
            llm = std::make_unique<LlamaInference>(model_path, n_ctx, /*n_threads=*/4, /*n_gpu_layers=*/-1);

            // BatchInferenceFn runs all prompts in a single multi-sequence decode loop,
            // giving the GPU real batch utilization. One worker is enough (GPU is the bottleneck).
            BatchInferenceFn batch_fn = [&llm](const std::vector<std::string>& payloads) {
                return llm->generate_batch(payloads);
            };
            workers = std::make_unique<WorkerPool>(1, batch_fn, metrics);
            spdlog::info("Using real LLM inference with model-level batching");
        } catch (const std::exception& e) {
            spdlog::error("Failed to load model: {} — falling back to stub", e.what());
        }
    }

    if (!workers) {
        spdlog::warn("MODEL_PATH not set or load failed — using stub inference");
        spdlog::warn("Set MODEL_PATH=/path/to/model.gguf to enable real inference");
        InferenceFn stub_fn = [](const std::string& payload) -> std::string {
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
            return "stub:" + payload;
        };
        workers = std::make_unique<WorkerPool>(4, stub_fn, metrics);
    }

    // Dispatch thread: pull batches from the scheduler, hand to the worker pool.
    std::thread dispatch([&] {
        while (scheduler.is_running()) {
            auto batch = scheduler.next_batch();
            if (!batch.empty())
                workers->submit_batch(std::move(batch));
        }
    });

    // HTTP server
    ServerConfig srv_cfg;
    Server server(srv_cfg, queue, metrics, scheduler);

    g_server    = &server;
    g_scheduler = &scheduler;
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    spdlog::info("Send SIGINT or SIGTERM to shut down cleanly");
    server.run();  // blocks until stop()

    scheduler.stop();
    if (dispatch.joinable()) dispatch.join();
    workers->shutdown();

    return 0;
}
