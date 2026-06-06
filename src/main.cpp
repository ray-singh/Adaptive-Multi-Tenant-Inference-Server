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
    if (g_scheduler) g_scheduler->stop();
    if (g_server)    g_server->stop();
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

    // Inference function: use a real LLM if MODEL_PATH is set, else stub.
    std::unique_ptr<LlamaInference> llm;
    InferenceFn inference_fn;

    const char* model_path = std::getenv("MODEL_PATH");
    if (model_path && std::filesystem::exists(model_path)) {
        try {
            llm = std::make_unique<LlamaInference>(model_path, /*n_ctx=*/512, /*n_threads=*/4, /*n_gpu_layers=*/-1);
            inference_fn = [&llm](const std::string& payload) {
                return llm->generate(payload);
            };
        } catch (const std::exception& e) {
            spdlog::error("Failed to load model: {} — falling back to stub", e.what());
        }
    }

    if (!inference_fn) {
        spdlog::warn("MODEL_PATH not set or load failed — using stub inference");
        spdlog::warn("Set MODEL_PATH=/path/to/model.gguf to enable real inference");
        inference_fn = [](const std::string& payload) -> std::string {
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
            return "stub:" + payload;
        };
    }

    // Worker pool
    WorkerPool workers(4, inference_fn, metrics);

    // Scheduler dispatch loop
    std::thread dispatch([&] {
        while (scheduler.is_running()) {
            auto batch = scheduler.next_batch();
            if (!batch.empty())
                workers.submit_batch(std::move(batch));
        }
    });

    // HTTP server
    ServerConfig srv_cfg;
    Server server(srv_cfg, queue, metrics);

    g_server    = &server;
    g_scheduler = &scheduler;
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    spdlog::info("Send SIGINT or SIGTERM to shut down cleanly");
    server.run();

    scheduler.stop();
    if (dispatch.joinable()) dispatch.join();
    workers.shutdown();

    return 0;
}
