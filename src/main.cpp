#include "server.h"
#include "scheduler.h"
#include "worker_pool.h"
#include "metrics.h"
#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <spdlog/spdlog.h>
#include <csignal>
#include <thread>
#include <chrono>

// Pointers set before signal registration; only written once from main.
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

    // Stub inference function — replace with real model call
    auto inference_fn = [](const std::string& payload) -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds{10}); // simulate compute
        return "result:" + payload;
    };

    // Worker pool
    WorkerPool workers(4, inference_fn, metrics);

    // Scheduler dispatch loop (runs on its own thread)
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

    // Register signal handlers after all objects are constructed
    g_server    = &server;
    g_scheduler = &scheduler;
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    spdlog::info("Send SIGINT or SIGTERM to shut down cleanly");
    server.run(); // blocks until server.stop() is called

    // Ordered shutdown: stop scheduler first so dispatch thread exits,
    // then drain the worker pool.
    scheduler.stop();
    if (dispatch.joinable()) dispatch.join();
    workers.shutdown();

    return 0;
}
