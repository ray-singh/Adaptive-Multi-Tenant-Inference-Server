#include "server.h"
#include "scheduler.h"
#include "worker_pool.h"
#include "metrics.h"
#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>

int main() {
    // Metrics
    auto registry = std::make_shared<prometheus::Registry>();
    Metrics metrics(registry);
    prometheus::Exposer exposer{"0.0.0.0:9090"};
    exposer.RegisterCollectable(registry);

    // Request queue + scheduler
    RequestQueue queue;
    SchedulerConfig sched_cfg;
    sched_cfg.policy   = SchedulerPolicy::AdaptiveBatch;
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
        while (true) {
            auto batch = scheduler.next_batch();
            if (!batch.empty())
                workers.submit_batch(std::move(batch));
        }
    });
    dispatch.detach();

    // HTTP server (blocks)
    ServerConfig srv_cfg;
    Server server(srv_cfg, queue, metrics);
    server.run();

    return 0;
}
