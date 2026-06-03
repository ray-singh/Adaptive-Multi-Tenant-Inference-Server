#include "worker_pool.h"
#include <spdlog/spdlog.h>

WorkerPool::WorkerPool(std::size_t num_workers, InferenceFn fn, Metrics& metrics)
    : num_workers_(num_workers), inference_fn_(std::move(fn)), metrics_(metrics) {
    for (std::size_t i = 0; i < num_workers_; ++i)
        threads_.emplace_back([this] { worker_loop(); });
}

WorkerPool::~WorkerPool() { shutdown(); }

void WorkerPool::submit_batch(std::vector<Request> batch) {
    metrics_.batch_size.Observe(static_cast<double>(batch.size()));
    for (auto& req : batch) {
        auto wait = std::chrono::steady_clock::now() - req.enqueue_time;
        metrics_.queue_wait_seconds.Observe(
            std::chrono::duration<double>(wait).count());

        std::string result = inference_fn_(req.payload);

        auto latency = std::chrono::steady_clock::now() - req.enqueue_time;
        metrics_.latency_seconds.Observe(
            std::chrono::duration<double>(latency).count());
        metrics_.requests_total.Increment();

        if (req.on_complete) req.on_complete(std::move(result));
    }
}

void WorkerPool::shutdown() {
    running_ = false;
    for (auto& t : threads_)
        if (t.joinable()) t.join();
    threads_.clear();
}

void WorkerPool::worker_loop() {
    spdlog::info("Worker thread started");
    // Workers are driven by submit_batch; this loop is a placeholder
    // for future async dispatch queues per worker.
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
