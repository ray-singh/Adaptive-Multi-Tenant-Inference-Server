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
    {
        std::lock_guard lock(work_mutex_);
        for (auto& req : batch)
            work_queue_.push(std::move(req));
    }
    work_cv_.notify_all();
}

void WorkerPool::shutdown() {
    running_ = false;
    work_cv_.notify_all();
    for (auto& t : threads_)
        if (t.joinable()) t.join();
    threads_.clear();
}

void WorkerPool::worker_loop() {
    spdlog::info("Worker thread started");
    while (true) {
        Request req;
        {
            std::unique_lock lock(work_mutex_);
            work_cv_.wait(lock, [this] { return !work_queue_.empty() || !running_; });
            if (work_queue_.empty()) break; // running_ is false and nothing left
            req = std::move(work_queue_.front());
            work_queue_.pop();
        }

        auto now  = std::chrono::steady_clock::now();
        auto wait = now - req.enqueue_time;
        metrics_.queue_wait_seconds.Observe(std::chrono::duration<double>(wait).count());

        if (now > req.enqueue_time + req.deadline) {
            if (req.on_complete) req.on_complete(R"({"error":"deadline_exceeded"})");
            metrics_.requests_total.Increment();
            continue;
        }

        std::string result = inference_fn_(req.payload);

        auto latency = std::chrono::steady_clock::now() - req.enqueue_time;
        metrics_.latency_seconds.Observe(std::chrono::duration<double>(latency).count());
        metrics_.requests_total.Increment();

        if (req.on_complete) req.on_complete(std::move(result));
    }
}
