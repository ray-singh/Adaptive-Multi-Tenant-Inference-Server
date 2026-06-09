#include "worker_pool.h"
#include <spdlog/spdlog.h>

WorkerPool::WorkerPool(std::size_t num_workers, InferenceFn fn, Metrics& metrics)
    : num_workers_(num_workers), single_fn_(std::move(fn)), batch_mode_(false), metrics_(metrics) {
    for (std::size_t i = 0; i < num_workers_; ++i)
        threads_.emplace_back([this] { worker_loop(); });
}

WorkerPool::WorkerPool(std::size_t num_workers, BatchInferenceFn fn, Metrics& metrics)
    : num_workers_(num_workers), batch_fn_(std::move(fn)), batch_mode_(true), metrics_(metrics) {
    for (std::size_t i = 0; i < num_workers_; ++i)
        threads_.emplace_back([this] { batch_worker_loop(); });
}

WorkerPool::~WorkerPool() { shutdown(); }

void WorkerPool::submit_batch(std::vector<Request> batch) {
    metrics_.batch_size.Observe(static_cast<double>(batch.size()));
    {
        std::lock_guard lock(work_mutex_);
        if (batch_mode_) {
            batch_work_queue_.push(std::move(batch));
        } else {
            for (auto& req : batch)
                work_queue_.push(std::move(req));
        }
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

// Single-request mode: each worker pulls one Request at a time.
void WorkerPool::worker_loop() {
    spdlog::info("Worker thread started");
    while (true) {
        Request req;
        {
            std::unique_lock lock(work_mutex_);
            work_cv_.wait(lock, [this] { return !work_queue_.empty() || !running_; });
            if (work_queue_.empty()) break;
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

        std::string result = single_fn_(req.payload);

        auto latency = std::chrono::steady_clock::now() - req.enqueue_time;
        metrics_.latency_seconds.Observe(std::chrono::duration<double>(latency).count());
        metrics_.requests_total.Increment();

        if (req.on_complete) req.on_complete(std::move(result));
    }
}

// Batch mode: each worker pulls an entire batch and runs it through batch_fn_ in one call.
// This lets the GPU process all sequences in parallel rather than serially.
void WorkerPool::batch_worker_loop() {
    spdlog::info("Batch worker thread started");
    while (true) {
        std::vector<Request> batch;
        {
            std::unique_lock lock(work_mutex_);
            work_cv_.wait(lock, [this] { return !batch_work_queue_.empty() || !running_; });
            if (batch_work_queue_.empty()) break;
            batch = std::move(batch_work_queue_.front());
            batch_work_queue_.pop();
        }

        auto now = std::chrono::steady_clock::now();

        // Separate expired requests (deadline already exceeded) from active ones.
        std::vector<std::string> payloads;
        std::vector<std::size_t> active_indices;

        for (std::size_t i = 0; i < batch.size(); ++i) {
            metrics_.queue_wait_seconds.Observe(
                std::chrono::duration<double>(now - batch[i].enqueue_time).count());

            if (now > batch[i].enqueue_time + batch[i].deadline) {
                if (batch[i].on_complete) batch[i].on_complete(R"({"error":"deadline_exceeded"})");
                metrics_.requests_total.Increment();
            } else {
                payloads.push_back(batch[i].payload);
                active_indices.push_back(i);
            }
        }

        if (payloads.empty()) continue;

        auto results = batch_fn_(payloads);
        auto end     = std::chrono::steady_clock::now();

        for (std::size_t j = 0; j < active_indices.size(); ++j) {
            std::size_t i = active_indices[j];
            metrics_.latency_seconds.Observe(
                std::chrono::duration<double>(end - batch[i].enqueue_time).count());
            metrics_.requests_total.Increment();

            std::string res = (j < results.size()) ? std::move(results[j]) : "(no result)";
            if (batch[i].on_complete) batch[i].on_complete(std::move(res));
        }
    }
}
