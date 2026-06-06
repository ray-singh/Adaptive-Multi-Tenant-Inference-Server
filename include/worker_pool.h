#pragma once
#include "request.h"
#include "metrics.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// Simulates model inference. Replace with real model call later.
using InferenceFn = std::function<std::string(const std::string& payload)>;

// Fixed pool of threads. Each thread pulls from an internal work queue and runs inference.
class WorkerPool {
public:
    WorkerPool(std::size_t num_workers, InferenceFn fn, Metrics& metrics);
    ~WorkerPool();

    // Non-blocking: enqueues the batch for workers to process concurrently.
    void submit_batch(std::vector<Request> batch);
    void shutdown();

private:
    void worker_loop();

    std::size_t num_workers_;
    InferenceFn inference_fn_;
    Metrics& metrics_;

    std::queue<Request> work_queue_;
    std::mutex work_mutex_;
    std::condition_variable work_cv_;

    std::vector<std::thread> threads_;
    std::atomic<bool> running_{true};
};
