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

// Single-request inference: called once per request by a worker thread.
using InferenceFn = std::function<std::string(const std::string& payload)>;

// Batch inference: called once per scheduler batch; returns one result per input.
// Use this when the inference backend supports true parallel decoding (e.g. generate_batch).
using BatchInferenceFn = std::function<std::vector<std::string>(const std::vector<std::string>&)>;

// Fixed pool of threads. Supports two modes:
//   InferenceFn    — N threads each process one request at a time (good for CPU stubs).
//   BatchInferenceFn — N threads each process a whole batch in one call (good for GPU).
class WorkerPool {
public:
    WorkerPool(std::size_t num_workers, InferenceFn fn, Metrics& metrics);
    WorkerPool(std::size_t num_workers, BatchInferenceFn fn, Metrics& metrics);
    ~WorkerPool();

    // Non-blocking: enqueues work for the pool to process.
    void submit_batch(std::vector<Request> batch);
    void shutdown();

private:
    void worker_loop();
    void batch_worker_loop();

    std::size_t      num_workers_;
    InferenceFn      single_fn_;
    BatchInferenceFn batch_fn_;
    bool             batch_mode_ = false;
    Metrics&         metrics_;

    std::queue<Request>              work_queue_;       // single-request mode
    std::queue<std::vector<Request>> batch_work_queue_; // batch mode
    std::mutex                       work_mutex_;
    std::condition_variable          work_cv_;
    std::vector<std::thread>         threads_;
    std::atomic<bool>                running_{true};
};
