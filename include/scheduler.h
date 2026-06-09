#pragma once
#include "request.h"
#include "request_queue.h"
#include <atomic>
#include <vector>

enum class SchedulerPolicy { FIFO, FixedBatch, AdaptiveBatch, PriorityBatch };

struct SchedulerConfig {
    SchedulerPolicy policy    = SchedulerPolicy::AdaptiveBatch;
    std::size_t max_batch     = 32;
    std::size_t min_batch     = 1;
    std::chrono::milliseconds max_wait{20};
};

class Scheduler {
public:
    explicit Scheduler(RequestQueue& queue, SchedulerConfig config = {});

    // Blocking: waits until a batch is ready, then returns it.
    // Returns empty immediately if stop() has been called.
    std::vector<Request> next_batch();

    void set_policy(SchedulerPolicy policy);  // thread-safe
    void stop();
    bool is_running() const;

private:
    std::vector<Request> form_fifo_batch();
    std::vector<Request> form_fixed_batch();
    std::vector<Request> form_adaptive_batch();
    std::vector<Request> form_priority_batch();

    RequestQueue& queue_;
    SchedulerConfig config_;
    std::atomic<SchedulerPolicy> policy_;
    std::atomic<bool> running_{true};
};
