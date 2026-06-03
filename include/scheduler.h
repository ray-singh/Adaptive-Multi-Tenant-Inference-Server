#pragma once
#include "request.h"
#include "request_queue.h"
#include <vector>

enum class SchedulerPolicy { FIFO, FixedBatch, AdaptiveBatch, PriorityBatch };

struct SchedulerConfig {
    SchedulerPolicy policy    = SchedulerPolicy::AdaptiveBatch;
    std::size_t max_batch     = 32;
    std::size_t min_batch     = 1;
    std::chrono::milliseconds max_wait{20}; // flush batch after this even if not full
};

// Pulls requests from the queue and forms batches for dispatch.
class Scheduler {
public:
    explicit Scheduler(RequestQueue& queue, SchedulerConfig config = {});

    // Blocking: waits until a batch is ready, then returns it.
    std::vector<Request> next_batch();

    void set_policy(SchedulerPolicy policy);

private:
    std::vector<Request> form_fifo_batch();
    std::vector<Request> form_fixed_batch();
    std::vector<Request> form_adaptive_batch();
    std::vector<Request> form_priority_batch();

    RequestQueue& queue_;
    SchedulerConfig config_;
};
