#include "scheduler.h"
#include <thread>

Scheduler::Scheduler(RequestQueue& queue, SchedulerConfig config)
    : queue_(queue), config_(config) {}

void Scheduler::set_policy(SchedulerPolicy policy) {
    config_.policy = policy;
}

std::vector<Request> Scheduler::next_batch() {
    switch (config_.policy) {
        case SchedulerPolicy::FIFO:         return form_fifo_batch();
        case SchedulerPolicy::FixedBatch:   return form_fixed_batch();
        case SchedulerPolicy::AdaptiveBatch:
        case SchedulerPolicy::PriorityBatch:
        default:                            return form_adaptive_batch();
    }
}

std::vector<Request> Scheduler::form_fifo_batch() {
    std::vector<Request> batch;
    Request r;
    if (queue_.pop(r, config_.max_wait))
        batch.push_back(std::move(r));
    return batch;
}

std::vector<Request> Scheduler::form_fixed_batch() {
    std::vector<Request> batch;
    auto deadline = std::chrono::steady_clock::now() + config_.max_wait;
    while (batch.size() < config_.max_batch) {
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (remaining <= std::chrono::milliseconds::zero()) break;
        Request r;
        if (queue_.pop(r, remaining))
            batch.push_back(std::move(r));
        else
            break;
    }
    return batch;
}

std::vector<Request> Scheduler::form_adaptive_batch() {
    std::vector<Request> batch;
    auto deadline = std::chrono::steady_clock::now() + config_.max_wait;

    // Always wait for at least one request.
    {
        Request r;
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (!queue_.pop(r, remaining)) return batch;
        batch.push_back(std::move(r));
    }

    // Drain additional requests that are already queued, up to max_batch.
    std::size_t target = std::min(config_.max_batch, std::max(config_.min_batch, queue_.size() + 1));
    while (batch.size() < target) {
        Request r;
        if (!queue_.pop(r, std::chrono::milliseconds{0})) break;
        batch.push_back(std::move(r));
    }

    return batch;
}
