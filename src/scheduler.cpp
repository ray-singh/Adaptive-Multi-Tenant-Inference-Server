#include "scheduler.h"
#include <algorithm>
#include <thread>

Scheduler::Scheduler(RequestQueue& queue, SchedulerConfig config)
    : queue_(queue), config_(config), policy_(config.policy) {}

void Scheduler::set_policy(SchedulerPolicy policy) {
    policy_.store(policy, std::memory_order_relaxed);
}

std::vector<Request> Scheduler::next_batch() {
    if (!running_) return {};
    switch (policy_.load(std::memory_order_relaxed)) {
        case SchedulerPolicy::FIFO:          return form_fifo_batch();
        case SchedulerPolicy::FixedBatch:    return form_fixed_batch();
        case SchedulerPolicy::AdaptiveBatch: return form_adaptive_batch();
        case SchedulerPolicy::PriorityBatch: return form_priority_batch();
    }
    return {};
}

void Scheduler::stop() {
    running_ = false;
}

bool Scheduler::is_running() const {
    return running_;
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

    // Wait for at least one request.
    {
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        Request r;
        if (!queue_.pop(r, remaining)) return batch;
        batch.push_back(std::move(r));
    }

    // Keep collecting until max_batch or the time window exhausts.
    // If the queue already has items, drain them without blocking (high-load path).
    // If the queue is empty, wait up to the remaining window for new arrivals (low-load path).
    // This naturally adapts: under load the backlog is drained instantly; at low concurrency
    // we pause briefly to catch straggler or co-arriving requests before dispatching.
    while (batch.size() < config_.max_batch) {
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (remaining <= std::chrono::milliseconds::zero()) break;

        auto wait = queue_.size() > 0 ? std::chrono::milliseconds{0} : remaining;
        Request r;
        if (!queue_.pop(r, wait)) break;
        batch.push_back(std::move(r));
    }

    return batch;
}

std::vector<Request> Scheduler::form_priority_batch() {
    auto batch = form_adaptive_batch();
    if (batch.size() <= 1) return batch;

    // Sort by priority descending, then by soonest absolute deadline ascending.
    std::sort(batch.begin(), batch.end(), [](const Request& a, const Request& b) {
        if (a.priority != b.priority)
            return static_cast<int>(a.priority) > static_cast<int>(b.priority);
        auto deadline_a = a.enqueue_time + a.deadline;
        auto deadline_b = b.enqueue_time + b.deadline;
        return deadline_a < deadline_b;
    });

    return batch;
}
