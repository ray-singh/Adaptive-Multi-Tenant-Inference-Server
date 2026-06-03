#include "scheduler.h"
#include <gtest/gtest.h>

static Request make_request(uint64_t id, Priority p = Priority::Normal) {
    Request r;
    r.id = id;
    r.priority = p;
    r.enqueue_time = std::chrono::steady_clock::now();
    r.deadline = std::chrono::milliseconds{1000};
    return r;
}

TEST(Scheduler, FIFOReturnsSingleRequest) {
    RequestQueue q;
    q.push(make_request(1));
    SchedulerConfig cfg;
    cfg.policy   = SchedulerPolicy::FIFO;
    cfg.max_wait = std::chrono::milliseconds{50};
    Scheduler s(q, cfg);
    auto batch = s.next_batch();
    EXPECT_EQ(batch.size(), 1u);
}

TEST(Scheduler, PriorityBatchOrdersByPriorityThenDeadline) {
    RequestQueue q;

    // Two High-priority requests: one with a tight deadline, one loose.
    Request tight;
    tight.id = 1; tight.priority = Priority::High;
    tight.enqueue_time = std::chrono::steady_clock::now();
    tight.deadline = std::chrono::milliseconds{50};

    Request loose;
    loose.id = 2; loose.priority = Priority::High;
    loose.enqueue_time = std::chrono::steady_clock::now();
    loose.deadline = std::chrono::milliseconds{5000};

    // One Low-priority request.
    Request low;
    low.id = 3; low.priority = Priority::Low;
    low.enqueue_time = std::chrono::steady_clock::now();
    low.deadline = std::chrono::milliseconds{1000};

    // Push out-of-order to confirm sorting is done by the scheduler, not insertion order.
    q.push(low);
    q.push(loose);
    q.push(tight);

    SchedulerConfig cfg;
    cfg.policy    = SchedulerPolicy::PriorityBatch;
    cfg.max_batch = 16;
    cfg.max_wait  = std::chrono::milliseconds{50};
    Scheduler s(q, cfg);

    auto batch = s.next_batch();
    ASSERT_EQ(batch.size(), 3u);
    EXPECT_EQ(batch[0].id, 1u);  // High + tight deadline first
    EXPECT_EQ(batch[1].id, 2u);  // High + loose deadline second
    EXPECT_EQ(batch[2].id, 3u);  // Low last
}

TEST(Scheduler, AdaptiveBatchDrainsQueue) {
    RequestQueue q;
    for (uint64_t i = 0; i < 8; ++i) q.push(make_request(i));
    SchedulerConfig cfg;
    cfg.policy = SchedulerPolicy::AdaptiveBatch;
    cfg.max_batch = 16;
    cfg.max_wait = std::chrono::milliseconds{50};
    Scheduler s(q, cfg);
    auto batch = s.next_batch();
    EXPECT_GE(batch.size(), 1u);
    EXPECT_LE(batch.size(), 16u);
}
