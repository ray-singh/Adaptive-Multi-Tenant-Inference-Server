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

TEST(Scheduler, AdaptiveBatchDrainsQueue) {
    RequestQueue q;
    for (uint64_t i = 0; i < 8; ++i) q.push(make_request(i));
    SchedulerConfig cfg;
    cfg.policy    = SchedulerPolicy::AdaptiveBatch;
    cfg.max_batch = 16;
    cfg.max_wait  = std::chrono::milliseconds{50};
    Scheduler s(q, cfg);
    auto batch = s.next_batch();
    EXPECT_GE(batch.size(), 1u);
    EXPECT_LE(batch.size(), 16u);
}
