#include "request_queue.h"
#include <gtest/gtest.h>
#include <thread>

TEST(RequestQueue, PushPop) {
    RequestQueue q;
    Request r;
    r.id       = 1;
    r.priority = Priority::Normal;
    r.enqueue_time = std::chrono::steady_clock::now();
    r.deadline = std::chrono::milliseconds{1000};
    q.push(r);

    Request out;
    EXPECT_TRUE(q.pop(out, std::chrono::milliseconds{100}));
    EXPECT_EQ(out.id, 1u);
}

TEST(RequestQueue, PopTimesOutWhenEmpty) {
    RequestQueue q;
    Request out;
    EXPECT_FALSE(q.pop(out, std::chrono::milliseconds{20}));
}

TEST(RequestQueue, HighPriorityFirst) {
    RequestQueue q;
    auto make = [](uint64_t id, Priority p) {
        Request r;
        r.id = id;
        r.priority = p;
        r.enqueue_time = std::chrono::steady_clock::now();
        r.deadline = std::chrono::milliseconds{1000};
        return r;
    };
    q.push(make(1, Priority::Low));
    q.push(make(2, Priority::High));
    q.push(make(3, Priority::Normal));

    Request out;
    ASSERT_TRUE(q.pop(out, std::chrono::milliseconds{10}));
    EXPECT_EQ(out.priority, Priority::High);
}
