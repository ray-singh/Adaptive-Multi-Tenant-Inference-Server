#include "worker_pool.h"
#include "metrics.h"
#include <gtest/gtest.h>
#include <prometheus/registry.h>

TEST(WorkerPool, CallsOnComplete) {
    auto registry = std::make_shared<prometheus::Registry>();
    Metrics metrics(registry);

    std::atomic<int> completed{0};
    WorkerPool pool(1, [](const std::string& p) { return "ok:" + p; }, metrics);

    Request r;
    r.id = 1;
    r.payload = "hello";
    r.priority = Priority::Normal;
    r.enqueue_time = std::chrono::steady_clock::now();
    r.deadline = std::chrono::milliseconds{1000};
    r.on_complete = [&](std::string) { completed++; };

    std::vector<Request> batch;
    batch.push_back(std::move(r));
    pool.submit_batch(std::move(batch));

    EXPECT_EQ(completed.load(), 1);
}
