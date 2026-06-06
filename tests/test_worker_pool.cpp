#include "worker_pool.h"
#include "metrics.h"
#include <gtest/gtest.h>
#include <prometheus/registry.h>
#include <condition_variable>
#include <mutex>

TEST(WorkerPool, CallsOnComplete) {
    auto registry = std::make_shared<prometheus::Registry>();
    Metrics metrics(registry);

    std::mutex mu;
    std::condition_variable cv;
    int completed = 0;

    WorkerPool pool(2, [](const std::string& p) { return "ok:" + p; }, metrics);

    Request r;
    r.id = 1;
    r.payload = "hello";
    r.priority = Priority::Normal;
    r.enqueue_time = std::chrono::steady_clock::now();
    r.deadline = std::chrono::milliseconds{1000};
    r.on_complete = [&](std::string) {
        std::lock_guard lock(mu);
        ++completed;
        cv.notify_one();
    };

    std::vector<Request> batch;
    batch.push_back(std::move(r));
    pool.submit_batch(std::move(batch));

    std::unique_lock lock(mu);
    EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds{2}, [&] { return completed == 1; }));
    EXPECT_EQ(completed, 1);
}

TEST(WorkerPool, ConcurrentBatchProcessed) {
    auto registry = std::make_shared<prometheus::Registry>();
    Metrics metrics(registry);

    constexpr int N = 8;
    std::mutex mu;
    std::condition_variable cv;
    int completed = 0;

    WorkerPool pool(4, [](const std::string& p) { return "r:" + p; }, metrics);

    std::vector<Request> batch;
    for (int i = 0; i < N; ++i) {
        Request r;
        r.id = static_cast<uint64_t>(i);
        r.payload = std::to_string(i);
        r.priority = Priority::Normal;
        r.enqueue_time = std::chrono::steady_clock::now();
        r.deadline = std::chrono::milliseconds{1000};
        r.on_complete = [&](std::string) {
            std::lock_guard lock(mu);
            ++completed;
            cv.notify_one();
        };
        batch.push_back(std::move(r));
    }
    pool.submit_batch(std::move(batch));

    std::unique_lock lock(mu);
    EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds{5}, [&] { return completed == N; }));
    EXPECT_EQ(completed, N);
}
