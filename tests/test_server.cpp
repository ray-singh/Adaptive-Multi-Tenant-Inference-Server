#include "server.h"
#include "scheduler.h"
#include "worker_pool.h"
#include "metrics.h"
#include "httplib.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <prometheus/registry.h>
#include <thread>
#include <chrono>

using json = nlohmann::json;

// Shared fixture: full server stack (queue → scheduler → workers → HTTP)
// with a fast stub inference function and a generous rate limit.
class ServerFixture : public ::testing::Test {
protected:
    static constexpr int kPort = 18080;

    void SetUp() override {
        registry_ = std::make_shared<prometheus::Registry>();
        metrics_  = std::make_unique<Metrics>(registry_);

        SchedulerConfig cfg;
        cfg.policy   = SchedulerPolicy::FIFO;
        cfg.max_wait = std::chrono::milliseconds{100};
        scheduler_ = std::make_unique<Scheduler>(queue_, cfg);

        workers_ = std::make_unique<WorkerPool>(
            2,
            [](const std::string& p) { return "echo:" + p; },
            *metrics_);

        dispatch_ = std::thread([this] {
            while (scheduler_->is_running()) {
                auto b = scheduler_->next_batch();
                if (!b.empty()) workers_->submit_batch(std::move(b));
            }
        });

        ServerConfig srv;
        srv.host             = "127.0.0.1";
        srv.port             = kPort;
        srv.rate_limit_rps   = 10000.0; // effectively unlimited for tests
        srv.rate_limit_burst = 10000;
        server_ = std::make_unique<Server>(srv, queue_, *metrics_, *scheduler_);

        srv_thread_ = std::thread([this] { server_->run(); });
        // Give the listening socket time to bind.
        std::this_thread::sleep_for(std::chrono::milliseconds{150});
    }

    void TearDown() override {
        server_->stop();
        if (srv_thread_.joinable()) srv_thread_.join();
        scheduler_->stop();
        if (dispatch_.joinable()) dispatch_.join();
        workers_->shutdown();
    }

    httplib::Client client_{"127.0.0.1", kPort};
    RequestQueue queue_;
    std::shared_ptr<prometheus::Registry> registry_;
    std::unique_ptr<Metrics>   metrics_;
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<WorkerPool> workers_;
    std::unique_ptr<Server>    server_;
    std::thread dispatch_;
    std::thread srv_thread_;
};

TEST_F(ServerFixture, HealthCheck) {
    auto res = client_.Get("/health");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(res->body, R"({"status":"ok"})");
}

TEST_F(ServerFixture, InferReturnsEchoedResult) {
    json body = {{"tenant_id", "t1"}, {"payload", "hello"}, {"deadline_ms", 5000}};
    auto res = client_.Post("/infer", body.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    auto resp = json::parse(res->body);
    EXPECT_EQ(resp["result"], "echo:hello");
}

TEST_F(ServerFixture, InvalidJsonReturns400) {
    auto res = client_.Post("/infer", "not json at all", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
    auto resp = json::parse(res->body);
    EXPECT_EQ(resp["error"], "invalid JSON");
}

TEST_F(ServerFixture, PriorityFieldAccepted) {
    // Verify all priority values parse without error.
    for (auto& prio : {"low", "normal", "high"}) {
        json body = {{"tenant_id", "t1"}, {"payload", "x"}, {"priority", prio}, {"deadline_ms", 5000}};
        auto res = client_.Post("/infer", body.dump(), "application/json");
        ASSERT_TRUE(res) << "priority=" << prio;
        EXPECT_EQ(res->status, 200) << "priority=" << prio;
    }
}

TEST_F(ServerFixture, DefaultsApplyWhenFieldsOmitted) {
    // Only required field is payload; tenant_id/priority/deadline_ms all have defaults.
    json body = {{"payload", "minimal"}};
    auto res = client_.Post("/infer", body.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
}

TEST_F(ServerFixture, AdminPolicyChangeSucceeds) {
    for (auto& policy : {"fifo", "fixed_batch", "adaptive_batch", "priority_batch"}) {
        json body = {{"policy", policy}};
        auto res = client_.Post("/admin/policy", body.dump(), "application/json");
        ASSERT_TRUE(res) << "policy=" << policy;
        EXPECT_EQ(res->status, 200) << "policy=" << policy;
        auto resp = json::parse(res->body);
        EXPECT_EQ(resp["policy"], policy);
    }
}

TEST_F(ServerFixture, AdminPolicyInvalidNameReturns400) {
    json body = {{"policy", "round_robin"}};
    auto res = client_.Post("/admin/policy", body.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(ServerFixture, AdminPolicyBadJsonReturns400) {
    auto res = client_.Post("/admin/policy", "{bad", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

// Separate fixture with a very tight rate limit to test 429 handling.
class RateLimitedFixture : public ::testing::Test {
protected:
    static constexpr int kPort = 18081;

    void SetUp() override {
        registry_ = std::make_shared<prometheus::Registry>();
        metrics_  = std::make_unique<Metrics>(registry_);

        SchedulerConfig cfg;
        cfg.policy   = SchedulerPolicy::FIFO;
        cfg.max_wait = std::chrono::milliseconds{100};
        scheduler_ = std::make_unique<Scheduler>(queue_, cfg);

        workers_ = std::make_unique<WorkerPool>(
            1,
            [](const std::string& p) { return "ok:" + p; },
            *metrics_);

        dispatch_ = std::thread([this] {
            while (scheduler_->is_running()) {
                auto b = scheduler_->next_batch();
                if (!b.empty()) workers_->submit_batch(std::move(b));
            }
        });

        ServerConfig srv;
        srv.host             = "127.0.0.1";
        srv.port             = kPort;
        srv.rate_limit_rps   = 1.0; // 1 request/second
        srv.rate_limit_burst = 1;   // no burst
        server_ = std::make_unique<Server>(srv, queue_, *metrics_, *scheduler_);

        srv_thread_ = std::thread([this] { server_->run(); });
        std::this_thread::sleep_for(std::chrono::milliseconds{150});
    }

    void TearDown() override {
        server_->stop();
        if (srv_thread_.joinable()) srv_thread_.join();
        scheduler_->stop();
        if (dispatch_.joinable()) dispatch_.join();
        workers_->shutdown();
    }

    httplib::Client client_{"127.0.0.1", kPort};
    RequestQueue queue_;
    std::shared_ptr<prometheus::Registry> registry_;
    std::unique_ptr<Metrics>   metrics_;
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<WorkerPool> workers_;
    std::unique_ptr<Server>    server_;
    std::thread dispatch_;
    std::thread srv_thread_;
};

TEST_F(RateLimitedFixture, FirstRequestAllowedThenDenied) {
    json body = {{"tenant_id", "rl-tenant"}, {"payload", "x"}, {"deadline_ms", 5000}};
    // First request consumes the single burst token.
    auto res1 = client_.Post("/infer", body.dump(), "application/json");
    ASSERT_TRUE(res1);
    EXPECT_EQ(res1->status, 200);

    // Second request immediately — bucket is empty, should be rate-limited.
    auto res2 = client_.Post("/infer", body.dump(), "application/json");
    ASSERT_TRUE(res2);
    EXPECT_EQ(res2->status, 429);
    auto resp = json::parse(res2->body);
    EXPECT_EQ(resp["error"], "rate limit exceeded");
}

TEST_F(RateLimitedFixture, DifferentTenantsHaveIndependentLimits) {
    json body_a = {{"tenant_id", "a"}, {"payload", "x"}, {"deadline_ms", 5000}};
    json body_b = {{"tenant_id", "b"}, {"payload", "y"}, {"deadline_ms", 5000}};

    auto res_a = client_.Post("/infer", body_a.dump(), "application/json");
    ASSERT_TRUE(res_a);
    EXPECT_EQ(res_a->status, 200); // a's first request allowed

    auto res_b = client_.Post("/infer", body_b.dump(), "application/json");
    ASSERT_TRUE(res_b);
    EXPECT_EQ(res_b->status, 200); // b is an independent bucket, also allowed
}
