#include "server.h"
#include "metrics.h"
#include "rate_limiter.h"
#include "httplib.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <prometheus/registry.h>
#include <thread>
#include <chrono>

using json = nlohmann::json;

// Stub enqueue: immediately calls on_complete so future.get() returns inline.
static void stub_enqueue(Request r) {
    if (r.on_complete) r.on_complete("echo:" + r.payload);
}

class ServerFixture : public ::testing::Test {
protected:
    static constexpr int kPort = 18080;

    void SetUp() override {
        registry_ = std::make_shared<prometheus::Registry>();
        metrics_  = std::make_unique<Metrics>(registry_);

        ServerConfig srv;
        srv.host             = "127.0.0.1";
        srv.port             = kPort;
        srv.rate_limit_rps   = 10000.0;
        srv.rate_limit_burst = 10000;

        rate_limiter_ = std::make_unique<RateLimiter>(srv.rate_limit_rps, srv.rate_limit_burst);
        server_ = std::make_unique<Server>(srv, stub_enqueue, *metrics_, *rate_limiter_);

        srv_thread_ = std::thread([this] { server_->run(); });
        std::this_thread::sleep_for(std::chrono::milliseconds{150});
    }

    void TearDown() override {
        server_->stop();
        if (srv_thread_.joinable()) srv_thread_.join();
    }

    httplib::Client client_{"127.0.0.1", kPort};
    std::shared_ptr<prometheus::Registry> registry_;
    std::unique_ptr<Metrics>      metrics_;
    std::unique_ptr<RateLimiter>  rate_limiter_;
    std::unique_ptr<Server>       server_;
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
    for (auto& prio : {"low", "normal", "high"}) {
        json body = {{"tenant_id", "t1"}, {"payload", "x"}, {"priority", prio}, {"deadline_ms", 5000}};
        auto res = client_.Post("/infer", body.dump(), "application/json");
        ASSERT_TRUE(res) << "priority=" << prio;
        EXPECT_EQ(res->status, 200) << "priority=" << prio;
    }
}

TEST_F(ServerFixture, DefaultsApplyWhenFieldsOmitted) {
    json body = {{"payload", "minimal"}};
    auto res = client_.Post("/infer", body.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
}

TEST_F(ServerFixture, ChatCompletionsReturnsOpenAIShape) {
    json body = {
        {"model", "local-model"},
        {"messages", json::array({
            {{"role", "user"}, {"content", "hello"}}
        })}
    };
    auto res = client_.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    auto resp = json::parse(res->body);
    EXPECT_EQ(resp["object"], "chat.completion");
    EXPECT_TRUE(resp.contains("choices"));
    EXPECT_EQ(resp["choices"][0]["message"]["role"], "assistant");
}

TEST_F(ServerFixture, ChatCompletionsMissingMessagesReturns400) {
    json body = {{"model", "local-model"}};
    auto res = client_.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
}

TEST_F(ServerFixture, ModelsEndpointReturnsList) {
    auto res = client_.Get("/v1/models");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    auto resp = json::parse(res->body);
    EXPECT_EQ(resp["object"], "list");
    EXPECT_FALSE(resp["data"].empty());
}

// ── Rate-limited fixture ──────────────────────────────────────────────────────

class RateLimitedFixture : public ::testing::Test {
protected:
    static constexpr int kPort = 18081;

    void SetUp() override {
        registry_ = std::make_shared<prometheus::Registry>();
        metrics_  = std::make_unique<Metrics>(registry_);

        ServerConfig srv;
        srv.host             = "127.0.0.1";
        srv.port             = kPort;
        srv.rate_limit_rps   = 1.0;
        srv.rate_limit_burst = 1;

        rate_limiter_ = std::make_unique<RateLimiter>(srv.rate_limit_rps, srv.rate_limit_burst);
        server_ = std::make_unique<Server>(srv, stub_enqueue, *metrics_, *rate_limiter_);

        srv_thread_ = std::thread([this] { server_->run(); });
        std::this_thread::sleep_for(std::chrono::milliseconds{150});
    }

    void TearDown() override {
        server_->stop();
        if (srv_thread_.joinable()) srv_thread_.join();
    }

    httplib::Client client_{"127.0.0.1", kPort};
    std::shared_ptr<prometheus::Registry> registry_;
    std::unique_ptr<Metrics>      metrics_;
    std::unique_ptr<RateLimiter>  rate_limiter_;
    std::unique_ptr<Server>       server_;
    std::thread srv_thread_;
};

TEST_F(RateLimitedFixture, FirstRequestAllowedThenDenied) {
    json body = {{"tenant_id", "rl-tenant"}, {"payload", "x"}, {"deadline_ms", 5000}};
    auto res1 = client_.Post("/infer", body.dump(), "application/json");
    ASSERT_TRUE(res1);
    EXPECT_EQ(res1->status, 200);

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
    EXPECT_EQ(res_a->status, 200);

    auto res_b = client_.Post("/infer", body_b.dump(), "application/json");
    ASSERT_TRUE(res_b);
    EXPECT_EQ(res_b->status, 200);
}
