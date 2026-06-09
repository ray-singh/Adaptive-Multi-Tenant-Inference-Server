#include "rate_limiter.h"
#include <gtest/gtest.h>
#include <thread>

TEST(RateLimiter, AllowsUpToBurst) {
    RateLimiter rl(1000.0, 5);
    for (int i = 0; i < 5; ++i)
        EXPECT_TRUE(rl.allow("tenant"));
    EXPECT_FALSE(rl.allow("tenant"));
}

TEST(RateLimiter, IndependentPerTenant) {
    RateLimiter rl(1000.0, 2);
    EXPECT_TRUE(rl.allow("a"));
    EXPECT_TRUE(rl.allow("a"));
    EXPECT_FALSE(rl.allow("a")); // a exhausted
    EXPECT_TRUE(rl.allow("b")); // b has its own bucket, full
    EXPECT_TRUE(rl.allow("b"));
    EXPECT_FALSE(rl.allow("b"));
}

TEST(RateLimiter, NewTenantStartsWithFullBucket) {
    RateLimiter rl(1000.0, 3);
    EXPECT_TRUE(rl.allow("new-tenant"));
    EXPECT_TRUE(rl.allow("new-tenant"));
    EXPECT_TRUE(rl.allow("new-tenant"));
    EXPECT_FALSE(rl.allow("new-tenant"));
}

TEST(RateLimiter, RefillsTokensOverTime) {
    // 1000 rps = 1 token/ms; burst=1 so bucket refills quickly
    RateLimiter rl(1000.0, 1);
    EXPECT_TRUE(rl.allow("t"));
    EXPECT_FALSE(rl.allow("t")); // exhausted
    std::this_thread::sleep_for(std::chrono::milliseconds{5}); // ~5 tokens earned, capped at 1
    EXPECT_TRUE(rl.allow("t")); // refilled
}

TEST(RateLimiter, ZeroRateAlwaysDeniesAfterBurst) {
    // burst=2, rate=0 means no refill — only the initial burst is allowed
    RateLimiter rl(0.0, 2);
    EXPECT_TRUE(rl.allow("z"));
    EXPECT_TRUE(rl.allow("z"));
    EXPECT_FALSE(rl.allow("z"));
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    EXPECT_FALSE(rl.allow("z")); // still denied — rate=0 means no refill
}
