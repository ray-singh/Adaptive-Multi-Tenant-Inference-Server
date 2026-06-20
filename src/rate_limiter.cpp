#include "rate_limiter.h"
#include <hiredis/hiredis.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>

// Atomic token-bucket in Lua so read-modify-write is safe across nodes.
static const char* kTokenBucketScript = R"lua(
local key     = KEYS[1]
local rate    = tonumber(ARGV[1])
local burst   = tonumber(ARGV[2])
local now_ms  = tonumber(ARGV[3])
local data    = redis.call('HMGET', key, 'tokens', 'last_ms')
local tokens  = tonumber(data[1])
local last_ms = tonumber(data[2])
if not tokens then tokens = burst; last_ms = now_ms end
local elapsed = math.max(0, (now_ms - last_ms) / 1000.0)
tokens = math.min(burst, tokens + elapsed * rate)
if tokens < 1.0 then
    redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_ms', tostring(now_ms))
    redis.call('EXPIRE', key, 86400)
    return 0
end
tokens = tokens - 1.0
redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_ms', tostring(now_ms))
redis.call('EXPIRE', key, 86400)
return 1
)lua";

RateLimiter::RateLimiter(double rate_rps, int burst, std::string redis_url)
    : rate_rps_(rate_rps), burst_(burst) {
    if (redis_url.empty()) return;

    std::string host = "127.0.0.1";
    int port = 6379;

    auto url = redis_url;
    if (url.starts_with("redis://")) url = url.substr(8);
    auto colon = url.rfind(':');
    if (colon != std::string::npos) {
        host = url.substr(0, colon);
        try { port = std::stoi(url.substr(colon + 1)); } catch (...) {}
    } else {
        host = url;
    }

    redis_ = redisConnect(host.c_str(), port);
    if (!redis_ || redis_->err) {
        spdlog::warn("RateLimiter: Redis unavailable ({}), falling back to in-memory",
                     redis_ ? redis_->errstr : "null context");
        if (redis_) { redisFree(redis_); redis_ = nullptr; }
    } else {
        spdlog::info("RateLimiter: connected to Redis at {}:{}", host, port);
    }
}

RateLimiter::~RateLimiter() {
    if (redis_) redisFree(redis_);
}

void RateLimiter::refill(Bucket& b, std::chrono::steady_clock::time_point now) const {
    double elapsed = std::chrono::duration<double>(now - b.last_refill).count();
    b.tokens = std::min(static_cast<double>(burst_), b.tokens + elapsed * rate_rps_);
    b.last_refill = now;
}

bool RateLimiter::allow(const std::string& tenant_id) {
    return redis_ ? allow_redis(tenant_id) : allow_local(tenant_id);
}

bool RateLimiter::allow_redis(const std::string& tenant_id) {
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    std::string key = "ratelimit:" + tenant_id;
    auto* reply = static_cast<redisReply*>(redisCommand(redis_,
        "EVAL %s 1 %s %f %d %lld",
        kTokenBucketScript, key.c_str(), rate_rps_, burst_, now_ms));

    if (!reply) {
        spdlog::warn("RateLimiter: Redis command failed, failing open and dropping connection");
        redisFree(redis_);
        redis_ = nullptr;
        return true;  // fail open to avoid blocking all traffic on Redis outage
    }

    bool allowed = (reply->type == REDIS_REPLY_INTEGER && reply->integer == 1);
    freeReplyObject(reply);
    return allowed;
}

bool RateLimiter::allow_local(const std::string& tenant_id) {
    std::lock_guard lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    auto [it, inserted] = buckets_.try_emplace(tenant_id,
        Bucket{static_cast<double>(burst_), now});
    if (!inserted) refill(it->second, now);
    if (it->second.tokens < 1.0) return false;
    it->second.tokens -= 1.0;
    return true;
}
