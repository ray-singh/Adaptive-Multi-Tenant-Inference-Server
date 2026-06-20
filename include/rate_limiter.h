#pragma once
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

struct redisContext;  // forward-declare; hiredis included only in the .cpp

class RateLimiter {
public:
    // redis_url: "redis://host:port" or "host:port". Empty = in-memory only.
    RateLimiter(double rate_rps, int burst, std::string redis_url = "");
    ~RateLimiter();

    RateLimiter(const RateLimiter&)            = delete;
    RateLimiter& operator=(const RateLimiter&) = delete;

    // Thread-safe. Returns true if the request is allowed.
    bool allow(const std::string& tenant_id);

private:
    struct Bucket {
        double tokens;
        std::chrono::steady_clock::time_point last_refill;
    };

    void refill(Bucket& b, std::chrono::steady_clock::time_point now) const;
    bool allow_redis(const std::string& tenant_id);
    bool allow_local(const std::string& tenant_id);

    double rate_rps_;
    int    burst_;

    redisContext* redis_ = nullptr;

    std::mutex mutex_;
    std::unordered_map<std::string, Bucket> buckets_;
};
