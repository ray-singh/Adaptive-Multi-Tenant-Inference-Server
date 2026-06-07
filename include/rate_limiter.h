#pragma once
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

class RateLimiter {
public:
    // rate_rps: sustained requests/second per tenant; burst: max tokens in bucket.
    RateLimiter(double rate_rps, int burst);

    // Returns true if the request is allowed; false if the tenant is over limit.
    bool allow(const std::string& tenant_id);

private:
    struct Bucket {
        double   tokens;
        std::chrono::steady_clock::time_point last_refill;
    };

    void refill(Bucket& b, std::chrono::steady_clock::time_point now) const;

    double rate_rps_;
    int    burst_;
    std::mutex mutex_;
    std::unordered_map<std::string, Bucket> buckets_;
};
