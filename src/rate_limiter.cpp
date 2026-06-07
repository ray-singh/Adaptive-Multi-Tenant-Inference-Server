#include "rate_limiter.h"
#include <algorithm>

RateLimiter::RateLimiter(double rate_rps, int burst)
    : rate_rps_(rate_rps), burst_(burst) {}

void RateLimiter::refill(Bucket& b, std::chrono::steady_clock::time_point now) const {
    double elapsed = std::chrono::duration<double>(now - b.last_refill).count();
    b.tokens = std::min(static_cast<double>(burst_), b.tokens + elapsed * rate_rps_);
    b.last_refill  = now;
}

bool RateLimiter::allow(const std::string& tenant_id) {
    std::lock_guard lock(mutex_);
    auto now = std::chrono::steady_clock::now();

    auto [it, inserted] = buckets_.try_emplace(tenant_id,
        Bucket{static_cast<double>(burst_), now});

    if (!inserted)
        refill(it->second, now);

    if (it->second.tokens < 1.0)
        return false;

    it->second.tokens -= 1.0;
    return true;
}
