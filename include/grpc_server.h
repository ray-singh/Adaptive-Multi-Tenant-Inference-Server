#pragma once
#include "request.h"
#include "rate_limiter.h"
#include "metrics.h"
#include <functional>
#include <memory>
#include <string>

// Runs an async gRPC server alongside the HTTP server.
// Both servers share the same enqueue function and rate limiter so rate limits
// are enforced uniformly regardless of which protocol the client uses.
class GrpcServer {
public:
    GrpcServer(std::string address,
               std::function<void(Request)> enqueue_fn,
               RateLimiter& rate_limiter,
               Metrics& metrics);
    ~GrpcServer();

    GrpcServer(const GrpcServer&)            = delete;
    GrpcServer& operator=(const GrpcServer&) = delete;

    void run();   // blocks until shutdown()
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
