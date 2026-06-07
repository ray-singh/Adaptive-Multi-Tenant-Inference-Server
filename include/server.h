#pragma once
#include "request_queue.h"
#include "metrics.h"
#include "rate_limiter.h"
#include "httplib.h"
#include <memory>
#include <string>

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port         = 8080;
    int metrics_port = 9090;
    double rate_limit_rps   = 10.0;  // sustained requests/second per tenant
    int    rate_limit_burst = 20;    // max burst tokens per tenant
};

class Server {
public:
    Server(ServerConfig config, RequestQueue& queue, Metrics& metrics);
    void run();   // blocks
    void stop();

private:
    void register_routes();

    ServerConfig  config_;
    RequestQueue& queue_;
    Metrics&      metrics_;
    RateLimiter   rate_limiter_;
    httplib::Server http_;
};
