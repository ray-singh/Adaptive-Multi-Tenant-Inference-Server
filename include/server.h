#pragma once
#include "request.h"
#include "metrics.h"
#include "rate_limiter.h"
#include "httplib.h"
#include <functional>
#include <string>

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port         = 8080;
    int metrics_port = 9090;
    double rate_limit_rps   = 10.0;
    int    rate_limit_burst = 20;
    std::string redis_url   = "";   // empty = in-memory rate limiting
    std::string model_name  = "local-model";
};

class Server {
public:
    // rate_limiter lifetime must exceed Server's.
    Server(ServerConfig config,
           std::function<void(Request)> enqueue_fn,
           Metrics& metrics,
           RateLimiter& rate_limiter);
    void run();   // blocks
    void stop();

private:
    void register_routes();

    ServerConfig  config_;
    std::function<void(Request)> enqueue_fn_;
    Metrics&      metrics_;
    RateLimiter&  rate_limiter_;
    httplib::Server http_;
};
