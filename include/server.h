#pragma once
#include "request_queue.h"
#include "metrics.h"
#include "httplib.h"
#include <memory>
#include <string>

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port         = 8080;
    int metrics_port = 9090;
};

class Server {
public:
    Server(ServerConfig config, RequestQueue& queue, Metrics& metrics);
    void run();   // blocks
    void stop();

private:
    void register_routes();

    ServerConfig config_;
    RequestQueue& queue_;
    Metrics& metrics_;
    httplib::Server http_;
};
