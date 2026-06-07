#include "server.h"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <atomic>
#include <future>

using json = nlohmann::json;

static std::atomic<uint64_t> request_id_counter{0};

Server::Server(ServerConfig config, RequestQueue& queue, Metrics& metrics)
    : config_(config), queue_(queue), metrics_(metrics),
      rate_limiter_(config.rate_limit_rps, config.rate_limit_burst) {
    register_routes();
}

void Server::register_routes() {
    // POST /infer — enqueue a request, block until result
    http_.Post("/infer", [this](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid JSON"})", "application/json");
            return;
        }

        auto tenant_id = body.value("tenant_id", "default");
        if (!rate_limiter_.allow(tenant_id)) {
            metrics_.requests_rate_limited.Increment();
            res.status = 429;
            res.set_content(R"({"error":"rate limit exceeded"})", "application/json");
            return;
        }

        std::promise<std::string> promise;
        auto future = promise.get_future();

        auto prio_str = body.value("priority", "normal");
        Priority prio = Priority::Normal;
        if (prio_str == "high")      prio = Priority::High;
        else if (prio_str == "low")  prio = Priority::Low;

        Request r;
        r.id           = ++request_id_counter;
        r.tenant_id    = tenant_id;
        r.payload      = body.value("payload", "");
        r.priority     = prio;
        r.enqueue_time = std::chrono::steady_clock::now();
        r.deadline     = std::chrono::milliseconds{body.value("deadline_ms", 5000)};
        r.on_complete  = [p = std::make_shared<std::promise<std::string>>(
                              std::move(promise))](std::string result) mutable {
            p->set_value(std::move(result));
        };

        metrics_.queue_depth.Increment();
        queue_.push(std::move(r));

        std::string result = future.get();
        metrics_.queue_depth.Decrement();

        res.set_content(json{{"result", result}}.dump(), "application/json");
    });

    // GET /health
    http_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
}

void Server::run() {
    spdlog::info("Server listening on {}:{}", config_.host, config_.port);
    http_.listen(config_.host.c_str(), config_.port);
}

void Server::stop() {
    http_.stop();
}
