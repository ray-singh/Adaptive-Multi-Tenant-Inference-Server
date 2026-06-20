#include "server.h"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <atomic>
#include <future>

using json = nlohmann::json;

static std::atomic<uint64_t> request_id_counter{0};

Server::Server(ServerConfig config,
               std::function<void(Request)> enqueue_fn,
               Metrics& metrics,
               RateLimiter& rate_limiter)
    : config_(config)
    , enqueue_fn_(std::move(enqueue_fn))
    , metrics_(metrics)
    , rate_limiter_(rate_limiter) {
    register_routes();
}

// Enqueue a request and block until on_complete fires.
static std::string dispatch(std::function<void(Request)>& enqueue_fn,
                            Metrics& metrics,
                            Request r) {
    std::promise<std::string> promise;
    auto future = promise.get_future();
    r.on_complete = [p = std::make_shared<std::promise<std::string>>(
                         std::move(promise))](std::string result) mutable {
        p->set_value(std::move(result));
    };
    metrics.queue_depth.Increment();
    enqueue_fn(std::move(r));
    std::string result = future.get();
    metrics.queue_depth.Decrement();
    return result;
}

static Priority parse_priority(const std::string& s) {
    if (s == "high") return Priority::High;
    if (s == "low")  return Priority::Low;
    return Priority::Normal;
}

// Converts an OpenAI messages array into a ChatML-formatted prompt string.
static std::string build_chatml(const json& messages) {
    std::string prompt;
    for (const auto& msg : messages) {
        prompt += "<|im_start|>" + msg.value("role", "user") + "\n"
                + msg.value("content", "") + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

void Server::register_routes() {
    // ── Original inference endpoint ───────────────────────────────────────────
    http_.Post("/infer", [this](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
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

        Request r;
        r.id             = ++request_id_counter;
        r.tenant_id      = tenant_id;
        r.payload        = body.value("payload", "");
        r.priority       = parse_priority(body.value("priority", "normal"));
        r.enqueue_time   = std::chrono::steady_clock::now();
        r.deadline       = std::chrono::milliseconds{body.value("deadline_ms", 5000)};
        r.max_new_tokens = body.value("max_new_tokens", 0);

        metrics_.requests_total.Increment();
        res.set_content(json{{"result", dispatch(enqueue_fn_, metrics_, std::move(r))}}.dump(),
                        "application/json");
    });

    // ── OpenAI: POST /v1/chat/completions ────────────────────────────────────
    http_.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content(json{{"error", {{"message", "invalid JSON"},
                                            {"type",    "invalid_request_error"}}}}.dump(),
                            "application/json");
            return;
        }

        // Tenant from Authorization: Bearer <key> or X-Tenant-ID header.
        std::string tenant_id = "default";
        if (req.has_header("Authorization")) {
            auto auth = req.get_header_value("Authorization");
            if (auth.starts_with("Bearer ")) tenant_id = auth.substr(7);
        } else if (req.has_header("X-Tenant-ID")) {
            tenant_id = req.get_header_value("X-Tenant-ID");
        }

        if (!rate_limiter_.allow(tenant_id)) {
            metrics_.requests_rate_limited.Increment();
            res.status = 429;
            res.set_content(json{{"error", {{"message", "Rate limit exceeded"},
                                            {"type",    "rate_limit_error"}}}}.dump(),
                            "application/json");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(json{{"error", {{"message", "'messages' array is required"},
                                            {"type",    "invalid_request_error"}}}}.dump(),
                            "application/json");
            return;
        }

        std::string model = body.value("model", config_.model_name);

        Request r;
        r.id             = ++request_id_counter;
        r.tenant_id      = tenant_id;
        r.payload        = build_chatml(body["messages"]);
        r.priority       = Priority::Normal;
        r.enqueue_time   = std::chrono::steady_clock::now();
        r.deadline       = std::chrono::milliseconds{body.value("timeout_ms", 30000)};
        r.max_new_tokens = body.value("max_tokens", 0);

        uint64_t req_id = r.id;
        metrics_.requests_total.Increment();
        std::string content = dispatch(enqueue_fn_, metrics_, std::move(r));

        json response = {
            {"id",      "chatcmpl-" + std::to_string(req_id)},
            {"object",  "chat.completion"},
            {"created", std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count()},
            {"model",   model},
            {"choices", json::array({json{
                {"index",         0},
                {"message",       {{"role", "assistant"}, {"content", content}}},
                {"finish_reason", "stop"}
            }})},
            {"usage", {{"prompt_tokens", 0}, {"completion_tokens", 0}, {"total_tokens", 0}}}
        };
        res.set_content(response.dump(), "application/json");
    });

    // ── OpenAI: GET /v1/models ────────────────────────────────────────────────
    http_.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        json response = {
            {"object", "list"},
            {"data", json::array({json{
                {"id",       config_.model_name},
                {"object",   "model"},
                {"created",  0},
                {"owned_by", "local"}
            }})}
        };
        res.set_content(response.dump(), "application/json");
    });

    // ── Health ────────────────────────────────────────────────────────────────
    http_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
}

void Server::run() {
    spdlog::info("HTTP server listening on {}:{}", config_.host, config_.port);
    http_.listen(config_.host.c_str(), config_.port);
}

void Server::stop() { http_.stop(); }
