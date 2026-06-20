#include "grpc_server.h"
#include "inference.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <spdlog/spdlog.h>
#include <atomic>
#include <future>
#include <memory>

static std::atomic<uint64_t> grpc_request_id_counter{0};

// ── Service implementation ────────────────────────────────────────────────────

class InferenceServiceImpl final : public inference::InferenceService::Service {
public:
    InferenceServiceImpl(std::function<void(Request)> enqueue_fn,
                         RateLimiter& rate_limiter,
                         Metrics& metrics)
        : enqueue_fn_(std::move(enqueue_fn))
        , rate_limiter_(rate_limiter)
        , metrics_(metrics) {}

    grpc::Status Infer(grpc::ServerContext* /*ctx*/,
                       const inference::InferRequest* req,
                       inference::InferResponse* resp) override {
        std::string tenant_id = req->tenant_id().empty() ? "default" : req->tenant_id();

        if (!rate_limiter_.allow(tenant_id)) {
            metrics_.requests_rate_limited.Increment();
            return {grpc::StatusCode::RESOURCE_EXHAUSTED, "rate limit exceeded"};
        }

        Priority prio = Priority::Normal;
        if (req->priority() == "high") prio = Priority::High;
        else if (req->priority() == "low") prio = Priority::Low;

        Request r;
        r.id             = ++grpc_request_id_counter;
        r.tenant_id      = tenant_id;
        r.payload        = req->payload();
        r.priority       = prio;
        r.enqueue_time   = std::chrono::steady_clock::now();
        r.deadline       = std::chrono::milliseconds{req->deadline_ms() > 0 ? req->deadline_ms() : 5000};
        r.max_new_tokens = req->max_new_tokens();

        auto result = dispatch(std::move(r));
        resp->set_result(result);
        resp->set_request_id(r.id);
        return grpc::Status::OK;
    }

    grpc::Status ChatComplete(grpc::ServerContext* /*ctx*/,
                              const inference::ChatRequest* req,
                              inference::ChatResponse* resp) override {
        std::string tenant_id = req->tenant_id().empty() ? "default" : req->tenant_id();

        if (!rate_limiter_.allow(tenant_id)) {
            metrics_.requests_rate_limited.Increment();
            return {grpc::StatusCode::RESOURCE_EXHAUSTED, "rate limit exceeded"};
        }

        std::string prompt;
        for (const auto& msg : req->messages()) {
            prompt += "<|im_start|>" + msg.role() + "\n" + msg.content() + "<|im_end|>\n";
        }
        prompt += "<|im_start|>assistant\n";

        Request r;
        r.id             = ++grpc_request_id_counter;
        r.tenant_id      = tenant_id;
        r.payload        = prompt;
        r.priority       = Priority::Normal;
        r.enqueue_time   = std::chrono::steady_clock::now();
        r.deadline       = std::chrono::milliseconds{req->deadline_ms() > 0 ? req->deadline_ms() : 30000};
        r.max_new_tokens = req->max_tokens();

        uint64_t req_id = r.id;
        std::string content = dispatch(std::move(r));

        resp->set_id("chatcmpl-grpc-" + std::to_string(req_id));
        resp->set_model(req->model().empty() ? "local-model" : req->model());
        auto* choice = resp->add_choices();
        choice->set_index(0);
        choice->mutable_message()->set_role("assistant");
        choice->mutable_message()->set_content(content);
        choice->set_finish_reason("stop");

        return grpc::Status::OK;
    }

private:
    std::string dispatch(Request r) {
        std::promise<std::string> promise;
        auto future = promise.get_future();
        r.on_complete = [p = std::make_shared<std::promise<std::string>>(
                             std::move(promise))](std::string result) mutable {
            p->set_value(std::move(result));
        };
        metrics_.queue_depth.Increment();
        enqueue_fn_(std::move(r));
        std::string result = future.get();
        metrics_.queue_depth.Decrement();
        return result;
    }

    std::function<void(Request)> enqueue_fn_;
    RateLimiter& rate_limiter_;
    Metrics&     metrics_;
};

// ── GrpcServer pimpl ─────────────────────────────────────────────────────────

struct GrpcServer::Impl {
    InferenceServiceImpl service;
    std::unique_ptr<grpc::Server> server;

    Impl(std::string address,
         std::function<void(Request)> enqueue_fn,
         RateLimiter& rate_limiter,
         Metrics& metrics)
        : service(std::move(enqueue_fn), rate_limiter, metrics) {
        grpc::ServerBuilder builder;
        builder.AddListeningPort(address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service);
        server = builder.BuildAndStart();
        if (server)
            spdlog::info("GrpcServer: listening on {}", address);
        else
            spdlog::error("GrpcServer: failed to start on {}", address);
    }
};

GrpcServer::GrpcServer(std::string address,
                        std::function<void(Request)> enqueue_fn,
                        RateLimiter& rate_limiter,
                        Metrics& metrics)
    : impl_(std::make_unique<Impl>(std::move(address),
                                   std::move(enqueue_fn),
                                   rate_limiter,
                                   metrics)) {}

GrpcServer::~GrpcServer() { stop(); }

void GrpcServer::run() {
    if (impl_->server) impl_->server->Wait();
}

void GrpcServer::stop() {
    if (impl_ && impl_->server) impl_->server->Shutdown();
}
