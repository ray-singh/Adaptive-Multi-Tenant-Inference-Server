#include "server.h"
#include "grpc_server.h"
#include "continuous_batch_engine.h"
#include "metrics.h"
#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <spdlog/spdlog.h>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <thread>

static Server*                g_server      = nullptr;
static GrpcServer*            g_grpc_server = nullptr;
static ContinuousBatchEngine* g_engine      = nullptr;

static void on_signal(int) {
    if (g_engine)      g_engine->shutdown();
    if (g_grpc_server) g_grpc_server->stop();
    if (g_server)      g_server->stop();
}

int main() {
    auto registry = std::make_shared<prometheus::Registry>();
    Metrics metrics(registry);
    prometheus::Exposer exposer{"0.0.0.0:9090"};
    exposer.RegisterCollectable(registry);

    const char* model_path = std::getenv("MODEL_PATH");
    if (!model_path || !std::filesystem::exists(model_path)) {
        spdlog::error("MODEL_PATH not set or file not found — set MODEL_PATH=/path/to/model.gguf");
        return 1;
    }

    EngineConfig eng_cfg;
    eng_cfg.max_slots    = 16;
    eng_cfg.max_tokens   = 256;
    eng_cfg.n_threads    = 4;
    eng_cfg.n_gpu_layers = -1;  // offload all layers to Metal/CUDA

    ContinuousBatchEngine engine(model_path, eng_cfg, metrics);

    auto enqueue = [&engine](Request r) { engine.enqueue(std::move(r)); };

    ServerConfig srv_cfg;
    if (const char* redis_url = std::getenv("REDIS_URL")) srv_cfg.redis_url = redis_url;
    if (const char* model_name = std::getenv("MODEL_NAME")) srv_cfg.model_name = model_name;

    RateLimiter rate_limiter(srv_cfg.rate_limit_rps, srv_cfg.rate_limit_burst, srv_cfg.redis_url);

    Server     server(srv_cfg, enqueue, metrics, rate_limiter);
    GrpcServer grpc_server("0.0.0.0:50051", enqueue, rate_limiter, metrics);

    g_engine      = &engine;
    g_server      = &server;
    g_grpc_server = &grpc_server;
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // gRPC blocks in its own thread; HTTP blocks on the main thread.
    std::thread grpc_thread([&grpc_server] { grpc_server.run(); });

    spdlog::info("Send SIGINT or SIGTERM to shut down cleanly");
    server.run();  // blocks until stop()

    grpc_server.stop();
    if (grpc_thread.joinable()) grpc_thread.join();
    engine.shutdown();
    return 0;
}
