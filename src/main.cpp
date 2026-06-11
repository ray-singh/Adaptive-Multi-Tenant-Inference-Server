#include "server.h"
#include "continuous_batch_engine.h"
#include "metrics.h"
#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <spdlog/spdlog.h>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <memory>

static Server*               g_server = nullptr;
static ContinuousBatchEngine* g_engine = nullptr;

static void on_signal(int) {
    if (g_engine) g_engine->shutdown();
    if (g_server) g_server->stop();
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
    eng_cfg.n_gpu_layers = -1;  // offload all layers to Metal

    ContinuousBatchEngine engine(model_path, eng_cfg, metrics);

    ServerConfig srv_cfg;
    Server server(srv_cfg, [&engine](Request r) { engine.enqueue(std::move(r)); }, metrics);

    g_engine = &engine;
    g_server = &server;
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    spdlog::info("Send SIGINT or SIGTERM to shut down cleanly");
    server.run();  // blocks until stop()

    engine.shutdown();
    return 0;
}
