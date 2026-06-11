# Adaptive Multi-Tenant Inference Server

## Problem

Static batching wastes GPU time. When you run a batch of N sequences, short responses finish early but their KV cache slots stay reserved until the slowest sequence in the batch completes. Under variable response lengths — the normal case in production — GPU utilization drops significantly while you wait.

Continuous batching fixes this problem. The moment a sequence finishes, its KV cache slots are freed and the next waiting request is admitted immediately. The active set is a sliding window rather than a discrete batch, so the GPU stays utilized regardless of response length variance. This is the core innovation behind vLLM's original performance gains.

This project implements continuous batching from scratch in C++ against llama.cpp, with a multi-tenant HTTP frontend, per-tenant rate limiting, priority scheduling, and deadline tracking. The goal is a system where the scheduling layer and the KV cache memory layer are the same layer — admission decisions are memory allocation decisions.

## Architecture

```text
Clients
   │
   ▼
HTTP Server (cpp-httplib, port 8080)
   │  POST /infer  GET /health
   ▼
RateLimiter (token bucket, per tenant)
   │
   ▼
ContinuousBatchEngine
   │  ┌─────────────────────────────────────────────┐
   │  │  Admission loop (priority + deadline order) │
   │  │  KVSlotManager (seq_id ↔ KV cell tracking)  │
   │  │  Decode loop (llama_decode per step)        │
   │  │  Per-sequence sampling + EOS detection      │
   │  └─────────────────────────────────────────────┘
   │
   ├── Metrics (prometheus-cpp, port 9090)
   └── on_complete callback → HTTP response
```

## How it works

The engine runs a single decode thread. On each iteration:

1. **Admit**: any free KV slots are filled from the waiting queue, highest priority / soonest deadline first.
2. **Decode**: one `llama_decode` call across all active sequences simultaneously.
3. **Sample**: one token is sampled per active sequence.
4. **Retire**: sequences that hit EOS or their deadline fire their callback and release their KV slots via `llama_kv_self_seq_rm`, making room for the next admission.

The KV cache is pre-allocated at startup (`n_ctx = max_slots × max_seq_len`). Each sequence is tagged with a `seq_id`; retiring a sequence untags its cells without touching any other sequence's state.

## Stack

| Concern | Library |
|---------|---------|
| HTTP server | [cpp-httplib](https://github.com/yhirose/cpp-httplib) |
| LLM inference | [llama.cpp](https://github.com/ggerganov/llama.cpp) |
| JSON | [nlohmann/json](https://github.com/nlohmann/json) |
| Metrics | [prometheus-cpp](https://github.com/jupp0r/prometheus-cpp) |
| Logging | [spdlog](https://github.com/gabime/spdlog) |
| Testing | [GoogleTest](https://github.com/google/googletest) |
| Observability | Prometheus + Grafana |
| Load testing | [wrk](https://github.com/wg/wrk) |

## Project Structure

```
.
├── include/          # Headers: request, engine, slot manager, metrics, rate limiter, server
├── src/              # Implementation + main.cpp
├── tests/            # GoogleTest unit tests (server, rate limiter, KV slot manager)
├── bench/            # wrk load test scripts and results
├── docker/           # docker-compose for Prometheus + Grafana
└── CMakeLists.txt
```

## Building

**Prerequisites:** CMake ≥ 3.20, C++20 compiler, Homebrew (macOS).

```bash
# Install dependencies
brew install nlohmann-json spdlog googletest prometheus-cpp

# Configure and build (llama.cpp fetched automatically via FetchContent)
cmake -B build
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure

# Start server (MODEL_PATH is required)
MODEL_PATH=/path/to/model.gguf ./build/inference_server
```

## Observability

```bash
cd docker
docker compose up -d   # Prometheus :9090, Grafana :3000
```

Grafana default credentials: `admin` / `admin`. The Prometheus datasource and dashboard are provisioned automatically.

Key metrics:
- `inference_kv_slot_utilization` — fraction of KV slots in use; the primary signal for engine efficiency
- `inference_latency_seconds` — end-to-end request latency
- `inference_queue_wait_seconds` — time waiting for a free slot
- `inference_batch_size` — active sequences per decode step
- `inference_requests_deadline_exceeded_total` — requests that expired before a slot was available

## API

### `POST /infer`
```json
{
  "tenant_id": "team-a",
  "payload": "your prompt",
  "priority": "high",
  "deadline_ms": 5000
}
```
Response:
```json
{ "result": "..." }
```

### `GET /health`
```json
{ "status": "ok" }
```

## Benchmarking

```bash
# Start the server
MODEL_PATH=/path/to/model.gguf ./build/inference_server &

# Run the concurrency sweep (1, 4, 8, 16 connections × 30s each)
./bench/run_bench.sh
```

The bench script (`bench/infer.lua`) mixes short prompts (a few tokens) and long prompts (paragraph responses) to produce variable response lengths — the workload where continuous batching's advantage over static batching is most pronounced.

### Static batching baseline

Measured on `SmolLM2-135M-Instruct-Q4_K_M.gguf`, Apple Silicon (Metal/MPS), before the continuous batching rewrite. Included as a reference point.

| Concurrency | Policy | Req/s | p50 | p99 |
|-------------|--------|-------|-----|-----|
| 4 conns | Fixed Batch | 2.63 | 1515 ms | 2748 ms |
| 16 conns | Fixed Batch | **3.22** | 4390 ms | 5386 ms |

Under static batching, throughput plateaus at 16 connections because short responses waste their slot waiting for the slowest sequence in the batch to finish. The continuous batching engine eliminates that dead time.
