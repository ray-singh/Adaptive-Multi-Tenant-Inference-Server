# Adaptive Multi-Tenant Inference Server

## Motivation

Modern ML workloads face a constant tradeoff between latency and throughput:

- Large batches improve hardware utilization but increase queueing delays.
- Small batches reduce latency but waste compute resources.
- Bursty traffic can cause severe p95/p99 latency spikes.

This project implements a production-style inference server in C++ and Python to explore how scheduling and batching strategies affect performance under realistic load — with precise, low-overhead measurements unclouded by interpreter jitter.

## Stack

| Concern | Library |
|---------|---------|
| HTTP server | [cpp-httplib](https://github.com/yhirose/cpp-httplib) |
| JSON | [nlohmann/json](https://github.com/nlohmann/json) |
| Metrics | [prometheus-cpp](https://github.com/jupp0r/prometheus-cpp) |
| Logging | [spdlog](https://github.com/gabime/spdlog) |
| Testing | [GoogleTest](https://github.com/google/googletest) |
| Build | CMake + vcpkg |
| Observability | Prometheus + Grafana |
| Load testing | [wrk](https://github.com/wg/wrk) |

## Architecture

```text
Clients
   │
   ▼
HTTP Server (cpp-httplib, port 8080)
   │  POST /infer  GET /health
   ▼
RequestQueue (thread-safe priority queue)
   │
   ▼
Scheduler (dispatch thread)
   │  FIFO / FixedBatch / AdaptiveBatch / PriorityBatch
   ▼
WorkerPool (N threads)
   │
   ▼
InferenceFn (stub → real model)
   │
   ├── Metrics (prometheus-cpp, port 9090)
   └── on_complete callback → HTTP response
```

## Project Structure

```
.
├── include/          # Headers: request, queue, scheduler, workers, metrics, server
├── src/              # Implementation + main.cpp
├── tests/            # GoogleTest unit tests
├── bench/            # Load test scripts and results
├── docker/           # docker-compose for Prometheus + Grafana
└── CMakeLists.txt
```

## Building

**Prerequisites:** CMake ≥ 3.20, a C++20 compiler, [vcpkg](https://github.com/microsoft/vcpkg).

```bash
# Install dependencies via vcpkg
vcpkg install cpp-httplib nlohmann-json prometheus-cpp spdlog gtest

# Configure and build
cmake -B build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure

# Start server
./build/inference_server
```

## Observability

```bash
cd docker
docker compose up -d   # Prometheus :9090, Grafana :3000
```

Grafana default credentials: `admin` / `admin`. Add Prometheus as a data source (`http://prometheus:9090`) and import the dashboard from `docker/grafana_dashboard.json` (coming soon).

## API

### `POST /infer`
```json
{
  "tenant_id": "team-a",
  "payload": "your input",
  "deadline_ms": 500
}
```
Response:
```json
{ "result": "result:your input" }
```

### `GET /health`
```json
{ "status": "ok" }
```

## Scheduling Policies

| Policy | Description |
|--------|-------------|
| `FIFO` | One request at a time, in arrival order |
| `FixedBatch` | Wait until `max_batch` requests or `max_wait` timeout |
| `AdaptiveBatch` | Batch size scales with current queue depth |
| `PriorityBatch` | Like AdaptiveBatch but respects per-request priority |

## Planned Experiments

| Strategy | Throughput | p50 | p95 | p99 |
|----------|------------|-----|-----|-----|
| FIFO | TBD | TBD | TBD | TBD |
| Fixed Batch | TBD | TBD | TBD | TBD |
| Adaptive Batch | TBD | TBD | TBD | TBD |
| Priority Scheduling | TBD | TBD | TBD | TBD |
