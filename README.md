# Adaptive-Multi-Tenant-Inference-Server
## Motivation

Modern ML workloads face a constant tradeoff between latency and throughput:

- Large batches improve hardware utilization but increase queueing delays.
- Small batches reduce latency but waste compute resources.
- Bursty traffic can cause severe p95/p99 latency spikes.

This project explores how scheduling and batching strategies impact inference performance under production-like load.

## Features

### Core Serving Infrastructure
- Asynchronous request handling with `asyncio`
- Concurrent worker execution
- Configurable request queueing
- Model abstraction layer for PyTorch models

### Scheduling & Batching
- Dynamic batch formation
- Queue-aware batch sizing
- Priority-based request scheduling
- Deadline-aware dispatching (planned)

### Observability
- Request throughput tracking
- p50 / p95 / p99 latency metrics
- Queue wait-time monitoring
- Prometheus metrics endpoint
- Grafana dashboards

### Benchmarking
- Synthetic workload generation
- Burst traffic simulation
- Load testing with Locust
- Throughput vs. latency analysis

## Architecture

```text
Clients
   │
   ▼
API Gateway
   │
   ▼
Async Request Queue
   │
   ▼
Adaptive Batch Scheduler
   │
   ▼
Inference Workers
   │
   ▼
Model Execution Engine
   │
   ├── Metrics Exporter
   └── Monitoring Dashboard
```

## Goals

- Sustain high request throughput under heavy load
- Minimize tail latency (p95/p99)
- Improve GPU/CPU utilization through adaptive batching
- Compare scheduling strategies using reproducible benchmarks

## Planned Experiments

| Strategy | Throughput | p95 | p99 |
|----------|------------|-----|-----|
| FIFO | TBD | TBD | TBD |
| Fixed Batch | TBD | TBD | TBD |
| Adaptive Batch | TBD | TBD | TBD |
| Priority Scheduling | TBD | TBD | TBD |
