#pragma once
#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>
#include <memory>

struct Metrics {
    explicit Metrics(std::shared_ptr<prometheus::Registry> registry);

    prometheus::Counter&   requests_total;
    prometheus::Counter&   requests_rate_limited;
    prometheus::Counter&   requests_deadline_exceeded;
    prometheus::Gauge&     queue_depth;
    prometheus::Gauge&     requests_inflight;
    prometheus::Gauge&     kv_slot_utilization;
    prometheus::Histogram& latency_seconds;
    prometheus::Histogram& queue_wait_seconds;
    prometheus::Histogram& batch_size;
};
