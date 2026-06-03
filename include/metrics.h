#pragma once
#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>
#include <memory>

struct Metrics {
    explicit Metrics(std::shared_ptr<prometheus::Registry> registry);

    prometheus::Counter&   requests_total;
    prometheus::Gauge&     queue_depth;
    prometheus::Histogram& latency_seconds;   // end-to-end per request
    prometheus::Histogram& queue_wait_seconds;
    prometheus::Histogram& batch_size;
};
