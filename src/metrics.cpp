#include "metrics.h"
#include <prometheus/histogram.h>

Metrics::Metrics(std::shared_ptr<prometheus::Registry> registry)
    : requests_total(
          prometheus::BuildCounter()
              .Name("inference_requests_total")
              .Help("Total inference requests received")
              .Register(*registry)
              .Add({})),
      requests_rate_limited(
          prometheus::BuildCounter()
              .Name("inference_requests_rate_limited_total")
              .Help("Requests rejected due to per-tenant rate limiting")
              .Register(*registry)
              .Add({})),
      queue_depth(
          prometheus::BuildGauge()
              .Name("inference_queue_depth")
              .Help("Current number of requests waiting in queue")
              .Register(*registry)
              .Add({})),
      latency_seconds(
          prometheus::BuildHistogram()
              .Name("inference_latency_seconds")
              .Help("End-to-end request latency")
              .Register(*registry)
              .Add({}, prometheus::Histogram::BucketBoundaries{
                           0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0})),
      queue_wait_seconds(
          prometheus::BuildHistogram()
              .Name("inference_queue_wait_seconds")
              .Help("Time a request spent waiting in the queue")
              .Register(*registry)
              .Add({}, prometheus::Histogram::BucketBoundaries{
                           0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0})),
      batch_size(
          prometheus::BuildHistogram()
              .Name("inference_batch_size")
              .Help("Number of requests per batch dispatched to workers")
              .Register(*registry)
              .Add({}, prometheus::Histogram::BucketBoundaries{
                           1, 2, 4, 8, 16, 32, 64})) {}
