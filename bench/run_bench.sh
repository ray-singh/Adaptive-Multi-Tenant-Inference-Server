#!/usr/bin/env bash
# Benchmark the continuous batching engine at increasing concurrency levels.
# Requires: wrk (brew install wrk), curl, and the server already running.
#
# Usage:
#   MODEL_PATH=/path/to/model.gguf ./build/inference_server &
#   ./bench/run_bench.sh
#
# Env overrides:
#   SERVER_URL      default: http://localhost:8080
#   BENCH_DURATION  default: 30s
#   BENCH_THREADS   default: 4
set -euo pipefail

SERVER="${SERVER_URL:-http://localhost:8080}"
DURATION="${BENCH_DURATION:-30s}"
THREADS="${BENCH_THREADS:-4}"
TIMEOUT="30s"
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$BENCH_DIR/results"
LUA_SCRIPT="$BENCH_DIR/infer.lua"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

for cmd in wrk curl; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: $cmd not found."; exit 1; }
done

curl -sf "$SERVER/health" >/dev/null || {
    echo "ERROR: Server not reachable at $SERVER. Start it first."
    exit 1
}

mkdir -p "$RESULTS_DIR"

run_concurrency() {
    local conns="$1"
    local out="$RESULTS_DIR/conns${conns}_${RUN_TS}.txt"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Connections: $conns"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    wrk -t"$THREADS" -c"$conns" -d"$DURATION" --timeout "$TIMEOUT" \
        -s "$LUA_SCRIPT" "$SERVER" 2>&1 | tee "$out"
    echo "  (saved → $out)"
}

echo "=== Continuous Batching Engine Benchmark ==="
echo "  Server:   $SERVER"
echo "  Duration: $DURATION per run"
echo "  Threads:  $THREADS"
echo "  Run ID:   $RUN_TS"

for conns in 1 4 8 16; do
    run_concurrency "$conns"
done

echo ""
echo "=== Benchmark complete ==="
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Throughput summary (Req/sec):"
for conns in 1 4 8 16; do
    f="$RESULTS_DIR/conns${conns}_${RUN_TS}.txt"
    if [[ -f "$f" ]]; then
        rps=$(grep "Requests/sec" "$f" | awk '{print $2}')
        printf "  %2d connections   %s req/s\n" "$conns" "$rps"
    fi
done
