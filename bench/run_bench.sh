#!/usr/bin/env bash
# Benchmark all four scheduler policies in sequence.
# Requires: wrk (brew install wrk), curl, jq, and the server already running.
#
# Usage:
#   ./bench/run_bench.sh
#   SERVER_URL=http://localhost:8080 BENCH_DURATION=60s ./bench/run_bench.sh
set -euo pipefail

SERVER="${SERVER_URL:-http://localhost:8080}"
DURATION="${BENCH_DURATION:-30s}"
THREADS="${BENCH_THREADS:-4}"
CONNECTIONS="${BENCH_CONNECTIONS:-4}"
TIMEOUT="${BENCH_TIMEOUT:-30s}"
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$BENCH_DIR/results"
LUA_SCRIPT="$BENCH_DIR/infer.lua"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

# Sanity checks
for cmd in wrk curl jq; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: $cmd not found. Install it first."; exit 1; }
done

curl -sf "$SERVER/health" >/dev/null || {
    echo "ERROR: Server not reachable at $SERVER. Start it first."
    exit 1
}

mkdir -p "$RESULTS_DIR"

switch_policy() {
    local policy="$1"
    curl -sf -X POST "$SERVER/admin/policy" \
         -H "Content-Type: application/json" \
         -d "{\"policy\":\"$policy\"}" | jq -r '"  policy set to: " + .policy'
}

run_policy() {
    local policy="$1"
    local out="$RESULTS_DIR/${policy}_${RUN_TS}.txt"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Policy: $policy"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    switch_policy "$policy"
    sleep 1  # let in-flight requests from the previous policy drain

    wrk -t"$THREADS" -c"$CONNECTIONS" -d"$DURATION" --timeout "$TIMEOUT" -s "$LUA_SCRIPT" "$SERVER" \
        2>&1 | tee "$out"
    echo "  (saved → $out)"
}

echo "=== Inference Server Benchmark ==="
echo "  Server:      $SERVER"
echo "  Duration:    $DURATION per policy"
echo "  Threads:     $THREADS"
echo "  Connections: $CONNECTIONS"
echo "  Timeout:     $TIMEOUT"
echo "  Run ID:      $RUN_TS"
echo ""

run_policy fifo
run_policy fixed_batch
run_policy adaptive_batch
run_policy priority_batch

echo ""
echo "=== Benchmark complete ==="
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Quick summary (Req/sec):"
for policy in fifo fixed_batch adaptive_batch priority_batch; do
    f="$RESULTS_DIR/${policy}_${RUN_TS}.txt"
    if [[ -f "$f" ]]; then
        rps=$(grep "Requests/sec" "$f" | awk '{print $2}')
        printf "  %-18s %s req/s\n" "$policy" "$rps"
    fi
done
