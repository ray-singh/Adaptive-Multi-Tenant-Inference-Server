-- wrk Lua script: multi-tenant inference requests with variable response lengths.
-- Short prompts (yes/no, single word) finish in a few tokens; verbose prompts
-- generate longer responses. The mix is intentional — it stresses the
-- continuous batching engine's ability to keep slots full despite variance.
--
-- Usage: wrk -t4 -c16 -d30s -s bench/infer.lua http://localhost:8080

local counter = 0
local tenants = {"team-a", "team-b", "team-c", "team-d"}

-- Alternating short and long prompts to produce variable response lengths.
local payloads = {
    -- short (a few tokens)
    "What is 7 plus 8?",
    "Is the sky blue? Answer yes or no.",
    "What is the capital of France?",
    "Name one primary color.",
    -- long (paragraph-length responses)
    "Explain how TCP flow control works.",
    "Describe the differences between a process and a thread.",
    "What is gradient descent and why does it work?",
    "Summarize the key ideas behind the transformer architecture.",
}

function request()
    counter = counter + 1
    local tenant  = tenants[(counter % #tenants) + 1]
    local payload = payloads[(counter % #payloads) + 1]
    local body    = string.format(
        '{"tenant_id":"%s","payload":"%s","priority":"normal","deadline_ms":10000}',
        tenant, payload
    )
    return wrk.format("POST", "/infer", {
        ["Content-Type"]   = "application/json",
        ["Content-Length"] = tostring(#body),
    }, body)
end

function response(status, headers, body)
    if status ~= 200 then
        io.write(string.format("[ERR] status=%d body=%s\n", status, body))
    end
end

function done(summary, latency, requests)
    io.write(string.format(
        "\n  Latency p50=%.0fms  p95=%.0fms  p99=%.0fms\n",
        latency:percentile(50)  / 1000,
        latency:percentile(95)  / 1000,
        latency:percentile(99)  / 1000
    ))
end
