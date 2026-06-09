-- wrk Lua script: sends realistic multi-tenant inference requests.
-- Usage: wrk -t4 -c16 -d30s -s bench/infer.lua http://localhost:8080

local counter  = 0
local tenants  = {"team-a", "team-b", "team-c", "team-d"}
local payloads = {
    "Summarize the key differences between TCP and UDP.",
    "What is the capital of France?",
    "Explain gradient descent in one sentence.",
    "Write a haiku about network latency.",
    "What is 17 times 23?",
    "Describe the OSI model in two sentences.",
    "What does REST stand for and what are its constraints?",
    "Give one example of a greedy algorithm.",
}

function request()
    counter = counter + 1
    local tenant  = tenants[(counter % #tenants) + 1]
    local payload = payloads[(counter % #payloads) + 1]
    local body    = string.format(
        '{"tenant_id":"%s","payload":"%s","priority":"normal","deadline_ms":5000}',
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
        "\n  Latency p50=%.2fms p95=%.2fms p99=%.2fms\n",
        latency:percentile(50)  / 1000,
        latency:percentile(95)  / 1000,
        latency:percentile(99)  / 1000
    ))
end
