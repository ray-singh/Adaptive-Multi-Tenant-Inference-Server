#pragma once
#include <chrono>
#include <functional>
#include <string>

enum class Priority { Low = 0, Normal = 1, High = 2 };

struct Request {
    uint64_t id;
    std::string tenant_id;
    std::string payload;
    Priority priority;
    std::chrono::steady_clock::time_point enqueue_time;
    std::chrono::milliseconds deadline; // max allowed wait from enqueue_time

    using Callback = std::function<void(std::string result)>;
    Callback on_complete;
};
