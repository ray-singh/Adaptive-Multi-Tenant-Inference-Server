#pragma once
#include "request.h"
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

// Thread-safe priority queue for inference requests.
class RequestQueue {
public:
    void push(Request req);
    bool pop(Request& out, std::chrono::milliseconds timeout);
    std::size_t size() const;
    bool empty() const;

private:
    struct Compare {
        bool operator()(const Request& a, const Request& b) const {
            return static_cast<int>(a.priority) < static_cast<int>(b.priority);
        }
    };

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::priority_queue<Request, std::vector<Request>, Compare> queue_;
};
