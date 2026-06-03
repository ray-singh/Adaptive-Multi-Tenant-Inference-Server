#include "request_queue.h"

void RequestQueue::push(Request req) {
    {
        std::lock_guard lock(mutex_);
        queue_.push(std::move(req));
    }
    cv_.notify_one();
}

bool RequestQueue::pop(Request& out, std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); }))
        return false;
    out = std::move(const_cast<Request&>(queue_.top()));
    queue_.pop();
    return true;
}

std::size_t RequestQueue::size() const {
    std::lock_guard lock(mutex_);
    return queue_.size();
}

bool RequestQueue::empty() const {
    std::lock_guard lock(mutex_);
    return queue_.empty();
}
