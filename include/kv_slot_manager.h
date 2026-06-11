#pragma once
#include "request.h"
#include <atomic>
#include <functional>
#include <vector>

struct Slot {
    bool    occupied   = false;
    Request request    = {};
    int32_t kv_pos     = 0;  // next KV write position (= prompt tokens + generated tokens so far)
    int32_t last_token = -1; // most recently sampled token, fed as input to the next decode step
};

// Manages a fixed pool of KV cache slots, one per concurrent sequence.
// Each slot index doubles as the llama seq_id passed to llama_decode.
//
// The engine thread is the sole caller of acquire/release/slot().
// free_count() is safe to read from any thread.
class KVSlotManager {
public:
    // free_fn is called with the seq_id when a slot is released.
    // In production this wraps llama_kv_self_seq_rm; in tests it can be a no-op.
    using FreeFn = std::function<void(int seq_id)>;

    KVSlotManager(int max_slots, FreeFn free_fn);

    // Returns a free slot index (= seq_id) and marks it occupied, or -1 if all full.
    int acquire(Request req);

    // Calls free_fn(seq_id) and clears the slot.
    void release(int seq_id);

    Slot&       slot(int seq_id)       { return slots_[seq_id]; }
    const Slot& slot(int seq_id) const { return slots_[seq_id]; }

    int free_count() const { return free_count_.load(std::memory_order_relaxed); }
    int max_slots()  const { return max_slots_; }

private:
    int               max_slots_;
    FreeFn            free_fn_;
    std::vector<Slot> slots_;
    std::atomic<int>  free_count_;
};
