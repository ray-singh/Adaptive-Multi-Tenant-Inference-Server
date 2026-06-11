#include "kv_slot_manager.h"

KVSlotManager::KVSlotManager(int max_slots, FreeFn free_fn)
    : max_slots_(max_slots), free_fn_(std::move(free_fn)),
      slots_(max_slots), free_count_(max_slots) {}

int KVSlotManager::acquire(Request req) {
    for (int i = 0; i < max_slots_; ++i) {
        if (!slots_[i].occupied) {
            slots_[i].occupied   = true;
            slots_[i].request    = std::move(req);
            slots_[i].kv_pos     = 0;
            slots_[i].last_token = -1;
            free_count_.fetch_sub(1, std::memory_order_relaxed);
            return i;
        }
    }
    return -1;
}

void KVSlotManager::release(int seq_id) {
    free_fn_(seq_id);
    slots_[seq_id] = Slot{};
    free_count_.fetch_add(1, std::memory_order_relaxed);
}
