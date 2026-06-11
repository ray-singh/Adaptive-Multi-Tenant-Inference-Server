#include "kv_slot_manager.h"
#include <gtest/gtest.h>

static Request make_request(uint64_t id, Priority prio = Priority::Normal) {
    Request r;
    r.id           = id;
    r.tenant_id    = "t1";
    r.payload      = "hello";
    r.priority     = prio;
    r.enqueue_time = std::chrono::steady_clock::now();
    r.deadline     = std::chrono::milliseconds{5000};
    return r;
}

TEST(KVSlotManager, InitiallyAllFree) {
    KVSlotManager mgr(4, [](int) {});
    EXPECT_EQ(mgr.free_count(), 4);
    EXPECT_EQ(mgr.max_slots(), 4);
}

TEST(KVSlotManager, AcquireReturnsSlotIndex) {
    KVSlotManager mgr(4, [](int) {});
    int id = mgr.acquire(make_request(1));
    EXPECT_GE(id, 0);
    EXPECT_LT(id, 4);
    EXPECT_EQ(mgr.free_count(), 3);
    EXPECT_TRUE(mgr.slot(id).occupied);
    EXPECT_EQ(mgr.slot(id).request.id, 1u);
}

TEST(KVSlotManager, AcquireReturnsMinus1WhenFull) {
    KVSlotManager mgr(2, [](int) {});
    EXPECT_GE(mgr.acquire(make_request(1)), 0);
    EXPECT_GE(mgr.acquire(make_request(2)), 0);
    EXPECT_EQ(mgr.acquire(make_request(3)), -1);
    EXPECT_EQ(mgr.free_count(), 0);
}

TEST(KVSlotManager, ReleaseFiresFreeFnAndClearsSlot) {
    int freed_id = -1;
    KVSlotManager mgr(2, [&](int id) { freed_id = id; });

    int slot = mgr.acquire(make_request(1));
    mgr.release(slot);

    EXPECT_EQ(freed_id, slot);
    EXPECT_FALSE(mgr.slot(slot).occupied);
    EXPECT_EQ(mgr.free_count(), 2);
}

TEST(KVSlotManager, ReleasedSlotIsReacquirable) {
    KVSlotManager mgr(1, [](int) {});
    int first = mgr.acquire(make_request(1));
    EXPECT_GE(first, 0);
    mgr.release(first);
    int second = mgr.acquire(make_request(2));
    EXPECT_GE(second, 0);
    EXPECT_EQ(mgr.slot(second).request.id, 2u);
}

TEST(KVSlotManager, SlotStateIsAccessible) {
    KVSlotManager mgr(4, [](int) {});
    int id = mgr.acquire(make_request(42));
    mgr.slot(id).kv_pos     = 10;
    mgr.slot(id).last_token = 99;
    EXPECT_EQ(mgr.slot(id).kv_pos,     10);
    EXPECT_EQ(mgr.slot(id).last_token, 99);
}
