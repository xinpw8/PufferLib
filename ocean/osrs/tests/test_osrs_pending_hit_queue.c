#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ocean/osrs/osrs_encounter.h"

#include "ocean/osrs/tests/osrs_test_check.h"

static void assert_child_aborts(const char* label, void (*fn)(void)) {
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) {
        fn();
        _exit(0);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    tests_run++;
    if (WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) != 0)) {
        tests_passed++;
    } else {
        tests_failed++;
        printf("  FAIL: %s - child returned successfully\n", label);
    }
}

static EncounterPendingHit make_hit(int damage, int ticks_remaining) {
    return (EncounterPendingHit){
        .damage = damage,
        .ticks_remaining = ticks_remaining,
        .attack_style = ATTACK_STYLE_MAGIC,
        .spell_type = ENCOUNTER_SPELL_BLOOD,
        .hit_success = damage > 0,
    };
}

static void test_push_preserves_insertion_order(void) {
    printf("--- push preserves insertion order ---\n");

    EncounterPendingHitQueue q = {0};
    EncounterPendingHit* first = encounter_pending_hit_queue_push(
        &q, make_hit(11, 4), "test", 7, 1, 2);
    EncounterPendingHit* second = encounter_pending_hit_queue_push(
        &q, make_hit(19, 2), "test", 7, 1, 2);

    ASSERT_INT_EQ("count after two pushes", q.count, 2);
    ASSERT_INT_EQ("first hit active", first->active, 1);
    ASSERT_INT_EQ("second hit active", second->active, 1);
    ASSERT_INT_EQ("first hit remains first", q.hits[0].damage, 11);
    ASSERT_INT_EQ("second hit remains second", q.hits[1].damage, 19);
}

static void test_remove_compacts_and_clears_tail(void) {
    printf("--- remove compacts and clears tail ---\n");

    EncounterPendingHitQueue q = {0};
    encounter_pending_hit_queue_push(&q, make_hit(1, 5), "test", 7, 1, 2);
    encounter_pending_hit_queue_push(&q, make_hit(2, 4), "test", 7, 1, 2);
    encounter_pending_hit_queue_push(&q, make_hit(3, 3), "test", 7, 1, 2);

    encounter_pending_hit_queue_remove(&q, 1, "test");

    ASSERT_INT_EQ("count after remove", q.count, 2);
    ASSERT_INT_EQ("first slot kept", q.hits[0].damage, 1);
    ASSERT_INT_EQ("third slot compacted", q.hits[1].damage, 3);
    ASSERT_INT_EQ("tail active cleared", q.hits[2].active, 0);
    ASSERT_INT_EQ("tail damage cleared", q.hits[2].damage, 0);
}

static void test_earliest_and_damage_sum(void) {
    printf("--- earliest and damage sum ---\n");

    EncounterPendingHitQueue q = {0};
    encounter_pending_hit_queue_push(&q, make_hit(5, 7), "test", 7, 1, 2);
    encounter_pending_hit_queue_push(&q, make_hit(9, 2), "test", 7, 1, 2);
    encounter_pending_hit_queue_push(&q, make_hit(13, 4), "test", 7, 1, 2);
    q.hits[2].active = 0;

    const EncounterPendingHit* earliest = encounter_pending_hit_queue_earliest(&q);

    ASSERT_INT_EQ("earliest hit is second slot", earliest == &q.hits[1], 1);
    ASSERT_INT_EQ("damage sum ignores inactive hits",
        encounter_pending_hit_queue_damage_sum(&q), 14);
}

static void test_clear_zeroes_queue(void) {
    printf("--- clear zeroes queue ---\n");

    EncounterPendingHitQueue q = {0};
    encounter_pending_hit_queue_push(&q, make_hit(5, 7), "test", 7, 1, 2);
    encounter_pending_hit_queue_push(&q, make_hit(9, 2), "test", 7, 1, 2);

    encounter_pending_hit_queue_clear(&q);

    ASSERT_INT_EQ("count cleared", q.count, 0);
    ASSERT_INT_EQ("first slot cleared", q.hits[0].active, 0);
    ASSERT_INT_EQ("second slot cleared", q.hits[1].damage, 0);
}

static void child_queue_overflow(void) {
    EncounterPendingHitQueue q = {0};
    for (int i = 0; i < ENCOUNTER_MAX_PENDING_HITS; i++)
        encounter_pending_hit_queue_push(&q, make_hit(i + 1, i + 1), "test", 7, 1, 2);
    encounter_pending_hit_queue_push(&q, make_hit(99, 1), "test", 7, 1, 2);
}

static void test_overflow_aborts(void) {
    printf("--- overflow aborts ---\n");

    assert_child_aborts("shared pending-hit overflow aborts", child_queue_overflow);
}

int main(void) {
    test_push_preserves_insertion_order();
    test_remove_compacts_and_clears_tail();
    test_earliest_and_damage_sum();
    test_clear_zeroes_queue();
    test_overflow_aborts();

    return osrs_test_summary();
}
