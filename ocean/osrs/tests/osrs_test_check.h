#ifndef OSRS_TEST_CHECK_H
#define OSRS_TEST_CHECK_H

#include <stdio.h>

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(label, cond) do { \
    tests_run++; \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s\n", (label)); } \
} while (0)

#define ASSERT_INT_EQ(label, actual, expected) do { \
    tests_run++; \
    if ((actual) == (expected)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s - got %d, expected %d\n", \
            (label), (actual), (expected)); \
    } \
} while (0)

#define ASSERT_FLOAT_NEAR(label, actual, expected, tol) do { \
    tests_run++; \
    float diff = (float)((actual) - (expected)); \
    if (diff < 0.0f) diff = -diff; \
    if (diff <= (tol)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s - got %.6f, expected %.6f (tol %.6f)\n", \
            (label), (float)(actual), (float)(expected), (float)(tol)); \
    } \
} while (0)

static int osrs_test_summary(void) {
    printf("\n%d/%d tests passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(" (%d failed)\n", tests_failed);
        return 1;
    }
    printf("\n");
    return 0;
}

#endif
