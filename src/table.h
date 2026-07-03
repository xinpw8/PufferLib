#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PUF_TABLE_MAX_COLS 256

typedef struct {
    char name[64];
    int rows;
    int cols;
    char* labels[PUF_TABLE_MAX_COLS];
    float* values;
} Table;

static char* table_strdup(const char* s) {
    size_t n = strlen(s) + 1;
    char* out = (char*)malloc(n);
    if (!out) {
        perror("malloc");
        exit(1);
    }
    memcpy(out, s, n);
    return out;
}

static int table_col(Table* table, const char* label) {
    for (int i = 0; i < table->cols; i++) {
        if (strcmp(table->labels[i], label) == 0) {
            return i;
        }
    }
    return -1;
}

static int table_add_col(Table* table, const char* label) {
    if (table->cols >= PUF_TABLE_MAX_COLS) {
        fprintf(stderr, "table %s has too many columns\n", table->name);
        exit(1);
    }

    int col = table->cols++;
    table->labels[col] = table_strdup(label);
    float* old = table->values;
    table->values = (float*)calloc((size_t)table->rows * (size_t)table->cols, sizeof(float));
    if (table->rows > 0 && !table->values) {
        perror("realloc");
        exit(1);
    }
    for (int r = 0; r < table->rows; r++) {
        for (int c = 0; c < table->cols - 1; c++) {
            table->values[r * table->cols + c] = old[r * (table->cols - 1) + c];
        }
    }
    free(old);
    return col;
}

static int table_require_col(Table* table, const char* label) {
    int col = table_col(table, label);
    if (col >= 0) {
        return col;
    }
    fprintf(stderr, "table %s missing column %s\n", table->name, label);
    exit(1);
}

static int table_ensure_col(Table* table, const char* label) {
    int col = table_col(table, label);
    if (col >= 0) {
        return col;
    }
    return table_add_col(table, label);
}

static void table_resize_rows(Table* table, int rows) {
    if (rows == table->rows) {
        return;
    }

    float* old = table->values;
    int old_rows = table->rows;
    table->values = (float*)calloc((size_t)rows * (size_t)table->cols, sizeof(float));
    if (rows > 0 && table->cols > 0 && !table->values) {
        perror("calloc");
        exit(1);
    }
    for (int r = 0; r < old_rows && r < rows; r++) {
        for (int c = 0; c < table->cols; c++) {
            table->values[r * table->cols + c] = old[r * table->cols + c];
        }
    }
    free(old);
    table->rows = rows;
}

static void table_set(Table* table, int row, int col, float value) {
    table->values[row * table->cols + col] = value;
}

static float table_get(Table* table, int row, int col) {
    return table->values[row * table->cols + col];
}

static void table_free(Table* table) {
    for (int i = 0; i < table->cols; i++) {
        free(table->labels[i]);
    }
    free(table->values);
    memset(table, 0, sizeof(*table));
}
