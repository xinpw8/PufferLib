#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PUF_DICT_MAX_ITEMS 256
#define PUF_DICT_MAX_KEY 128
#define PUF_DICT_MAX_STR 1024

typedef struct {
    char key[PUF_DICT_MAX_KEY];
    char str_buf[PUF_DICT_MAX_STR];
    char* str;
    double value;
    void* ptr;
} DictItem;

typedef struct {
    DictItem items[PUF_DICT_MAX_ITEMS];
    int size;
} Dict;

static inline double dict_get(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return dict->items[i].value;
        }
    }
    fprintf(stderr, "dict missing key: %s\n", key);
    exit(1);
}

static inline void dict_set(Dict* dict, const char* key, double value) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            dict->items[i].str = NULL;
            dict->items[i].str_buf[0] = 0;
            dict->items[i].value = value;
            return;
        }
    }

    if (dict->size >= PUF_DICT_MAX_ITEMS) {
        fprintf(stderr, "dict full while inserting key: %s\n", key);
        exit(1);
    }
    if (strlen(key) >= PUF_DICT_MAX_KEY) {
        fprintf(stderr, "dict key too long: %s\n", key);
        exit(1);
    }

    DictItem* item = &dict->items[dict->size++];
    memset(item, 0, sizeof(*item));
    snprintf(item->key, sizeof(item->key), "%s", key);
    item->value = value;
}

static inline const char* dict_get_str(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0 && dict->items[i].str) {
            return dict->items[i].str;
        }
    }
    fprintf(stderr, "dict missing string key: %s\n", key);
    exit(1);
}

static inline void dict_set_str(Dict* dict, const char* key, const char* value) {
    if (strlen(value) >= PUF_DICT_MAX_STR) {
        fprintf(stderr, "dict string too long for key: %s\n", key);
        exit(1);
    }

    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            snprintf(dict->items[i].str_buf, sizeof(dict->items[i].str_buf), "%s", value);
            dict->items[i].str = dict->items[i].str_buf;
            return;
        }
    }

    if (dict->size >= PUF_DICT_MAX_ITEMS) {
        fprintf(stderr, "dict full while inserting key: %s\n", key);
        exit(1);
    }
    if (strlen(key) >= PUF_DICT_MAX_KEY) {
        fprintf(stderr, "dict key too long: %s\n", key);
        exit(1);
    }

    DictItem* item = &dict->items[dict->size++];
    memset(item, 0, sizeof(*item));
    snprintf(item->key, sizeof(item->key), "%s", key);
    snprintf(item->str_buf, sizeof(item->str_buf), "%s", value);
    item->str = item->str_buf;
}
