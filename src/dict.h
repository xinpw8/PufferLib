#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PUF_DICT_MAX_KEY 128

typedef struct {
    char key[PUF_DICT_MAX_KEY];
    char* str;
    double value;
    double* values;
    int len;
} DictItem;

typedef struct {
    char* name;
    DictItem* items;
    int size;
    int cap;
} Dict;

static inline char* dict_strdup(const char* s) {
    size_t n = strlen(s) + 1;
    char* out = (char*)malloc(n);
    if (!out) {
        perror("malloc");
        exit(1);
    }
    memcpy(out, s, n);
    return out;
}

static inline DictItem* dict_find(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

static inline void dict_reserve(Dict* dict, int cap) {
    if (cap <= dict->cap) {
        return;
    }
    dict->items = (DictItem*)realloc(dict->items, (size_t)cap * sizeof(DictItem));
    if (!dict->items) {
        perror("realloc");
        exit(1);
    }
    memset(dict->items + dict->cap, 0, (size_t)(cap - dict->cap) * sizeof(DictItem));
    dict->cap = cap;
}

static inline DictItem* dict_insert(Dict* dict, const char* key) {
    if (strlen(key) >= PUF_DICT_MAX_KEY) {
        fprintf(stderr, "dict key too long: %s\n", key);
        exit(1);
    }
    if (dict->size == dict->cap) {
        dict_reserve(dict, dict->cap ? 2 * dict->cap : 8);
    }

    DictItem* item = &dict->items[dict->size++];
    memset(item, 0, sizeof(*item));
    snprintf(item->key, sizeof(item->key), "%s", key);
    return item;
}

static inline DictItem* dict_item(Dict* dict, const char* key) {
    DictItem* item = dict_find(dict, key);
    if (item) {
        return item;
    }
    return dict_insert(dict, key);
}

static inline void dict_item_clear(DictItem* item) {
    free(item->str);
    free(item->values);
    item->str = NULL;
    item->values = NULL;
    item->len = 0;
}

static inline double dict_get(Dict* dict, const char* key) {
    DictItem* item = dict_find(dict, key);
    if (!item) {
        fprintf(stderr, "dict missing key: %s\n", key);
        exit(1);
    }
    return item->value;
}

static inline DictItem* dict_set(Dict* dict, const char* key, double value) {
    DictItem* item = dict_item(dict, key);
    dict_item_clear(item);
    item->value = value;
    return item;
}

static inline const char* dict_get_str(Dict* dict, const char* key) {
    DictItem* item = dict_find(dict, key);
    if (!item || !item->str) {
        fprintf(stderr, "dict missing string key: %s\n", key);
        exit(1);
    }
    return item->str;
}

static inline DictItem* dict_set_str(Dict* dict, const char* key, const char* value) {
    DictItem* item = dict_item(dict, key);
    dict_item_clear(item);
    item->str = dict_strdup(value);
    return item;
}

static inline void dict_copy(Dict* dst, Dict* src) {
    memset(dst, 0, sizeof(*dst));
    if (src->name) {
        dst->name = dict_strdup(src->name);
    }
    if (src->size) {
        dst->cap = src->size;
        dst->items = (DictItem*)calloc((size_t)dst->cap, sizeof(DictItem));
        if (!dst->items) {
            perror("calloc");
            exit(1);
        }
    }
    for (int i = 0; i < src->size; i++) {
        DictItem* s = &src->items[i];
        DictItem* d = dict_insert(dst, s->key);
        d->value = s->value;
        d->len = s->len;
        if (s->str) {
            d->str = dict_strdup(s->str);
        }
        if (s->values) {
            d->values = (double*)calloc((size_t)s->len, sizeof(double));
            if (!d->values) {
                perror("calloc");
                exit(1);
            }
            memcpy(d->values, s->values, (size_t)s->len * sizeof(double));
        }
    }
}

static inline void dict_clear(Dict* dict) {
    for (int i = 0; i < dict->size; i++) {
        dict_item_clear(&dict->items[i]);
    }
    free(dict->name);
    free(dict->items);
    memset(dict, 0, sizeof(*dict));
}
