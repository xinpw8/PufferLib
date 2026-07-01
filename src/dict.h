#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* key;
    char* str;
    double value;
    void* ptr;
} DictItem;

typedef struct {
    DictItem* items;
    int size;
    int capacity;
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

static inline void dict_reserve(Dict* dict, int capacity) {
    if (capacity <= dict->capacity) {
        return;
    }

    dict->items = (DictItem*)realloc(dict->items, (size_t)capacity * sizeof(DictItem));
    if (!dict->items) {
        perror("realloc");
        exit(1);
    }
    memset(dict->items + dict->capacity, 0,
        (size_t)(capacity - dict->capacity) * sizeof(DictItem));
    dict->capacity = capacity;
}

static inline Dict* create_dict(int capacity) {
    Dict* dict = (Dict*)calloc(1, sizeof(Dict));
    if (!dict) {
        perror("calloc");
        exit(1);
    }
    dict_reserve(dict, capacity > 0 ? capacity : 1);
    return dict;
}

static inline DictItem* dict_get_unsafe(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

static inline DictItem* dict_get(Dict* dict, const char* key) {
    DictItem* item = dict_get_unsafe(dict, key);
    if (item == NULL) {
        fprintf(stderr, "dict_get failed to find key: %s\n", key);
        exit(1);
    }
    return item;
}

static inline DictItem* dict_insert(Dict* dict, const char* key) {
    DictItem* item = dict_get_unsafe(dict, key);
    if (item) {
        return item;
    }

    if (dict->size == dict->capacity) {
        dict_reserve(dict, dict->capacity ? 2 * dict->capacity : 16);
    }

    item = &dict->items[dict->size++];
    memset(item, 0, sizeof(*item));
    item->key = dict_strdup(key);
    return item;
}

static inline void dict_set(Dict* dict, const char* key, double value) {
    DictItem* item = dict_insert(dict, key);
    free(item->str);
    item->str = NULL;
    item->value = value;
}

static inline void dict_set_str(Dict* dict, const char* key, const char* value) {
    DictItem* item = dict_insert(dict, key);
    free(item->str);
    item->str = dict_strdup(value);
}

static inline double dict_get_val(Dict* dict, const char* key) {
    return dict_get(dict, key)->value;
}

static inline const char* dict_get_str(Dict* dict, const char* key) {
    DictItem* item = dict_get(dict, key);
    if (!item->str) {
        fprintf(stderr, "dict_get_str expected string for key: %s\n", key);
        exit(1);
    }
    return item->str;
}

static inline void dict_clear(Dict* dict) {
    for (int i = 0; i < dict->size; i++) {
        free(dict->items[i].key);
        free(dict->items[i].str);
    }
    free(dict->items);
    memset(dict, 0, sizeof(*dict));
}

static inline void dict_free(Dict* dict) {
    if (!dict) {
        return;
    }
    dict_clear(dict);
    free(dict);
}

static inline void dict_copy(Dict* dst, Dict* src) {
    memset(dst, 0, sizeof(*dst));
    dict_reserve(dst, src->size);
    for (int i = 0; i < src->size; i++) {
        DictItem* item = &src->items[i];
        if (item->str) {
            dict_set_str(dst, item->key, item->str);
            dict_get(dst, item->key)->value = item->value;
        } else {
            dict_set(dst, item->key, item->value);
        }
        dict_get(dst, item->key)->ptr = item->ptr;
    }
}

static inline Dict* dict_copy_prefix(Dict* src, const char* prefix) {
    size_t prefix_len = strlen(prefix);
    Dict* dst = create_dict(16);
    for (int i = 0; i < src->size; i++) {
        DictItem* item = &src->items[i];
        if (strncmp(item->key, prefix, prefix_len) != 0) {
            continue;
        }

        const char* key = item->key + prefix_len;
        if (item->str) {
            dict_set_str(dst, key, item->str);
            dict_get(dst, key)->value = item->value;
        } else {
            dict_set(dst, key, item->value);
        }
        dict_get(dst, key)->ptr = item->ptr;
    }
    return dst;
}
