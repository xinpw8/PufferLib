#ifndef OSRS_UI_INTERFACES_H
#define OSRS_UI_INTERFACES_H

#if __has_include("raylib.h")
#include "raylib.h"
#elif __has_include("raylib-5.5_macos/include/raylib.h")
#include "raylib-5.5_macos/include/raylib.h"
#else
#error "raylib.h not found"
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osrs_assets.h"

#define OSRS_UI_BIN_MAGIC "RCUIBIN2"
#define OSRS_UI_BIN_VERSION_MIN 1u
#define OSRS_UI_BIN_VERSION_MAX 2u
#define OSRS_UI_INTERFACE_MAX_ACTIONS 10

#define OSRS_UI_COMPONENT_ID(group, file) (((uint32_t)(group) << 16) | (uint32_t)(file))
#define OSRS_UI_GROUP_INVENTORY 149u
#define OSRS_UI_GROUP_WORNITEMS 387u
#define OSRS_UI_GROUP_PRAYERBOOK 541u
#define OSRS_UI_GROUP_MAGIC_SPELLBOOK 218u
#define OSRS_UI_GROUP_COMBAT_INTERFACE 593u

typedef enum {
    OSRS_UI_LISTENER_VALUE_INT = 0,
    OSRS_UI_LISTENER_VALUE_STRING
} OsrsUiListenerValueKind;

typedef struct {
    uint32_t id;
    int32_t parent_id;
    uint32_t group_id;
    uint32_t file_id;
    unsigned char is_if3;
    unsigned char type;
    unsigned char hidden;
    unsigned char sprite_tiling;
    unsigned char filled;
    unsigned char line_direction;
    unsigned char text_shadowed;
    unsigned char flipped_vertically;
    unsigned char flipped_horizontally;
    unsigned char no_click_through;
    unsigned char opacity;
    unsigned char border_type;
    unsigned char line_width;
    unsigned char line_height;
    int32_t content_type;
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
    int32_t width_mode;
    int32_t height_mode;
    int32_t x_position_mode;
    int32_t y_position_mode;
    int32_t scroll_width;
    int32_t scroll_height;
    int32_t sprite_id;
    int32_t texture_id;
    int32_t shadow_color;
    int32_t model_id;
    int32_t model_type;
    int32_t font_id;
    int32_t text_color;
    uint32_t click_mask;
    int32_t x_text_alignment;
    int32_t y_text_alignment;
    char* name;
    char* text;
    char* target_verb;
    char* actions[OSRS_UI_INTERFACE_MAX_ACTIONS];
    int action_count;
} OsrsUiComponent;

typedef struct {
    uint32_t id;
    char* name;
    OsrsUiComponent* components;
    int component_count;
} OsrsUiInterfaceGroup;

typedef struct {
    OsrsUiInterfaceGroup* groups;
    int group_count;
    int loaded;
} OsrsUiInterfaceStore;

typedef struct {
    const unsigned char* data;
    size_t size;
    size_t pos;
    int failed;
} OsrsUiReader;

static uint8_t osrs_ui_read_u8(OsrsUiReader* r) {
    if (r->pos + 1 > r->size) {
        r->failed = 1;
        return 0;
    }
    return r->data[r->pos++];
}

static uint16_t osrs_ui_read_u16(OsrsUiReader* r) {
    if (r->pos + 2 > r->size) {
        r->failed = 1;
        return 0;
    }
    uint16_t value = (uint16_t)r->data[r->pos]
        | ((uint16_t)r->data[r->pos + 1] << 8);
    r->pos += 2;
    return value;
}

static uint32_t osrs_ui_read_u32(OsrsUiReader* r) {
    if (r->pos + 4 > r->size) {
        r->failed = 1;
        return 0;
    }
    uint32_t value = (uint32_t)r->data[r->pos]
        | ((uint32_t)r->data[r->pos + 1] << 8)
        | ((uint32_t)r->data[r->pos + 2] << 16)
        | ((uint32_t)r->data[r->pos + 3] << 24);
    r->pos += 4;
    return value;
}

static int32_t osrs_ui_read_i32(OsrsUiReader* r) {
    return (int32_t)osrs_ui_read_u32(r);
}

static char* osrs_ui_read_string(OsrsUiReader* r) {
    uint16_t len = osrs_ui_read_u16(r);
    if (r->failed || r->pos + len > r->size) {
        r->failed = 1;
        return NULL;
    }
    char* out = (char*)calloc((size_t)len + 1, 1);
    if (!out) {
        fprintf(stderr, "osrs ui string allocation failed\n");
        abort();
    }
    memcpy(out, r->data + r->pos, len);
    r->pos += len;
    return out;
}

static unsigned char* osrs_ui_read_file_bytes(const char* path, size_t* out_size) {
    FILE* file = osrs_asset_fopen(path, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (size <= 0) {
        fclose(file);
        return NULL;
    }
    unsigned char* data = (unsigned char*)malloc((size_t)size);
    if (!data) {
        fclose(file);
        fprintf(stderr, "osrs ui file allocation failed\n");
        abort();
    }
    if (fread(data, 1, (size_t)size, file) != (size_t)size) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *out_size = (size_t)size;
    return data;
}

static void osrs_ui_free_component(OsrsUiComponent* component) {
    free(component->name);
    free(component->text);
    free(component->target_verb);
    for (int i = 0; i < component->action_count; i++) {
        free(component->actions[i]);
    }
}

static void osrs_ui_interfaces_unload(OsrsUiInterfaceStore* store) {
    if (!store) return;
    for (int g = 0; g < store->group_count; g++) {
        OsrsUiInterfaceGroup* group = &store->groups[g];
        free(group->name);
        for (int c = 0; c < group->component_count; c++) {
            osrs_ui_free_component(&group->components[c]);
        }
        free(group->components);
    }
    free(store->groups);
    memset(store, 0, sizeof(*store));
}

static int osrs_ui_interfaces_load(OsrsUiInterfaceStore* store, const char* path) {
    if (!store || !path) return 0;
    osrs_ui_interfaces_unload(store);

    size_t size = 0;
    unsigned char* bytes = osrs_ui_read_file_bytes(path, &size);
    if (!bytes) return 0;
    if (size < 16 || memcmp(bytes, OSRS_UI_BIN_MAGIC, 8) != 0) {
        free(bytes);
        return 0;
    }

    OsrsUiReader r = {.data = bytes, .size = size, .pos = 8};
    uint32_t version = osrs_ui_read_u32(&r);
    uint32_t group_count = osrs_ui_read_u32(&r);
    if (version < OSRS_UI_BIN_VERSION_MIN || version > OSRS_UI_BIN_VERSION_MAX
            || group_count > 4096) {
        free(bytes);
        return 0;
    }

    store->groups = (OsrsUiInterfaceGroup*)calloc(group_count, sizeof(*store->groups));
    if (!store->groups) {
        free(bytes);
        fprintf(stderr, "osrs ui group allocation failed\n");
        abort();
    }
    store->group_count = (int)group_count;

    for (uint32_t g = 0; g < group_count && !r.failed; g++) {
        OsrsUiInterfaceGroup* group = &store->groups[g];
        group->id = osrs_ui_read_u32(&r);
        group->name = osrs_ui_read_string(&r);
        uint32_t component_count = osrs_ui_read_u32(&r);
        if (component_count > 65535) {
            r.failed = 1;
            break;
        }
        group->component_count = (int)component_count;
        group->components = (OsrsUiComponent*)calloc(component_count, sizeof(*group->components));
        if (!group->components) {
            free(bytes);
            fprintf(stderr, "osrs ui component allocation failed\n");
            abort();
        }
        for (uint32_t c = 0; c < component_count && !r.failed; c++) {
            OsrsUiComponent* component = &group->components[c];
            component->id = osrs_ui_read_u32(&r);
            component->parent_id = osrs_ui_read_i32(&r);
            component->group_id = osrs_ui_read_u32(&r);
            component->file_id = osrs_ui_read_u32(&r);
            component->is_if3 = osrs_ui_read_u8(&r);
            component->type = osrs_ui_read_u8(&r);
            component->hidden = osrs_ui_read_u8(&r);
            component->sprite_tiling = osrs_ui_read_u8(&r);
            component->filled = osrs_ui_read_u8(&r);
            component->line_direction = osrs_ui_read_u8(&r);
            component->text_shadowed = osrs_ui_read_u8(&r);
            component->flipped_vertically = osrs_ui_read_u8(&r);
            component->flipped_horizontally = osrs_ui_read_u8(&r);
            component->no_click_through = osrs_ui_read_u8(&r);
            component->opacity = osrs_ui_read_u8(&r);
            component->border_type = osrs_ui_read_u8(&r);
            component->line_width = osrs_ui_read_u8(&r);
            component->line_height = osrs_ui_read_u8(&r);
            component->content_type = osrs_ui_read_i32(&r);
            component->x = osrs_ui_read_i32(&r);
            component->y = osrs_ui_read_i32(&r);
            component->width = osrs_ui_read_i32(&r);
            component->height = osrs_ui_read_i32(&r);
            component->width_mode = osrs_ui_read_i32(&r);
            component->height_mode = osrs_ui_read_i32(&r);
            component->x_position_mode = osrs_ui_read_i32(&r);
            component->y_position_mode = osrs_ui_read_i32(&r);
            component->scroll_width = osrs_ui_read_i32(&r);
            component->scroll_height = osrs_ui_read_i32(&r);
            component->sprite_id = osrs_ui_read_i32(&r);
            component->texture_id = osrs_ui_read_i32(&r);
            component->shadow_color = osrs_ui_read_i32(&r);
            component->model_id = osrs_ui_read_i32(&r);
            component->model_type = osrs_ui_read_i32(&r);
            component->font_id = osrs_ui_read_i32(&r);
            component->text_color = osrs_ui_read_i32(&r);
            component->click_mask = osrs_ui_read_u32(&r);
            component->x_text_alignment = osrs_ui_read_i32(&r);
            component->y_text_alignment = osrs_ui_read_i32(&r);
            component->name = osrs_ui_read_string(&r);
            component->text = osrs_ui_read_string(&r);
            component->target_verb = osrs_ui_read_string(&r);
            int encoded_action_count = (int)osrs_ui_read_u8(&r);
            component->action_count = encoded_action_count;
            if (component->action_count > OSRS_UI_INTERFACE_MAX_ACTIONS) {
                component->action_count = OSRS_UI_INTERFACE_MAX_ACTIONS;
            }
            for (int a = 0; a < encoded_action_count; a++) {
                char* action = osrs_ui_read_string(&r);
                if (a < OSRS_UI_INTERFACE_MAX_ACTIONS) {
                    component->actions[a] = action;
                } else {
                    free(action);
                }
            }
            if (version >= 2) {
                int encoded_listener_count = (int)osrs_ui_read_u8(&r);
                for (int l = 0; l < encoded_listener_count; l++) {
                    osrs_ui_read_u8(&r);
                    int encoded_value_count = (int)osrs_ui_read_u8(&r);
                    for (int v = 0; v < encoded_value_count; v++) {
                        int value_type = (int)osrs_ui_read_u8(&r);
                        if (value_type == OSRS_UI_LISTENER_VALUE_STRING) {
                            free(osrs_ui_read_string(&r));
                        } else {
                            osrs_ui_read_i32(&r);
                        }
                    }
                }

                int encoded_trigger_count = (int)osrs_ui_read_u8(&r);
                for (int t = 0; t < encoded_trigger_count; t++) {
                    osrs_ui_read_u8(&r);
                    int encoded_value_count = (int)osrs_ui_read_u8(&r);
                    for (int v = 0; v < encoded_value_count; v++) {
                        osrs_ui_read_i32(&r);
                    }
                }
            }
        }
    }

    free(bytes);
    if (r.failed) {
        osrs_ui_interfaces_unload(store);
        return 0;
    }
    store->loaded = 1;
    return 1;
}

static const OsrsUiInterfaceGroup* osrs_ui_interface_group(
    const OsrsUiInterfaceStore* store,
    const char* name
) {
    if (!store || !store->loaded || !name) return NULL;
    for (int i = 0; i < store->group_count; i++) {
        if (store->groups[i].name && strcmp(store->groups[i].name, name) == 0) {
            return &store->groups[i];
        }
    }
    return NULL;
}

static Rectangle osrs_ui_rect_expand_to_scroll(Rectangle rect, const OsrsUiComponent* component) {
    Rectangle out = rect;
    if (component->scroll_width > 0 && (float)component->scroll_width > out.width) {
        out.width = (float)component->scroll_width;
    }
    if (component->scroll_height > 0 && (float)component->scroll_height > out.height) {
        out.height = (float)component->scroll_height;
    }
    return out;
}

static Rectangle osrs_ui_layout_component(
    const OsrsUiComponent* component,
    Rectangle parent,
    int root
) {
    if (root) return parent;

    float w = (float)component->width;
    float h = (float)component->height;
    if (component->width_mode == 1) {
        w = parent.width - (float)component->width;
    } else if (component->width_mode == 2) {
        w = parent.width * (float)component->width / 16384.0f;
    }
    if (component->height_mode == 1) {
        h = parent.height - (float)component->height;
    } else if (component->height_mode == 2) {
        h = parent.height * (float)component->height / 16384.0f;
    }
    if (w < 0) w = 0;
    if (h < 0) h = 0;

    float x = parent.x + (float)component->x;
    float y = parent.y + (float)component->y;
    if (component->x_position_mode == 1) {
        x = parent.x + (parent.width - w) * 0.5f + (float)component->x;
    } else if (component->x_position_mode == 2) {
        x = parent.x + parent.width - w - (float)component->x;
    } else if (component->x_position_mode == 3) {
        x = parent.x + parent.width * (float)component->x / 16384.0f;
    } else if (component->x_position_mode == 4) {
        x = parent.x + (parent.width - w) * 0.5f
            + parent.width * (float)component->x / 16384.0f;
    } else if (component->x_position_mode == 5) {
        x = parent.x + parent.width - w
            - parent.width * (float)component->x / 16384.0f;
    }

    if (component->y_position_mode == 1) {
        y = parent.y + (parent.height - h) * 0.5f + (float)component->y;
    } else if (component->y_position_mode == 2) {
        y = parent.y + parent.height - h - (float)component->y;
    } else if (component->y_position_mode == 3) {
        y = parent.y + parent.height * (float)component->y / 16384.0f;
    } else if (component->y_position_mode == 4) {
        y = parent.y + (parent.height - h) * 0.5f
            + parent.height * (float)component->y / 16384.0f;
    } else if (component->y_position_mode == 5) {
        y = parent.y + parent.height - h
            - parent.height * (float)component->y / 16384.0f;
    }

    return (Rectangle){x, y, w, h};
}

static int osrs_ui_component_uses_mount_rect(const OsrsUiComponent* component) {
    return component->file_id == 0 && component->parent_id == -1;
}

static int osrs_ui_find_component_rect_recursive(
    const OsrsUiInterfaceGroup* group,
    const OsrsUiComponent* component,
    const char* component_name,
    Rectangle rect,
    Rectangle* out_rect
) {
    if (component->name && strcmp(component->name, component_name) == 0) {
        *out_rect = rect;
        return 1;
    }

    Rectangle child_parent = component->type == 0
        ? osrs_ui_rect_expand_to_scroll(rect, component)
        : rect;
    for (int i = 0; i < group->component_count; i++) {
        const OsrsUiComponent* child = &group->components[i];
        if (child->parent_id != (int32_t)component->id) continue;
        Rectangle child_rect = osrs_ui_layout_component(child, child_parent, 0);
        if (osrs_ui_find_component_rect_recursive(
                group, child, component_name, child_rect, out_rect)) {
            return 1;
        }
    }
    return 0;
}

static int osrs_ui_interfaces_component_rect(
    const OsrsUiInterfaceStore* store,
    const char* group_name,
    const char* component_name,
    Rectangle mount,
    Rectangle* out_rect
) {
    const OsrsUiInterfaceGroup* group = osrs_ui_interface_group(store, group_name);
    if (!group || !component_name || !out_rect) return 0;
    for (int i = 0; i < group->component_count; i++) {
        const OsrsUiComponent* component = &group->components[i];
        if (component->parent_id != -1) continue;
        Rectangle rect = osrs_ui_layout_component(
            component, mount, osrs_ui_component_uses_mount_rect(component));
        if (osrs_ui_find_component_rect_recursive(
                group, component, component_name, rect, out_rect)) {
            return 1;
        }
    }
    return 0;
}

#endif
