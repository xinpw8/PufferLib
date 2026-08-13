#ifndef OSRS_UI_INTENT_H
#define OSRS_UI_INTENT_H

#include <stdint.h>

typedef enum {
    OSRS_UI_INTENT_NONE = 0,
    OSRS_UI_INTENT_SELECT_ITEM,
    OSRS_UI_INTENT_SELECT_SPELL,
    OSRS_UI_INTENT_ITEM_ON_ITEM,
    OSRS_UI_INTENT_ITEM_ON_WIDGET,
    OSRS_UI_INTENT_SPELL_ON_TARGET,
    OSRS_UI_INTENT_SPELL_ON_WIDGET,
} OsrsUiIntentKind;

typedef struct {
    OsrsUiIntentKind kind;
    int source_inventory_slot;
    int target_inventory_slot;
    int item_db_idx;
    int item_osrs_id;
    int spell;
    int spell_gui_idx;
    int npc_slot;
    uint32_t widget_component_id;
} OsrsUiIntent;

static inline uint32_t osrs_ui_intent_widget_component_id(int group_id, int child_id) {
    return ((uint32_t)group_id << 16) | ((uint32_t)child_id & 0xffffu);
}

static inline int osrs_ui_intent_widget_group_id(uint32_t component_id) {
    return (int)(component_id >> 16);
}

static inline int osrs_ui_intent_widget_child_id(uint32_t component_id) {
    return (int)(component_id & 0xffffu);
}

static inline OsrsUiIntent osrs_ui_intent_select_item(
    int inventory_slot,
    int item_db_idx,
    int item_osrs_id
) {
    return (OsrsUiIntent){
        .kind = OSRS_UI_INTENT_SELECT_ITEM,
        .source_inventory_slot = inventory_slot,
        .item_db_idx = item_db_idx,
        .item_osrs_id = item_osrs_id,
    };
}

static inline OsrsUiIntent osrs_ui_intent_select_spell(int spell, int spell_gui_idx) {
    return (OsrsUiIntent){
        .kind = OSRS_UI_INTENT_SELECT_SPELL,
        .spell = spell,
        .spell_gui_idx = spell_gui_idx,
    };
}

static inline OsrsUiIntent osrs_ui_intent_item_on_item(
    int source_inventory_slot,
    int target_inventory_slot,
    int item_db_idx,
    int item_osrs_id
) {
    return (OsrsUiIntent){
        .kind = OSRS_UI_INTENT_ITEM_ON_ITEM,
        .source_inventory_slot = source_inventory_slot,
        .target_inventory_slot = target_inventory_slot,
        .item_db_idx = item_db_idx,
        .item_osrs_id = item_osrs_id,
    };
}

static inline OsrsUiIntent osrs_ui_intent_item_on_widget(
    int source_inventory_slot,
    int item_db_idx,
    int item_osrs_id,
    uint32_t widget_component_id
) {
    return (OsrsUiIntent){
        .kind = OSRS_UI_INTENT_ITEM_ON_WIDGET,
        .source_inventory_slot = source_inventory_slot,
        .item_db_idx = item_db_idx,
        .item_osrs_id = item_osrs_id,
        .widget_component_id = widget_component_id,
    };
}

static inline OsrsUiIntent osrs_ui_intent_spell_on_target(
    int spell,
    int spell_gui_idx,
    int npc_slot
) {
    return (OsrsUiIntent){
        .kind = OSRS_UI_INTENT_SPELL_ON_TARGET,
        .spell = spell,
        .spell_gui_idx = spell_gui_idx,
        .npc_slot = npc_slot,
    };
}

static inline OsrsUiIntent osrs_ui_intent_spell_on_widget(
    int spell,
    int spell_gui_idx,
    uint32_t widget_component_id
) {
    return (OsrsUiIntent){
        .kind = OSRS_UI_INTENT_SPELL_ON_WIDGET,
        .spell = spell,
        .spell_gui_idx = spell_gui_idx,
        .widget_component_id = widget_component_id,
    };
}

#endif
