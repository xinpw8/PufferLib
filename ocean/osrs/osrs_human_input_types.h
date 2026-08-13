#ifndef OSRS_HUMAN_INPUT_TYPES_H
#define OSRS_HUMAN_INPUT_TYPES_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osrs_ui_intent.h"

typedef enum {
    CURSOR_NORMAL = 0,
    CURSOR_ITEM_TARGET,
    CURSOR_SPELL_TARGET,
} CursorMode;

typedef enum {
    HUMAN_COMMAND_NONE = 0,
    HUMAN_COMMAND_WALK,
    HUMAN_COMMAND_ATTACK_NPC,
    HUMAN_COMMAND_OVERHEAD_PRAYER,
    HUMAN_COMMAND_OFFENSIVE_PRAYER,
    HUMAN_COMMAND_EAT,
    HUMAN_COMMAND_DRINK,
    HUMAN_COMMAND_SPELL_TARGET,
    HUMAN_COMMAND_SPEC_TOGGLE,
    HUMAN_COMMAND_EQUIP_INVENTORY_ITEM,
    HUMAN_COMMAND_FIGHT_STYLE,
    HUMAN_COMMAND_SET_AUTOCAST,
    HUMAN_COMMAND_ITEM_ON_ITEM,
    HUMAN_COMMAND_ITEM_ON_WIDGET,
    HUMAN_COMMAND_SPELL_ON_WIDGET,
    HUMAN_COMMAND_INVENTORY_PRIMARY_CLICK,
} HumanCommandKind;

typedef struct {
    HumanCommandKind kind;
    int world_x, world_y;
    int npc_slot;
    int overhead_prayer;
    int offensive_prayer;
    int food;
    int potion;
    int spell;
    int spell_gui_idx;
    int inventory_slot;
    int target_inventory_slot;
    int item_db_idx;
    int item_osrs_id;
    int gear_slot;
    int fight_style;
    int autocast_spell;
    int autocast_defensive;
    uint32_t widget_component_id;
} HumanCommand;

typedef struct {
    HumanCommand* items;
    int count;
    int capacity;
} HumanCommandQueue;

typedef struct HumanInput {
    int enabled;

    HumanCommandQueue commands;

    int pending_move_x, pending_move_y;
    int pending_attack;
    int pending_prayer;
    int pending_offensive_prayer;
    int pending_food;
    int pending_karambwan;
    int pending_potion;
    int pending_veng;
    int pending_spec;
    int pending_spell;
    int pending_target_idx;
    int pending_gear;
    int pending_modifier_select;
    int pending_grapple_slot;

    CursorMode cursor_mode;
    int selected_item_inventory_slot;
    int selected_item_db_idx;
    int selected_item_osrs_id;
    int selected_spell;
    int selected_spell_gui_idx;

    int click_screen_x, click_screen_y;
    int click_cross_timer;
    int click_cross_active;
    int click_is_attack;
} HumanInput;

static inline void human_command_queue_reserve(HumanCommandQueue* q, int min_capacity) {
    if (q->capacity >= min_capacity) return;
    int new_capacity = q->capacity > 0 ? q->capacity : 8;
    while (new_capacity < min_capacity)
        new_capacity *= 2;
    HumanCommand* next = (HumanCommand*)realloc(q->items, (size_t)new_capacity * sizeof(HumanCommand));
    if (!next) {
        fprintf(stderr, "human command queue: out of memory\n");
        abort();
    }
    q->items = next;
    q->capacity = new_capacity;
}

static inline void human_input_queue_command(HumanInput* hi, HumanCommand cmd) {
    human_command_queue_reserve(&hi->commands, hi->commands.count + 1);
    hi->commands.items[hi->commands.count++] = cmd;
}

static inline void human_input_clear_commands(HumanInput* hi) {
    hi->commands.count = 0;
}

static inline void human_input_destroy(HumanInput* hi) {
    free(hi->commands.items);
    hi->commands.items = NULL;
    hi->commands.count = 0;
    hi->commands.capacity = 0;
}

static inline void human_input_init(HumanInput* hi) {
    memset(hi, 0, sizeof(*hi));
    hi->pending_move_x = -1;
    hi->pending_move_y = -1;
    hi->pending_prayer = -1;
    hi->pending_offensive_prayer = -1;
    hi->pending_target_idx = -1;
    hi->selected_item_inventory_slot = -1;
    hi->selected_item_db_idx = -1;
    hi->selected_spell_gui_idx = -1;
    hi->click_cross_active = 0;
    human_command_queue_reserve(&hi->commands, 8);
}

static inline void human_input_clear_pending(HumanInput* hi) {
    hi->pending_attack = 0;
    hi->pending_prayer = -1;
    hi->pending_offensive_prayer = -1;
    hi->pending_food = 0;
    hi->pending_karambwan = 0;
    hi->pending_potion = 0;
    hi->pending_veng = 0;
    hi->pending_spec = 0;
    hi->pending_spell = 0;
    hi->pending_target_idx = -1;
    hi->pending_gear = 0;
    hi->pending_modifier_select = 0;
    hi->pending_grapple_slot = 0;
    human_input_clear_commands(hi);
}

static inline void human_input_clear_move(HumanInput* hi) {
    hi->pending_move_x = -1;
    hi->pending_move_y = -1;
}

static inline void human_input_clear_selected_ui_target(HumanInput* hi) {
    hi->cursor_mode = CURSOR_NORMAL;
    hi->selected_item_inventory_slot = -1;
    hi->selected_item_db_idx = -1;
    hi->selected_item_osrs_id = 0;
    hi->selected_spell = 0;
    hi->selected_spell_gui_idx = -1;
}

static inline void human_input_queue_walk(HumanInput* hi, int world_x, int world_y) {
    hi->pending_move_x = world_x;
    hi->pending_move_y = world_y;
    hi->pending_attack = 0;
    hi->pending_target_idx = -1;
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_WALK,
        .world_x = world_x,
        .world_y = world_y,
    });
}

static inline void human_input_queue_attack_npc(HumanInput* hi, int npc_slot) {
    human_input_clear_move(hi);
    hi->pending_attack = 1;
    hi->pending_target_idx = npc_slot;
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_ATTACK_NPC,
        .npc_slot = npc_slot,
    });
}

static inline void human_input_queue_overhead_prayer(HumanInput* hi, int overhead_prayer) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_OVERHEAD_PRAYER,
        .overhead_prayer = overhead_prayer,
    });
}

static inline void human_input_queue_offensive_prayer(HumanInput* hi, int offensive_prayer) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_OFFENSIVE_PRAYER,
        .offensive_prayer = offensive_prayer,
    });
}

static inline void human_input_queue_eat(HumanInput* hi, int food, int inventory_slot) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_EAT,
        .food = food,
        .inventory_slot = inventory_slot,
    });
}

static inline void human_input_queue_drink(HumanInput* hi, int potion, int inventory_slot) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_DRINK,
        .potion = potion,
        .inventory_slot = inventory_slot,
    });
}

static inline void human_input_queue_spell_target(HumanInput* hi, int spell, int npc_slot) {
    human_input_clear_move(hi);
    hi->pending_attack = 1;
    hi->pending_spell = spell;
    hi->pending_target_idx = npc_slot;
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_SPELL_TARGET,
        .npc_slot = npc_slot,
        .spell = spell,
    });
}

static inline void human_input_queue_item_on_item(
    HumanInput* hi,
    int source_inventory_slot,
    int target_inventory_slot,
    int item_db_idx,
    int item_osrs_id
) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_ITEM_ON_ITEM,
        .inventory_slot = source_inventory_slot,
        .target_inventory_slot = target_inventory_slot,
        .item_db_idx = item_db_idx,
        .item_osrs_id = item_osrs_id,
    });
}

static inline void human_input_queue_item_on_widget(
    HumanInput* hi,
    int source_inventory_slot,
    int item_db_idx,
    int item_osrs_id,
    uint32_t widget_component_id
) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_ITEM_ON_WIDGET,
        .inventory_slot = source_inventory_slot,
        .item_db_idx = item_db_idx,
        .item_osrs_id = item_osrs_id,
        .widget_component_id = widget_component_id,
    });
}

static inline void human_input_queue_spell_on_widget(
    HumanInput* hi,
    int spell,
    int spell_gui_idx,
    uint32_t widget_component_id
) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_SPELL_ON_WIDGET,
        .spell = spell,
        .spell_gui_idx = spell_gui_idx,
        .widget_component_id = widget_component_id,
    });
}

static inline void human_input_queue_spec_toggle(HumanInput* hi) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_SPEC_TOGGLE,
    });
}

static inline void human_input_queue_equip_inventory_item(
    HumanInput* hi, int inventory_slot, int item_db_idx, int gear_slot
) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_EQUIP_INVENTORY_ITEM,
        .inventory_slot = inventory_slot,
        .item_db_idx = item_db_idx,
        .gear_slot = gear_slot,
    });
}

static inline void human_input_queue_inventory_primary_click(
    HumanInput* hi,
    int inventory_slot
) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_INVENTORY_PRIMARY_CLICK,
        .inventory_slot = inventory_slot,
    });
}

static inline void human_input_queue_fight_style(HumanInput* hi, int fight_style) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_FIGHT_STYLE,
        .fight_style = fight_style,
    });
}

static inline void human_input_queue_set_autocast(
    HumanInput* hi,
    int autocast_spell,
    int autocast_defensive
) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_SET_AUTOCAST,
        .autocast_spell = autocast_spell,
        .autocast_defensive = autocast_defensive,
    });
}

static inline void human_input_apply_ui_intent(HumanInput* hi, OsrsUiIntent intent) {
    switch (intent.kind) {
        case OSRS_UI_INTENT_SELECT_ITEM:
            human_input_clear_selected_ui_target(hi);
            hi->cursor_mode = CURSOR_ITEM_TARGET;
            hi->selected_item_inventory_slot = intent.source_inventory_slot;
            hi->selected_item_db_idx = intent.item_db_idx;
            hi->selected_item_osrs_id = intent.item_osrs_id;
            break;
        case OSRS_UI_INTENT_SELECT_SPELL:
            human_input_clear_selected_ui_target(hi);
            hi->cursor_mode = CURSOR_SPELL_TARGET;
            hi->selected_spell = intent.spell;
            hi->selected_spell_gui_idx = intent.spell_gui_idx;
            break;
        case OSRS_UI_INTENT_ITEM_ON_ITEM:
            human_input_queue_item_on_item(
                hi,
                intent.source_inventory_slot,
                intent.target_inventory_slot,
                intent.item_db_idx,
                intent.item_osrs_id);
            human_input_clear_selected_ui_target(hi);
            break;
        case OSRS_UI_INTENT_ITEM_ON_WIDGET:
            human_input_queue_item_on_widget(
                hi,
                intent.source_inventory_slot,
                intent.item_db_idx,
                intent.item_osrs_id,
                intent.widget_component_id);
            human_input_clear_selected_ui_target(hi);
            break;
        case OSRS_UI_INTENT_SPELL_ON_TARGET:
            hi->pending_attack = 1;
            hi->pending_target_idx = intent.npc_slot;
            hi->pending_spell = intent.spell;
            hi->pending_move_x = -1;
            hi->pending_move_y = -1;
            human_input_queue_command(hi, (HumanCommand){
                .kind = HUMAN_COMMAND_SPELL_TARGET,
                .npc_slot = intent.npc_slot,
                .spell = intent.spell,
                .spell_gui_idx = intent.spell_gui_idx,
            });
            human_input_clear_selected_ui_target(hi);
            break;
        case OSRS_UI_INTENT_SPELL_ON_WIDGET:
            human_input_queue_spell_on_widget(
                hi,
                intent.spell,
                intent.spell_gui_idx,
                intent.widget_component_id);
            human_input_clear_selected_ui_target(hi);
            break;
        case OSRS_UI_INTENT_NONE:
            break;
        default:
            fprintf(stderr, "human_input_apply_ui_intent: bad intent kind: %d\n",
                (int)intent.kind);
            abort();
    }
}

#endif
