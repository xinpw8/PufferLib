#include "puffer_action_adapter.h"

#include <math.h>
#include <string.h>

static int commands_equal(
        RekG1SemanticCommand left,
        RekG1SemanticCommand right) {
    return left.kind == right.kind &&
        left.held_code == right.held_code &&
        left.duration_ticks == right.duration_ticks &&
        left.kick_registry_index == right.kick_registry_index;
}

static const uint8_t REQUIRED_HELD_MASKS[] = {
    0,
    REK_G1_HELD_FORWARD,
    REK_G1_HELD_BACKWARD,
    REK_G1_HELD_STRAFE_LEFT,
    REK_G1_HELD_STRAFE_RIGHT,
    REK_G1_HELD_YAW_LEFT,
    REK_G1_HELD_YAW_RIGHT,
    REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT,
    REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_RIGHT,
    REK_G1_HELD_BACKWARD | REK_G1_HELD_YAW_LEFT,
    REK_G1_HELD_BACKWARD | REK_G1_HELD_YAW_RIGHT,
    REK_G1_HELD_STRAFE_LEFT | REK_G1_HELD_YAW_LEFT,
    REK_G1_HELD_STRAFE_LEFT | REK_G1_HELD_YAW_RIGHT,
    REK_G1_HELD_STRAFE_RIGHT | REK_G1_HELD_YAW_LEFT,
    REK_G1_HELD_STRAFE_RIGHT | REK_G1_HELD_YAW_RIGHT,
};

static int has_locomotion_held_code(
        const RekG1PufferActionTable* table,
        uint8_t held_code) {
    for (uint32_t index = 1; index < table->count; index++) {
        const RekG1SemanticCommand* command = &table->categories[index].command;
        if (command->kind == REK_G1_SEMANTIC_LOCOMOTION &&
                command->held_code == held_code) {
            return 1;
        }
    }
    return 0;
}

static int yaw_only_locomotion_held_code(
        const RekG1PufferActionTable* table,
        uint32_t category,
        uint8_t* held_code_out) {
    if (table == 0 || table->categories == 0 || category == 0 ||
            category >= table->count || held_code_out == 0) {
        return 0;
    }
    const RekG1PufferCategory* selected = &table->categories[category];
    if (selected->kind != REK_G1_PUFFER_START ||
            selected->command.kind != REK_G1_SEMANTIC_LOCOMOTION) {
        return 0;
    }
    uint8_t held = 0;
    if (rek_g1_semantic_decode_held(selected->command.held_code, &held) !=
            REK_G1_SEMANTIC_OK ||
            (held & (uint8_t)~REK_G1_HELD_YAW_MASK) != 0) {
        return 0;
    }
    *held_code_out = selected->command.held_code;
    return 1;
}

static int has_kick_registry_index(
        const RekG1PufferActionTable* table,
        uint16_t kick_registry_index) {
    for (uint32_t index = 1; index < table->count; index++) {
        const RekG1SemanticCommand* command = &table->categories[index].command;
        if (command->kind == REK_G1_SEMANTIC_KICK &&
                command->kick_registry_index == kick_registry_index) {
            return 1;
        }
    }
    return 0;
}

void rek_g1_puffer_reset(RekG1PufferAdapter* adapter) {
    rek_g1_semantic_reset(&adapter->scheduler);
}

RekG1PufferStatus rek_g1_puffer_validate_table(
        const RekG1PufferActionTable* table) {
    if (table == 0 || table->categories == 0) {
        return REK_G1_PUFFER_TABLE_NULL;
    }
    if (table->count < 2 ||
            table->count > REK_G1_PUFFER_EXACT_CATEGORY_LIMIT) {
        return REK_G1_PUFFER_TABLE_COUNT_INVALID;
    }
    if (table->categories[0].kind != REK_G1_PUFFER_CONTINUE) {
        return REK_G1_PUFFER_TABLE_CONTINUE_INVALID;
    }
    if (table->kick_registry_count != REK_G1_REQUIRED_KICK_COUNT ||
            table->kick_move_indices == 0 ||
            table->kick_duration_ticks == 0) {
        return REK_G1_PUFFER_TABLE_KICK_REGISTRY_INVALID;
    }
    static const uint16_t required_move_indices[REK_G1_REQUIRED_KICK_COUNT] = {
        6, 7, 8, 9,
    };
    for (uint16_t kick = 0; kick < table->kick_registry_count; kick++) {
        if (table->kick_move_indices[kick] != required_move_indices[kick]) {
            return REK_G1_PUFFER_TABLE_KICK_REGISTRY_INVALID;
        }
        if (table->kick_duration_ticks[kick] == 0) {
            return REK_G1_PUFFER_TABLE_KICK_DURATION_INVALID;
        }
    }

    for (uint32_t index = 1; index < table->count; index++) {
        const RekG1PufferCategory* category = &table->categories[index];
        if (category->kind != REK_G1_PUFFER_START) {
            return REK_G1_PUFFER_TABLE_START_INVALID;
        }
        RekG1SemanticScheduler scratch;
        rek_g1_semantic_reset(&scratch);
        if (rek_g1_semantic_start(
                &scratch,
                category->command,
                table->kick_registry_count) != REK_G1_SEMANTIC_OK) {
            return REK_G1_PUFFER_TABLE_START_INVALID;
        }
        if (category->command.kind == REK_G1_SEMANTIC_KICK &&
                category->command.duration_ticks !=
                table->kick_duration_ticks[
                    category->command.kick_registry_index]) {
            return REK_G1_PUFFER_TABLE_KICK_DURATION_INVALID;
        }
        if (category->command.kind == REK_G1_SEMANTIC_KICK) {
            uint8_t template_held = 0;
            if (rek_g1_semantic_decode_held(
                    category->command.held_code, &template_held) !=
                    REK_G1_SEMANTIC_OK || template_held != 0) {
                return REK_G1_PUFFER_TABLE_START_INVALID;
            }
        }
        for (uint32_t prior = 1; prior < index; prior++) {
            if (commands_equal(
                    category->command,
                    table->categories[prior].command)) {
                return REK_G1_PUFFER_TABLE_DUPLICATE;
            }
        }
    }
    for (size_t index = 0;
            index < sizeof(REQUIRED_HELD_MASKS) / sizeof(REQUIRED_HELD_MASKS[0]);
            index++) {
        uint8_t held_code = 0;
        if (rek_g1_semantic_encode_held(
                REQUIRED_HELD_MASKS[index], &held_code) != REK_G1_SEMANTIC_OK ||
                !has_locomotion_held_code(table, held_code)) {
            return REK_G1_PUFFER_TABLE_HELD_COVERAGE_INVALID;
        }
    }
    for (uint16_t kick = 0; kick < table->kick_registry_count; kick++) {
        if (!has_kick_registry_index(table, kick)) {
            return REK_G1_PUFFER_TABLE_KICK_REGISTRY_INVALID;
        }
    }
    return REK_G1_PUFFER_OK;
}

RekG1PufferStatus rek_g1_puffer_init(
        RekG1PufferAdapter* adapter,
        const RekG1PufferActionTable* table) {
    RekG1PufferStatus status = rek_g1_puffer_validate_table(table);
    if (status != REK_G1_PUFFER_OK) return status;
    *adapter = (RekG1PufferAdapter){0};
    adapter->table = table;
    rek_g1_semantic_reset(&adapter->scheduler);
    return REK_G1_PUFFER_OK;
}

int rek_g1_puffer_category_legal(
        const RekG1PufferAdapter* adapter,
        uint32_t category,
        int translation_transition_settled,
        int action_busy) {
    const RekG1PufferActionTable* table = adapter->table;
    if (table == 0 || table->categories == 0 || category >= table->count) {
        return 0;
    }
    if (adapter->scheduler.active) {
        if (category == 0) return 1;
        if (adapter->scheduler.command.kind != REK_G1_SEMANTIC_KICK ||
                !adapter->scheduler.kick_accepted) {
            return 0;
        }
        uint8_t held_code = 0;
        return yaw_only_locomotion_held_code(
            table, category, &held_code);
    }
    if (category == 0) return 0;

    const RekG1PufferCategory* selected = &table->categories[category];
    if (selected->kind != REK_G1_PUFFER_START) return 0;
    if (selected->command.kind == REK_G1_SEMANTIC_LOCOMOTION) return 1;
    if (selected->command.kind != REK_G1_SEMANTIC_KICK) return 0;

    int translation_held =
        (adapter->scheduler.input_state.held &
         REK_G1_HELD_TRANSLATION_MASK) != 0;
    return !translation_held && translation_transition_settled && !action_busy;
}

RekG1PufferStatus rek_g1_puffer_write_mask(
        const RekG1PufferAdapter* adapter,
        int translation_transition_settled,
        int action_busy,
        uint8_t* mask,
        size_t mask_bytes) {
    const RekG1PufferActionTable* table = adapter->table;
    if (table == 0) return REK_G1_PUFFER_TABLE_NULL;
    if (mask == 0 || mask_bytes != table->count) {
        return REK_G1_PUFFER_TABLE_COUNT_INVALID;
    }
    memset(mask, 0, mask_bytes);
    int legal_count = 0;
    for (uint32_t category = 0; category < table->count; category++) {
        mask[category] = (uint8_t)rek_g1_puffer_category_legal(
            adapter,
            category,
            translation_transition_settled,
            action_busy);
        legal_count += mask[category] != 0;
    }
    return legal_count > 0 ?
        REK_G1_PUFFER_OK : REK_G1_PUFFER_PROTOCOL_ERROR;
}

RekG1PufferStep rek_g1_puffer_step(
        RekG1PufferAdapter* adapter,
        float action,
        RekG1InputTiming timing,
        int translation_transition_settled,
        int action_busy) {
    RekG1PufferStep result = {0};
    const RekG1PufferActionTable* table = adapter->table;
    if (table == 0) {
        result.status = REK_G1_PUFFER_TABLE_NULL;
        return result;
    }
    if (rek_g1_validate_input_timing(timing) != REK_G1_INPUT_ACCEPTED) {
        result.status = REK_G1_PUFFER_INPUT_TIMING_INVALID;
        return result;
    }
    result.status = REK_G1_PUFFER_OK;
    if (!isfinite(action)) {
        result.status = REK_G1_PUFFER_ACTION_NOT_FINITE;
        return result;
    }
    if (floorf(action) != action) {
        result.status = REK_G1_PUFFER_ACTION_NOT_INTEGRAL;
        return result;
    }
    if (action < 0.0f || action >= (float)table->count) {
        result.status = REK_G1_PUFFER_ACTION_OUT_OF_RANGE;
        return result;
    }

    result.category = (uint32_t)action;
    if (!rek_g1_puffer_category_legal(
            adapter,
            result.category,
            translation_transition_settled,
            action_busy)) {
        result.status = REK_G1_PUFFER_ACTION_MASKED;
        return result;
    }

    uint8_t active_kick_held_code = 0;
    int active_kick_input_update =
        adapter->scheduler.active &&
        adapter->scheduler.command.kind == REK_G1_SEMANTIC_KICK &&
        adapter->scheduler.kick_accepted &&
        yaw_only_locomotion_held_code(
            table, result.category, &active_kick_held_code);
    if (active_kick_input_update) {
        // Neutral/Q/E categories update desired keyboard state on this tick.
        // The active kick's registry identity, cursor, and remaining duration
        // are unchanged and the semantic tick still advances exactly once.
        adapter->scheduler.command.held_code = active_kick_held_code;
    } else if (result.category != 0) {
        RekG1SemanticCommand command =
            table->categories[result.category].command;
        if (command.kind == REK_G1_SEMANTIC_KICK) {
            // A kick suppresses effective yaw, but it does not release a Q/E
            // key that was already held. Carry the adapter's desired yaw into
            // the finite kick segment so its ramp continues while suppressed
            // and resumes without a false release after the segment.
            uint8_t held_code = 0;
            uint8_t held_yaw = adapter->scheduler.input_state.held &
                REK_G1_HELD_YAW_MASK;
            if (rek_g1_semantic_encode_held(held_yaw, &held_code) !=
                    REK_G1_SEMANTIC_OK) {
                result.status = REK_G1_PUFFER_PROTOCOL_ERROR;
                return result;
            }
            command.held_code = held_code;
        }
        RekG1SemanticStatus start_status = rek_g1_semantic_start(
            &adapter->scheduler,
            command,
            table->kick_registry_count);
        if (start_status != REK_G1_SEMANTIC_OK) {
            result.status = REK_G1_PUFFER_PROTOCOL_ERROR;
            return result;
        }
    }

    result.semantic = rek_g1_semantic_tick(
        &adapter->scheduler,
        timing,
        translation_transition_settled,
        action_busy);
    if (result.semantic.status != REK_G1_SEMANTIC_OK) {
        result.status = REK_G1_PUFFER_PROTOCOL_ERROR;
    }
    return result;
}
