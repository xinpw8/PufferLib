#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ocean/osrs/osrs_inventory_clicks.h"

static const struct {
    int item;
    uint32_t effect;
} EXPECTED_ITEM_EFFECTS[] = {
    {ITEM_TWISTED_BOW, OSRS_ITEM_EFFECT_TWISTED_BOW},
    {ITEM_VIRTUS_MASK, OSRS_ITEM_EFFECT_VIRTUS_PIECE},
    {ITEM_VIRTUS_ROBE_TOP, OSRS_ITEM_EFFECT_VIRTUS_PIECE},
    {ITEM_VIRTUS_ROBE_BOTTOM, OSRS_ITEM_EFFECT_VIRTUS_PIECE},
    {ITEM_CONFLICTION_GAUNTLETS, OSRS_ITEM_EFFECT_CONFLICTION},
    {ITEM_SANGUINESTI_STAFF, OSRS_ITEM_EFFECT_SANG_HEAL},
    {ITEM_RING_OF_RECOIL, OSRS_ITEM_EFFECT_RECOIL_RING},
    {ITEM_RING_OF_SUFFERING_RI, OSRS_ITEM_EFFECT_RECOIL_RING},
    {ITEM_LIGHTBEARER, OSRS_ITEM_EFFECT_LIGHTBEARER},
    {ITEM_DHAROKS_HELM, OSRS_ITEM_EFFECT_DHAROK_PIECE},
    {ITEM_DHAROKS_PLATELEGS, OSRS_ITEM_EFFECT_DHAROK_PIECE},
    {ITEM_ELYSIAN_SPIRIT_SHIELD, OSRS_ITEM_EFFECT_ELYSIAN},
    {ITEM_CRYSTAL_HELM, OSRS_ITEM_EFFECT_CRYSTAL_ARMOUR},
    {ITEM_CRYSTAL_BODY, OSRS_ITEM_EFFECT_CRYSTAL_ARMOUR},
    {ITEM_CRYSTAL_LEGS, OSRS_ITEM_EFFECT_CRYSTAL_ARMOUR},
    {ITEM_DRAGON_HUNTER_WAND, OSRS_ITEM_EFFECT_DRAGON_HUNTER_WAND},
    {ITEM_ECHO_BOOTS, OSRS_ITEM_EFFECT_ECHO_BOOTS},
    {ITEM_AMULET_OF_BLOOD_FURY, OSRS_ITEM_EFFECT_BLOOD_FURY},
    {ITEM_SERPENTINE_HELM, OSRS_ITEM_EFFECT_VENOM_IMMUNE},
    {ITEM_OSMUMTENS_FANG, OSRS_ITEM_EFFECT_FANG},
    {ITEM_TUMEKENS_SHADOW, OSRS_ITEM_EFFECT_TUMEKENS_SHADOW},
    {ITEM_VENATOR_BOW, OSRS_ITEM_EFFECT_VENATOR_BOUNCE},
};

static const int EXPECTED_ITEM_EFFECT_FREE_ITEMS[] = {
    ITEM_ABYSSAL_TENTACLE,
};

static int expected_gear_slot(EquipmentSlot slot) {
    switch (slot) {
        case SLOT_HEAD: return GEAR_SLOT_HEAD;
        case SLOT_CAPE: return GEAR_SLOT_CAPE;
        case SLOT_NECK: return GEAR_SLOT_NECK;
        case SLOT_WEAPON: return GEAR_SLOT_WEAPON;
        case SLOT_BODY: return GEAR_SLOT_BODY;
        case SLOT_SHIELD: return GEAR_SLOT_SHIELD;
        case SLOT_LEGS: return GEAR_SLOT_LEGS;
        case SLOT_HANDS: return GEAR_SLOT_HANDS;
        case SLOT_FEET: return GEAR_SLOT_FEET;
        case SLOT_RING: return GEAR_SLOT_RING;
        case SLOT_AMMO: return GEAR_SLOT_AMMO;
        default: return -1;
    }
}

static int verify_generated_gear_metadata(void) {
    int failures = 0;
    for (int item_idx = 0; item_idx < NUM_ITEMS; item_idx++) {
        uint16_t content_code =
            osrs_inventory_content_code_from_item((uint8_t)item_idx);
        const OsrsItemContentMetadata* metadata =
            osrs_item_content_metadata(content_code);
        const Item* item = &ITEM_DATABASE[item_idx];
        if (metadata->item_idx != item_idx ||
                metadata->item != item ||
                metadata->raw_osrs_id != item->item_id ||
                metadata->gear_slot != expected_gear_slot(item->slot) ||
                metadata->attack_style != get_item_attack_style(item_idx) ||
                metadata->click_action != OSRS_CLICK_EQUIP ||
                metadata->consumable_kind != OSRS_CONSUMABLE_NONE ||
                metadata->dose_count != 0 ||
                metadata->next_content_code != 0) {
            fprintf(stderr, "item %d generated metadata mismatch\n", item_idx);
            failures++;
        }
    }
    return failures;
}

int main(void) {
    const size_t n_expected =
        sizeof(EXPECTED_ITEM_EFFECTS) / sizeof(EXPECTED_ITEM_EFFECTS[0]);
    const size_t n_expected_effect_free =
        sizeof(EXPECTED_ITEM_EFFECT_FREE_ITEMS) /
        sizeof(EXPECTED_ITEM_EFFECT_FREE_ITEMS[0]);
    int failures = 0;

    for (size_t e = 0; e < n_expected; e++) {
        if (EXPECTED_ITEM_EFFECTS[e].effect == OSRS_ITEM_EFFECT_NONE) {
            fprintf(stderr,
                    "FAIL: EXPECTED_ITEM_EFFECTS row %zu lists OSRS_ITEM_EFFECT_NONE; "
                    "drop the row instead of pinning NONE\n",
                    e);
            failures++;
        }
    }

    for (size_t e = 0; e < n_expected_effect_free; e++) {
        int item = EXPECTED_ITEM_EFFECT_FREE_ITEMS[e];
        if (item < 0 || item >= NUM_ITEMS) {
            fprintf(stderr,
                    "FAIL: EXPECTED_ITEM_EFFECT_FREE_ITEMS row %zu has invalid "
                    "item index %d\n",
                    e, item);
            failures++;
            continue;
        }
        if (ITEM_DATABASE[item].effect_mask != OSRS_ITEM_EFFECT_NONE) {
            fprintf(stderr,
                    "FAIL: %s (ITEM idx %d): expected OSRS_ITEM_EFFECT_NONE, "
                    "got 0x%X\n",
                    ITEM_DATABASE[item].name, item,
                    ITEM_DATABASE[item].effect_mask);
            failures++;
        }
    }

    int non_none_seen = 0;
    for (int i = 0; i < NUM_ITEMS; i++) {
        uint32_t expected = OSRS_ITEM_EFFECT_NONE;
        for (size_t e = 0; e < n_expected; e++) {
            if (EXPECTED_ITEM_EFFECTS[e].item == i) {
                expected = EXPECTED_ITEM_EFFECTS[e].effect;
                break;
            }
        }

        uint32_t actual = ITEM_DATABASE[i].effect_mask;
        if (actual != expected) {
            fprintf(stderr,
                    "FAIL: %s (ITEM idx %d): effect_mask=0x%X, expected 0x%X\n",
                    ITEM_DATABASE[i].name, i, actual, expected);
            failures++;
        }
        if (actual != OSRS_ITEM_EFFECT_NONE) non_none_seen++;
    }

    if (non_none_seen != (int)n_expected) {
        fprintf(stderr,
                "FAIL: %d items carry an effect mask, expected %zu "
                "(item DB drifted from the guard table)\n",
                non_none_seen, n_expected);
        failures++;
    }

    failures += verify_generated_gear_metadata();
    if (failures) {
        fprintf(stderr, "test_osrs_item_effect_masks: %d failures\n", failures);
        return 1;
    }

    printf("test_osrs_item_effect_masks: OK (%zu effect items pinned, %d/%d effect-free)\n",
           n_expected, NUM_ITEMS - (int)n_expected, NUM_ITEMS);
    return 0;
}
