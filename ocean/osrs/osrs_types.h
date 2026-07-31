#ifndef OSRS_TYPES_H
#define OSRS_TYPES_H

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include "osrs_interaction.h"

#define NUM_AGENTS 2
#define MAX_PENDING_HITS 8
#define HISTORY_SIZE 5

#define TICK_DURATION_MS 600
#define MAX_EPISODE_TICKS 300

#define WILD_MIN_X 2940
#define WILD_MAX_X 3392
#define WILD_MIN_Y 3525
#define WILD_MAX_Y 3968
#define FIGHT_AREA_BASE_X 3041
#define FIGHT_AREA_BASE_Y 3530
#define FIGHT_AREA_WIDTH 61
#define FIGHT_AREA_HEIGHT 28
#define FIGHT_NEARBY_RADIUS 5

#define ONLY_SWITCH_GEAR_WHEN_ATTACK_SOON 1

#define ICE_RUSH_LEVEL 58
#define ICE_BURST_LEVEL 70
#define ICE_BLITZ_LEVEL 82
#define ICE_BARRAGE_LEVEL 94

#define BLOOD_RUSH_LEVEL 56
#define BLOOD_BURST_LEVEL 68
#define BLOOD_BLITZ_LEVEL 80
#define BLOOD_BARRAGE_LEVEL 92

#define ICE_RUSH_MAX_HIT 18
#define ICE_BURST_MAX_HIT 22
#define ICE_BLITZ_MAX_HIT 26
#define ICE_BARRAGE_MAX_HIT 30

#define BLOOD_RUSH_MAX_HIT 15
#define BLOOD_BURST_MAX_HIT 21
#define BLOOD_BLITZ_MAX_HIT 25
#define BLOOD_BARRAGE_MAX_HIT 29

#define NUM_GEAR_SLOTS 11

#define NUM_ACTION_HEADS 9

#define HEAD_LOADOUT    0
#define HEAD_COMBAT     1
#define HEAD_OVERHEAD   2
#define HEAD_FOOD       3
#define HEAD_POTION     4
#define HEAD_KARAMBWAN  5
#define HEAD_VENG       6
#define HEAD_OFFENSIVE  7
#define HEAD_MOVE       8

#define LOADOUT_DIM     9
#define COMBAT_DIM     13
#define OVERHEAD_DIM    7
#define FOOD_DIM        2
#define POTION_DIM      5
#define KARAMBWAN_DIM   2
#define VENG_DIM        2
#define OFFENSIVE_DIM   5
#define MOVE_DIM       25

#define ACTION_MASK_SIZE (LOADOUT_DIM + COMBAT_DIM + OVERHEAD_DIM + \
    FOOD_DIM + POTION_DIM + KARAMBWAN_DIM + VENG_DIM + OFFENSIVE_DIM + MOVE_DIM)

static const int ACTION_HEAD_DIMS[NUM_ACTION_HEADS] = {
    LOADOUT_DIM,
    COMBAT_DIM,
    OVERHEAD_DIM,
    FOOD_DIM,
    POTION_DIM,
    KARAMBWAN_DIM,
    VENG_DIM,
    OFFENSIVE_DIM,
    MOVE_DIM,
};

#define NUM_ITEM_STATS 18

#define MAX_ITEMS_PER_SLOT 10

#define NUM_DYNAMIC_GEAR_SLOTS 8

#define SLOT_NUM_OBSERVATIONS 221

#define MAXED_BASE_ATTACK 99
#define MAXED_BASE_STRENGTH 99
#define MAXED_BASE_DEFENCE 99
#define LMS_BASE_DEFENCE 75
#define MAXED_BASE_RANGED 99
#define MAXED_BASE_MAGIC 99
#define MAXED_BASE_PRAYER 77
#define MAXED_BASE_HITPOINTS 99

#define MAXED_FOOD_COUNT 11
#define MAXED_KARAMBWAN_COUNT 2
#define MAXED_BREW_DOSES 4
#define MAXED_RESTORE_DOSES 8
#define MAXED_COMBAT_POTION_DOSES 4
#define MAXED_RANGED_POTION_DOSES 4

#define RUN_ENERGY_RECOVER_TICKS 3
#define OSRS_RUN_ENERGY_UNITS_PER_PERCENT 100
#define OSRS_RUN_ENERGY_FULL 10000

typedef enum {
    ATTACK_STYLE_NONE = 0,
    ATTACK_STYLE_MELEE,
    ATTACK_STYLE_RANGED,
    ATTACK_STYLE_MAGIC
} AttackStyle;

typedef enum {
    MELEE_STYLE_STAB = 0,
    MELEE_STYLE_SLASH,
    MELEE_STYLE_CRUSH,
} MeleeStyle;

typedef enum {
    PRAYER_NONE = 0,
    PRAYER_PROTECT_MAGIC,
    PRAYER_PROTECT_RANGED,
    PRAYER_PROTECT_MELEE,
    PRAYER_SMITE,
    PRAYER_REDEMPTION
} OverheadPrayer;

typedef enum {
    GEAR_MAGE = 0,
    GEAR_RANGED,
    GEAR_MELEE,
    GEAR_SPEC,
    GEAR_TANK
} GearSet;

typedef enum {
    OFFENSIVE_PRAYER_NONE = 0,
    OFFENSIVE_PRAYER_MELEE_LOW,
    OFFENSIVE_PRAYER_RANGED_LOW,
    OFFENSIVE_PRAYER_MAGIC_LOW,
    OFFENSIVE_PRAYER_PIETY,
    OFFENSIVE_PRAYER_RIGOUR,
    OFFENSIVE_PRAYER_AUGURY
} OffensivePrayer;

typedef enum {
    FIGHT_STYLE_ACCURATE = 0,
    FIGHT_STYLE_AGGRESSIVE,
    FIGHT_STYLE_CONTROLLED,
    FIGHT_STYLE_DEFENSIVE,
    FIGHT_STYLE_RAPID,
    FIGHT_STYLE_LONGRANGE,
    FIGHT_STYLE_AUTOCAST,
    FIGHT_STYLE_DEFENSIVE_AUTOCAST,
} FightStyle;

typedef enum {
    MELEE_BONUS_STAB = 0,
    MELEE_BONUS_SLASH,
    MELEE_BONUS_CRUSH
} MeleeBonusType;

typedef enum {
    MELEE_SPEC_NONE = 0,
    MELEE_SPEC_AGS,
    MELEE_SPEC_DRAGON_CLAWS,
    MELEE_SPEC_GRANITE_MAUL,
    MELEE_SPEC_DRAGON_DAGGER,
    MELEE_SPEC_VOIDWAKER,
    MELEE_SPEC_DWH,
    MELEE_SPEC_BGS,
    MELEE_SPEC_ZGS,
    MELEE_SPEC_SGS,
    MELEE_SPEC_ANCIENT_GS,
    MELEE_SPEC_VESTAS,
    MELEE_SPEC_ABYSSAL_DAGGER,
    MELEE_SPEC_DRAGON_LONGSWORD,
    MELEE_SPEC_DRAGON_MACE,
    MELEE_SPEC_ABYSSAL_BLUDGEON
} MeleeSpecWeapon;

typedef enum {
    RANGED_SPEC_NONE = 0,
    RANGED_SPEC_DARK_BOW,
    RANGED_SPEC_BALLISTA,
    RANGED_SPEC_ACB,
    RANGED_SPEC_ZCB,
    RANGED_SPEC_DRAGON_KNIFE,
    RANGED_SPEC_MSB,
    RANGED_SPEC_MORRIGANS
} RangedSpecWeapon;

typedef enum {
    MAGIC_SPEC_NONE = 0,
    MAGIC_SPEC_VOLATILE_STAFF
} MagicSpecWeapon;

typedef enum {
    GEAR_SLOT_HEAD = 0,
    GEAR_SLOT_CAPE,
    GEAR_SLOT_NECK,
    GEAR_SLOT_AMMO,
    GEAR_SLOT_WEAPON,
    GEAR_SLOT_SHIELD,
    GEAR_SLOT_BODY,
    GEAR_SLOT_LEGS,
    GEAR_SLOT_HANDS,
    GEAR_SLOT_FEET,
    GEAR_SLOT_RING,
} GearSlotIndex;

static const int DYNAMIC_GEAR_SLOTS[NUM_DYNAMIC_GEAR_SLOTS] = {
    GEAR_SLOT_WEAPON, GEAR_SLOT_SHIELD, GEAR_SLOT_BODY, GEAR_SLOT_LEGS,
    GEAR_SLOT_HEAD, GEAR_SLOT_CAPE, GEAR_SLOT_NECK, GEAR_SLOT_RING
};

typedef enum {
    LOADOUT_KEEP = 0,
    LOADOUT_MELEE,
    LOADOUT_RANGE,
    LOADOUT_MAGE,
    LOADOUT_TANK,
    LOADOUT_SPEC_MELEE,
    LOADOUT_SPEC_RANGE,
    LOADOUT_SPEC_MAGIC,
    LOADOUT_GMAUL,
} LoadoutAction;

#define ATTACK_NONE      0
#define ATTACK_ATK       1
#define ATTACK_ICE       2
#define ATTACK_BLOOD     3
#define MOVE_ADJACENT    4
#define MOVE_UNDER       5
#define MOVE_DIAGONAL    6
#define MOVE_FARCAST_2   7
#define MOVE_FARCAST_3   8
#define MOVE_FARCAST_4   9
#define MOVE_FARCAST_5  10
#define MOVE_FARCAST_6  11
#define MOVE_FARCAST_7  12
#define MOVE_NONE ATTACK_NONE

static inline int is_attack_action(int v) { return v >= ATTACK_ATK && v <= ATTACK_BLOOD; }
static inline int is_move_action(int v) { return v >= MOVE_ADJACENT && v <= MOVE_FARCAST_7; }

typedef enum {
    OVERHEAD_NONE = 0,
    OVERHEAD_MAGE,
    OVERHEAD_RANGED,
    OVERHEAD_MELEE,
    OVERHEAD_SMITE,
    OVERHEAD_REDEMPTION,
} OverheadAction;

typedef enum {
    FOOD_NONE = 0,
    FOOD_EAT,
} FoodAction;

typedef enum {
    POTION_NONE = 0,
    POTION_BREW,
    POTION_RESTORE,
    POTION_COMBAT,
    POTION_RANGED,
    POTION_ANTIVENOM,
    POTION_BASTION,
    POTION_STAMINA,
    POTION_PRAYER_POT,
    POTION_SURGE,
} PotionAction;

typedef enum {
    KARAM_NONE = 0,
    KARAM_EAT,
} KaramAction;

typedef enum {
    VENG_NONE = 0,
    VENG_CAST,
} VengAction;

#define OSRS_INFERNO_IDLE_PHASE_COUNT 6

typedef struct {
    int stab_attack;
    int slash_attack;
    int crush_attack;
    int magic_attack;
    int ranged_attack;
    int stab_defence;
    int slash_defence;
    int crush_defence;
    int magic_defence;
    int ranged_defence;
    int melee_strength;
    int ranged_strength;
    int magic_strength;
    int attack_speed;
    int attack_range;
} GearBonuses;

typedef struct {
    int magic_attack;
    int magic_strength;
    int ranged_attack;
    int ranged_strength;
    int melee_attack;
    int melee_strength;
    int magic_defence;
    int ranged_defence;
    int melee_defence;
} VisibleGearBonuses;

typedef struct {
    int damage;
    int ticks_until_hit;
    AttackStyle attack_type;
    int is_special;
    int hit_success;
    int freeze_ticks;
    int heal_percent;
    int drain_type;
    int drain_percent;
    int flat_heal;
    int is_morr_bleed;
    OverheadPrayer defender_prayer_at_attack;
} PendingHit;

typedef enum {
    ENTITY_PLAYER = 0,
    ENTITY_NPC = 1,
} EntityType;

typedef enum {
    OSRS_MAGIC_ATTACK_NONE = 0,
    OSRS_MAGIC_ATTACK_ANCIENT_ICE,
    OSRS_MAGIC_ATTACK_ANCIENT_BLOOD,
    OSRS_MAGIC_ATTACK_STANDARD_SPELL,
    OSRS_MAGIC_ATTACK_POWERED_STAFF,
} OsrsMagicAttackKind;

typedef enum {
    OSRS_TARGET_NONE = 0,
    OSRS_TARGET_PLAYER,
    OSRS_TARGET_NPC,
} OsrsTargetKind;

typedef struct {
    OsrsTargetKind kind;
    int id;
} OsrsTargetRef;

typedef enum {
    OSRS_TARGET_CLASS_STANDARD = 0,
    OSRS_TARGET_CLASS_DRAGON,
} OsrsTargetClass;

typedef struct {
    int magic_level;
    int magic_attack_bonus;
    OsrsTargetClass target_class;
} OsrsTargetEffectContext;

typedef enum {
    OSRS_RECOIL_SOURCE_NONE = 0,
    OSRS_RECOIL_SOURCE_RING_OF_RECOIL,
    OSRS_RECOIL_SOURCE_RING_OF_SUFFERING_RI,
} OsrsRecoilSource;

typedef enum {
    OSRS_SPEC_REGEN_MODE_NORMAL = 0,
    OSRS_SPEC_REGEN_MODE_LIGHTBEARER,
} OsrsSpecRegenMode;

typedef struct {
    uint32_t effect_mask;
    uint8_t weapon_item;
    uint8_t ring_item;
    uint8_t shield_item;
    uint8_t virtus_piece_count;
    uint8_t dharok_piece_count;
    uint8_t crystal_armour_points;
    OsrsRecoilSource recoil_source;
    OsrsSpecRegenMode spec_regen_mode;
} OsrsEquipmentEffectProfile;

typedef struct {
    int special_regen_ticks;
    int recoil_charges;
    int echo_boot_charges;
    uint8_t confliction_is_primed;
    uint8_t confliction_weapon_item;
    OsrsMagicAttackKind confliction_magic_kind;
    OsrsTargetRef confliction_target;
} OsrsItemEffectState;

typedef struct {
    EntityType entity_type;
    int npc_def_id;
    int npc_visible;
    int npc_size;
    int npc_anim_id;

    int is_lms;

    int base_attack;
    int base_strength;
    int base_defence;
    int base_ranged;
    int base_magic;
    int base_prayer;
    int base_hitpoints;

    int current_attack;
    int current_strength;
    int current_defence;
    int current_ranged;
    int current_magic;
    int current_prayer;
    int current_hitpoints;

    int special_energy;
    int spec_regen_active;
    int spec_armed;
    OsrsInteraction interaction;
    OsrsItemEffectState item_effect_state;

    GearSet current_gear;
    GearSet visible_gear;

    int food_count;
    int karambwan_count;
    int brew_doses;
    int restore_doses;
    int prayer_pot_doses;
    int combat_potion_doses;
    int ranged_potion_doses;
    int bastion_doses;
    int stamina_doses;
    int antivenom_doses;
    int saturated_heart_count;
    int saturated_heart_active_ticks;

    int attack_timer;
    int attack_timer_uncapped;
    int has_attack_timer;
    int food_timer;
    int potion_timer;
    int karambwan_timer;

    uint8_t consumable_used_this_tick;
    int last_food_heal;
    int last_food_waste;
    int last_karambwan_heal;
    int last_karambwan_waste;
    int last_brew_heal;
    int last_brew_waste;
    int last_potion_type;
    int last_potion_was_waste;

    int frozen_ticks;
    int freeze_immunity_ticks;

    int veng_active;
    int veng_cooldown;

    OverheadPrayer prayer;

    OverheadPrayer prayer_display;
    OffensivePrayer offensive_prayer;
    FightStyle fight_style;
    int autocast_enabled;
    int autocast_defensive;
    int autocast_spell;
    int prayer_drain_counter;

    uint8_t prayer_just_activated;
    uint8_t offensive_prayer_just_activated;

    int x, y;
    int dest_x, dest_y;
    int is_moving;
    int is_running;
    int run_energy;
    int run_recovery_ticks;
    int last_obs_target_x;
    int last_obs_target_y;

    int just_attacked;
    AttackStyle last_attack_style;
    int last_queued_hit_damage;
    int attack_was_on_prayer;
    int attack_click_canceled;
    int attack_click_ready;
    int last_attack_dx;
    int last_attack_dy;
    int last_attack_dist;

    PendingHit pending_hits[MAX_PENDING_HITS];
    int num_pending_hits;
    int damage_applied_this_tick;
    int did_attack_auto_move;

    int hit_landed_this_tick;
    int hit_was_successful;
    int hit_damage;
    AttackStyle hit_style;
    OverheadPrayer hit_defender_prayer;
    int hit_was_on_prayer;
    int hit_attacker_idx;
    int freeze_applied_this_tick;
    int elysian_proc_this_tick;

    int morr_dot_remaining;
    int morr_dot_tick_counter;

    float last_target_health_percent;
    float tick_damage_scale;
    float damage_dealt_scale;
    float damage_received_scale;

    int total_target_hit_count;
    int target_hit_melee_count;
    int target_hit_ranged_count;
    int target_hit_magic_count;
    int target_hit_off_prayer_count;
    int target_hit_correct_count;

    int total_target_pray_count;
    int target_pray_melee_count;
    int target_pray_ranged_count;
    int target_pray_magic_count;
    int target_pray_correct_count;

    int player_hit_melee_count;
    int player_hit_ranged_count;
    int player_hit_magic_count;

    int player_pray_melee_count;
    int player_pray_ranged_count;
    int player_pray_magic_count;

    AttackStyle recent_target_attack_styles[HISTORY_SIZE];
    AttackStyle recent_player_attack_styles[HISTORY_SIZE];
    AttackStyle recent_target_prayer_styles[HISTORY_SIZE];
    AttackStyle recent_player_prayer_styles[HISTORY_SIZE];
    int recent_target_prayer_correct[HISTORY_SIZE];
    int recent_target_hit_correct[HISTORY_SIZE];
    int recent_target_attack_index;
    int recent_player_attack_index;
    int recent_target_prayer_index;
    int recent_player_prayer_index;
    int recent_target_prayer_correct_index;
    int recent_target_hit_correct_index;

    int target_magic_accuracy;
    int target_magic_strength;
    int target_ranged_accuracy;
    int target_ranged_strength;
    int target_melee_accuracy;
    int target_melee_strength;
    int target_magic_gear_magic_defence;
    int target_magic_gear_ranged_defence;
    int target_magic_gear_melee_defence;
    int target_ranged_gear_magic_defence;
    int target_ranged_gear_ranged_defence;
    int target_ranged_gear_melee_defence;
    int target_melee_gear_magic_defence;
    int target_melee_gear_ranged_defence;
    int target_melee_gear_melee_defence;

    int player_prayed_correct;
    int target_prayed_correct;

    float total_damage_dealt;
    float total_damage_received;

    int is_lunar_spellbook;
    int observed_target_lunar_spellbook;
    int has_blood_fury;

    MeleeSpecWeapon melee_spec_weapon;
    RangedSpecWeapon ranged_spec_weapon;
    MagicSpecWeapon magic_spec_weapon;

    float bolt_proc_damage;
    int bolt_ignores_defense;

    uint8_t equipped[NUM_GEAR_SLOTS];

    uint8_t inventory[NUM_GEAR_SLOTS][MAX_ITEMS_PER_SLOT];

    uint8_t num_items_in_slot[NUM_GEAR_SLOTS];

    GearBonuses slot_cached_bonuses;
    OsrsEquipmentEffectProfile equipment_effect_profile;
    int slot_gear_dirty;

    AttackStyle attack_style_this_tick;
    int magic_type_this_tick;
    int used_special_this_tick;
    int ate_food_this_tick;
    int ate_karambwan_this_tick;
    int ate_brew_this_tick;
    int cast_veng_this_tick;
    int clicks_this_tick;

    float prev_hp_percent;

    int gui_max_hit;
    int gui_attack_speed;
    int gui_attack_range;
    int gui_strength_bonus;
} Player;

typedef struct Log {
    float episode_return;
    float episode_length;
    float wins;
    float damage_dealt;
    float damage_received;
    float wave;
    float prayer_correct;
    float prayer_total;
    float idle_ticks;
    float attack_ready_no_attack_ticks;
    float target_available_no_attack_ticks;
    float safe_attack_opportunity_missed_ticks;
    float progressless_ticks;
    float npc_pressure_if_ready_count;
    float npc_pressure_this_tick_count;
    float npc_pressure_max_incoming_hit;
    float attack_ready_no_attack_ticks_by_phase[OSRS_INFERNO_IDLE_PHASE_COUNT];
    float target_available_no_attack_ticks_by_phase[OSRS_INFERNO_IDLE_PHASE_COUNT];
    float safe_attack_opportunity_missed_ticks_by_phase[OSRS_INFERNO_IDLE_PHASE_COUNT];
    float progressless_ticks_by_phase[OSRS_INFERNO_IDLE_PHASE_COUNT];
    float brews_used;
    float npc_kills;
    float zulrah_tier_n[3];
    float zulrah_tier_wins[3];
    float zulrah_tier_score_sum[3];
    float zulrah_tier_damage_received[3];
    float zulrah_tier_episode_length[3];
    float zulrah_tier_cloud_occupancy_ticks[3];
    float zulrah_tier_cloud_damage_received[3];
    float cloud_occupancy_ticks;
    float cloud_occupancy_frac;
    float cloud_damage_received;
    float active_cloud_count_ticks;
    float pending_cloud_count_ticks;
    float zulrah_kills;
    float offensive_prayer_attacks;
    float offensive_prayer_correct;
    float offensive_prayer_attacks_by_style[4];
    float offensive_prayer_correct_by_style[4];
    float brews_remaining;
    float restores_remaining;
    float food_remaining;
    float karambwan_remaining;
    float spec_energy_remaining;
    float attacks_landed;
    float off_prayer_hits;
    float behind_shield_pct;
    float min_zuk_hp_seen;
    float hp_restored;
    float zuk_healer_damage;
    float min_zuk_hp_normal;
    float n_normal;

    float phase_reached_normal_sum;

    float count_min_hp_le_240_normal;

    float count_all_zuk_healers_dead_normal;
    float count_healer_resolved_20_normal;
    float damage_after_all_zuk_healers_dead_normal_sum;
    float hp_restored_after_240_normal_sum;
    float spark_damage_after_240_normal_sum;

    float count_died_with_zuk_healer_alive_normal;
    float count_died_after_240_normal;
    float start_wave;
    float n;
    float post_healer_set_damage_reward_coeff_normal_sum;
    float post_healer_set_kill_bonus_coeff_normal_sum;
    float post_healer_set_alive_penalty_coeff_normal_sum;
    float post_healer_set_alive_penalty_cap_normal_sum;
    float post_healer_set_damage_reward_normal_sum;
    float post_healer_set_kill_bonus_reward_normal_sum;
    float post_healer_set_alive_penalty_normal_sum;
    float post_healer_set_pressure_normal_sum;
    float action_mask_checks_normal_sum;

    float hist_score_bank[8];
    float hist_n_bank[8];

    float colo_pray_faced_by_type[12];
    float colo_pray_correct_by_type[12];
    float colo_offpray_damage_by_type[12];
    float colo_total_damage_by_type[12];

    float colo_death_by_type[12];
    float colo_death_fatal_damage;

    float colo_offpray_damage_conflict;
    float colo_offpray_damage_solo;
    float colo_death_on_conflict_tick;

    float colo_death_dmg_unprayable;
    float colo_death_dmg_offpray;
    float colo_death_dmg_prayed;
    float colo_death_dmg_self;
    float colo_death_heal_remaining;
    float colo_farm_damage;
    float colo_typeless_damage_by_type[12];
    float colo_outcome_score;
    float colo_min_sol_hp;

    float colo_max_depth_reached;
} Log;

typedef struct {
    float damage_dealt_coef;
    float damage_received_coef;
    float correct_prayer_bonus;
    float wrong_prayer_penalty;
    float prayer_switch_no_attack_penalty;
    float off_prayer_hit_bonus;
    float melee_frozen_penalty;
    float wasted_eat_penalty;
    float premature_eat_penalty;
    float magic_no_staff_penalty;
    float gear_mismatch_penalty;
    float spec_off_prayer_bonus;
    float spec_low_defence_bonus;
    float spec_low_hp_bonus;
    float smart_triple_eat_bonus;
    float wasted_triple_eat_penalty;
    float damage_burst_bonus;
    int   damage_burst_threshold;
    float premature_eat_threshold;
    float ko_bonus;
    float ko_supplies_bonus_coef;
    float wasted_resources_penalty;
    float shaping_scale;
    int   enabled;
    int   prayer_penalty_enabled;
    int   click_penalty_enabled;
    int   click_penalty_threshold;
    float click_penalty_coef;
} RewardShapingConfig;

typedef enum {
    OPP_NONE = 0,
    OPP_TRUE_RANDOM,
    OPP_PANICKING,
    OPP_WEAK_RANDOM,
    OPP_SEMI_RANDOM,
    OPP_STICKY_PRAYER,
    OPP_RANDOM_EATER,
    OPP_PRAYER_ROOKIE,
    OPP_IMPROVED,
    OPP_MIXED_EASY,
    OPP_MIXED_MEDIUM,
    OPP_ONETICK,
    OPP_UNPREDICTABLE_IMPROVED,
    OPP_UNPREDICTABLE_ONETICK,
    OPP_MIXED_HARD,
    OPP_MIXED_HARD_BALANCED,
    OPP_PFSP,
    OPP_NOVICE_NH,
    OPP_APPRENTICE_NH,
    OPP_COMPETENT_NH,
    OPP_INTERMEDIATE_NH,
    OPP_ADVANCED_NH,
    OPP_PROFICIENT_NH,
    OPP_EXPERT_NH,
    OPP_MASTER_NH,
    OPP_SAVANT_NH,
    OPP_NIGHTMARE_NH,
    OPP_VENG_FIGHTER,
    OPP_BLOOD_HEALER,
    OPP_GMAUL_COMBO,
    OPP_RANGE_KITER,
    OPP_SELFPLAY,
} OpponentType;

#define MAX_OPPONENT_POOL 32

typedef struct {
    OpponentType pool[MAX_OPPONENT_POOL];
    int cum_weights[MAX_OPPONENT_POOL];
    int pool_size;
    int active_pool_idx;
    float wins[MAX_OPPONENT_POOL];
    float episodes[MAX_OPPONENT_POOL];
} PFSPState;

typedef struct {
    OpponentType type;
    OpponentType active_sub_policy;
    int chosen_prayer;
    int chosen_style;
    int current_prayer;
    int current_prayer_set;
    int food_cooldown;
    int potion_cooldown;
    int karambwan_cooldown;

    int fake_switch_pending;
    int fake_switch_style;
    int opponent_prayer_at_fake;
    int fake_switch_failed;
    int pending_prayer_value;
    int pending_prayer_delay;
    int last_target_gear_style;

    float eat_triple_threshold;
    float eat_double_threshold;
    float eat_brew_threshold;

    float prayer_accuracy;
    float off_prayer_rate;
    float offensive_prayer_rate;
    float action_delay_chance;
    float mistake_rate;

    float read_chance;
    int has_read_this_tick;
    AttackStyle read_agent_style;
    OverheadPrayer read_agent_prayer;
    int read_agent_moving;

    int prev_dist_to_target;
    int target_fleeing_ticks;

    int combo_state;
    float ko_threshold;

    float offensive_prayer_miss;

    float style_bias[3];
} OpponentState;

typedef struct {
    int is_pvp_arena;
    int use_c_opponent;
    int use_c_opponent_p0;
    int use_external_opponent_actions;
    int external_opponent_actions[NUM_ACTION_HEADS];
    OpponentState opponent;
    OpponentState opponent_p0;
    PFSPState pfsp;
    float gear_tier_weights[4];

    int walk_dest_x[NUM_AGENTS];
    int walk_dest_y[NUM_AGENTS];
} OsrsPvpRuntime;

typedef struct {
    float* agent_obs;
    float* agent_obs_p1;
    unsigned char* selfplay_mask;
    int* agent_actions;
    float* agent_rewards;
    unsigned char* agent_terminals;
} OsrsOceanBuffers;

#define OCEAN_OBS_SIZE (SLOT_NUM_OBSERVATIONS + ACTION_MASK_SIZE)

typedef struct {
    Log log;

    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* action_masks;
    unsigned char action_masks_agents;
    int num_agents;

    Player players[NUM_AGENTS];

    int tick;
    int episode_over;
    int winner;
    int auto_reset;
    int pid_holder;
    int pid_shuffle_countdown;

    int is_lms;

    uint32_t rng_state;
    uint32_t rng_seed;
    uint32_t rng_reset_count;
    int has_rng_seed;

    int pending_actions[NUM_AGENTS * NUM_ACTION_HEADS];
    int last_executed_actions[NUM_AGENTS * NUM_ACTION_HEADS];

    RewardShapingConfig shaping;

    OsrsPvpRuntime pvp_runtime;

    const void* encounter_def;
    void* encounter_state;
    void* encounter_context;

    void* collision_map;

    void* client;

    OsrsOceanBuffers ocean_io;
    float _episode_return;

    float _obs_buf[NUM_AGENTS * SLOT_NUM_OBSERVATIONS];
    int _acts_buf[NUM_AGENTS * NUM_ACTION_HEADS];
    float _rews_buf[NUM_AGENTS];
    unsigned char _terms_buf[NUM_AGENTS];
    unsigned char _masks_buf[NUM_AGENTS * ACTION_MASK_SIZE];

} OsrsEnv;

static inline int abs_int(int val) {
    return val < 0 ? -val : val;
}

static inline int min_int(int a, int b) {
    return a < b ? a : b;
}

static inline int max_int(int a, int b) {
    return a > b ? a : b;
}

static inline int clamp(int val, int min, int max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

static inline int osrs_run_energy_percent(int run_energy) {
    return clamp(run_energy / OSRS_RUN_ENERGY_UNITS_PER_PERCENT, 0, 100);
}

static inline float clampf(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

static inline int chebyshev_distance(int x1, int y1, int x2, int y2) {
    int dx = x1 - x2;
    int dy = y1 - y2;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    return (dx > dy) ? dx : dy;
}

static inline int is_in_melee_range(Player* p, Player* t) {
    int dx = abs_int(p->x - t->x);
    int dy = abs_int(p->y - t->y);
    return (dx == 1 && dy == 0) || (dx == 0 && dy == 1);
}

/** state must be non-zero or the stream sticks at zero. */
static inline uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline int rand_int(OsrsEnv* env, int max) {
    if (max <= 0) return 0;
    return xorshift32(&env->rng_state) % max;
}

static inline float rand_float(OsrsEnv* env) {
    return (float)xorshift32(&env->rng_state) / (float)UINT32_MAX;
}

static inline int is_in_wilderness(int x, int y) {
    return x >= WILD_MIN_X && x <= WILD_MAX_X && y >= WILD_MIN_Y && y <= WILD_MAX_Y;
}

static inline int tile_hash(int x, int y) {
    return (x << 15) | y;
}

static inline int remaining_ticks(int ticks) {
    return ticks > 0 ? ticks : 0;
}

static inline int get_attack_timer_uncapped(Player* p) {
    return p->has_attack_timer ? p->attack_timer_uncapped : -100;
}

static inline int can_attack_now(Player* p) {
    if (!p->has_attack_timer) return 1;
    return p->attack_timer < 0;
}

static inline int can_move(Player* p) {
    return p->frozen_ticks <= 0;
}

static inline float ratio_or_zero(int numerator, int denominator) {
    if (denominator == 0) {
        return 0.0f;
    }
    return (float)numerator / (float)denominator;
}

static inline float confidence_scale(int count) {
    if (count >= 10) {
        return 1.0f;
    }
    return (float)count / 10.0f;
}

#define RECOIL_MAX_CHARGES 40

#endif
