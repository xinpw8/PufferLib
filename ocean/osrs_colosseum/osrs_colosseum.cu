static constexpr OsrsEntityBranchDescriptor OSRS_COLOSSEUM_ENTITY_BRANCHES[] = {
    {
        .obs_start = 80,
        .num_records = NUM_GEAR_SLOTS,
        .obs_features = 1,
        .type_onehot = 0,
        .type_code_scale = OSRS_ITEM_OBS_CODE_SCALE,
        .expansion = OSRS_ENTITY_BRANCH_ITEM_TABLE,
    },
    {
        .obs_start = 101,
        .num_records = 24,
        .obs_features = 23,
        .type_onehot = 12,
        .type_code_scale = 1,
        .expansion = OSRS_ENTITY_BRANCH_TYPE_ONEHOT,
    },
};

static_assert(12 + 23 - 1 == 34);
static_assert(101 + 24 * 23 <= 904);

static constexpr OsrsEntityEncoderDescriptor OSRS_COLOSSEUM_ENTITY_DESCRIPTOR = {
    .branches = OSRS_COLOSSEUM_ENTITY_BRANCHES,
    .num_branches = 2,
};
