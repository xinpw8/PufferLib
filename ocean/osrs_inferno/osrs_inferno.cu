static constexpr OsrsEntityBranchDescriptor OSRS_INFERNO_ENTITY_BRANCHES[] = {
    {
        .obs_start = 80,
        .num_records = NUM_GEAR_SLOTS,
        .obs_features = 1,
        .type_onehot = 0,
        .type_code_scale = OSRS_ITEM_OBS_CODE_SCALE,
        .expansion = OSRS_ENTITY_BRANCH_ITEM_TABLE,
    },
    {
        .obs_start = 124,
        .num_records = 14,
        .obs_features = 13,
        .type_onehot = 14,
        .type_code_scale = 16,
        .expansion = OSRS_ENTITY_BRANCH_TYPE_ONEHOT,
    },
};

static_assert(14 + 13 - 1 == 26);
static_assert(124 + 14 * 13 <= 530);

static constexpr OsrsEntityEncoderDescriptor OSRS_INFERNO_ENTITY_DESCRIPTOR = {
    .env_name = "osrs_inferno",
    .obs_size = 530,
    .branches = OSRS_INFERNO_ENTITY_BRANCHES,
    .num_branches = 2,
};

static void* osrs_inferno_entity_encoder_create_weights(void* self) {
    return osrs_entity_encoder_create_weights(
        self, &OSRS_INFERNO_ENTITY_DESCRIPTOR);
}

static void create_osrs_inferno_encoder(Encoder* encoder) {
    osrs_entity_encoder_configure(
        encoder,
        osrs_inferno_entity_encoder_create_weights,
        &OSRS_INFERNO_ENTITY_DESCRIPTOR);
}
