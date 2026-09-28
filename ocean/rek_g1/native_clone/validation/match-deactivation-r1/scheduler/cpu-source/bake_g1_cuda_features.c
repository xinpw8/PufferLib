/* Offline asset bake. The original registry sampler owns all kinematic math. */
#include "g1_mujoco_feature_registry.c"

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL CLIP_DOF_F32LE OUTPUT_F32LE\n", argv[0]);
        return 64;
    }
    char error[1024] = {0};
    mjModel* model = mj_loadXML(argv[1], NULL, error, sizeof(error));
    if (model == NULL) { fprintf(stderr, "%s\n", error); return 1; }
    GearSonicNativeDuelVector duel = {0};
    duel.model = model;
    RekG1MujocoFeatureRegistry registry = {0};
    registry.duel = &duel;
    registry.scratch = mj_makeData(model);
    registry.root_body_id = mj_name2id(model, mjOBJ_BODY, PLAYER_ROOT_BODY);
    registry.left_ankle_roll_body_id = mj_name2id(
        model, mjOBJ_BODY, PLAYER_LEFT_ANKLE_ROLL_BODY);
    registry.right_ankle_roll_body_id = mj_name2id(
        model, mjOBJ_BODY, PLAYER_RIGHT_ANKLE_ROLL_BODY);
    if (!registry.scratch || registry.root_body_id <= 0
            || registry.left_ankle_roll_body_id <= 0
            || registry.right_ankle_roll_body_id <= 0) return 1;
    for (size_t joint = 0; joint < GEAR_SONIC_ACTION_DIM; ++joint) {
        const int id = mj_name2id(model, mjOBJ_JOINT, PLAYER_JOINT_NAMES[joint]);
        if (id < 0 || model->jnt_type[id] != mjJNT_HINGE) return 1;
        duel.fighters[GEAR_SONIC_DUEL_PLAYER].qpos_addresses[joint] = model->jnt_qposadr[id];
    }
    registry.initialized = 1;
    FILE* source = fopen(argv[2], "rb");
    if (!source || fseek(source, 0, SEEK_END) != 0) return 1;
    const long bytes = ftell(source);
    const size_t row_bytes = GEAR_SONIC_ACTION_DIM * sizeof(float);
    if (bytes <= 0 || (size_t)bytes % row_bytes != 0
            || fseek(source, 0, SEEK_SET) != 0) return 1;
    FILE* destination = fopen(argv[3], "wbx");
    if (!destination) { perror(argv[3]); return 1; }
    const size_t frames = (size_t)bytes / row_bytes;
    for (size_t frame = 0; frame < frames; ++frame) {
        float pose[GEAR_SONIC_ACTION_DIM];
        float feature[SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH];
        if (fread(pose, sizeof(float), GEAR_SONIC_ACTION_DIM, source)
                    != GEAR_SONIC_ACTION_DIM
                || !rek_g1_mujoco_feature_registry_sample(&registry, pose, feature)
                || fwrite(feature, sizeof(float), 6, destination) != 6) return 1;
    }
    if (fclose(source) || fclose(destination)) return 1;
    mj_deleteData(registry.scratch);
    mj_deleteModel(model);
    printf("{\"frames\":%zu,\"feature_width\":6,\"mujoco_version\":%d}\n",
        frames, mj_version());
    return 0;
}
