# Recorded invocation on spark-4ae3, using the native feature-enabled v2 build.
stage=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914
export REK_MUJOCO_KERNEL_CATALOG="$stage/kernel-catalog-v3.json"
export REK_MUJOCO_CONDITIONAL_PTX="$stage/build-native-v2/mujoco-conditional.ptx"
export REK_MUJOCO_CONDITIONAL_SHA256=2300284dfc4f4560234ef6ffe8a3fab8c918a1abc8ffcdcf270e180042837b21
bash "$stage/source/ocean/rek_g1/native5/physics_mujoco_gpu_probe.sh" \
  "$stage/adapter-wrapped-probe-r1" "$stage/build-native-v2" \
  /home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact/model.two_fighter_arena.xml \
  /home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
# The harness requires a new output directory. Choose a new suffix when rerunning.
