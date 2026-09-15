#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && ( "$1" == build || "$1" == train ) ]] || exit 2
task_mode=$1
task_stage=$(realpath "$2")
task_source=$task_stage/source-v3/ocean/rek_g1/native5
task_build=$task_stage/build-v3
task_cuda=/usr/local/cuda
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
task_nccl=/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl
task_raylib=/home/spark-advantage/pufferlib-5.0-wr64/raylib-5.5_linux_aarch64

if [[ "$task_mode" == build ]];then
  [[ ! -e "$task_build" && ! -e "$task_stage/eval-build-v3" ]]
  cp -a "$task_stage/build-v2" "$task_build"
  exec 9>"$task_build/commands.txt"
  BASH_XTRACEFD=9
  set -x
  "$task_cuda/bin/nvcc" -std=c++17 -O3 -arch=sm_121 -Xcompiler=-fPIC \
    -I"$task_source" -I"$task_source/.." -I"$task_stage/source-v3/vendor" \
    -I"$task_cuda/include/cccl" -c "$task_source/fast_runtime.cu" -o "$task_build/fast_runtime.o"
  "$task_cuda/bin/nvcc" -arch=sm_121 -Xcompiler=-fopenmp "$task_build/pufferl.o" \
    "$task_build/fast_runtime.o" "$task_build/fast_assets.o" "$task_build/native_policy.o" "$task_build/cJSON.o" \
    "$task_raylib/lib/libraylib.a" -L"$task_cuda/lib64" -L"$task_nccl/lib" -L"$task_mujoco" \
    -Xlinker=-rpath -Xlinker="$task_cuda/lib64" -Xlinker=-rpath -Xlinker="$task_nccl/lib" \
    -Xlinker=-rpath -Xlinker="$task_mujoco" -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
    -l:libmujoco.so.3.7.0 -lcrypto -lGL -lm -lpthread -lomp5 -o "$task_build/puffer-rek-native5"
  printf '%s\n' "$task_build/fast_runtime.o" "$task_build/fast_assets.o" "$task_build/native_policy.o" "$task_build/cJSON.o" > "$task_build/runtime-objects.txt"
  readelf -d "$task_build/puffer-rek-native5" > "$task_build/elf-dependencies.txt"
  if rg -qi '(libpython|libtorch)' "$task_build/elf-dependencies.txt";then exit 2;fi
  for task_object in pufferl.o fast_assets.o native_policy.o cJSON.o;do
    cmp "$task_stage/build-v2/$task_object" "$task_build/$task_object"
  done
  cmp "$task_stage/build-v2/config/default.ini" "$task_build/config/default.ini"
  cmp "$task_stage/build-v2/config/rek_native5.ini" "$task_build/config/rek_native5.ini"
  { printf 'backend=semantic_cuda\nversion=3\ncontrol_hz=50\ncpu_physics=0\npython_runtime=0\n';
    sha256sum "$task_source/fast_runtime.cu" "$task_source/fast_assets.cpp" "$task_source/fast_assets.h" \
      "$task_build/puffer-rek-native5" "$task_build"/*.o; } > "$task_build/fast-build.txt"
  { printf 'incremental_base=%s\nrecompiled=fast_runtime.o\n' "$task_stage/build-v2";
    sha256sum "$task_source/fast_runtime.cu" "$task_source/eval_worker.cpp" "$task_source/eval_renderer.h"; } > "$task_build/build-source-manifest.txt"
  REK_EVAL_RUNTIME=semantic_cuda bash "$task_source/build_eval.sh" "$task_build" "$task_stage/eval-build-v3"
  printf 'worker=%s/eval-build-v3/rek-eval-worker\n' "$task_stage"
else
  task_output=$task_stage/train-v3-33m
  mkdir "$task_output"
  task_assets=/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact
  task_command=(env REK_PHYSICS_BACKEND=semantic_cuda REK_ALLOW_CPU_EVALUATION=0
    "$task_build/puffer-rek-native5" train --headless --vec.total_agents=512
    --train.horizon=16 --train.minibatch_size=8192 --train.total_timesteps=33554432
    --sweep.metric=perf/train --sweep.downsample=64 --base.run_id=semantic-cuda-v3-512-16
    --base.checkpoint_interval=64 --base.log_dir="$task_output/logs"
    --base.checkpoint_dir="$task_output/checkpoints" --env.round_seconds=20 --env.seed=73
    --env.opponent_checkpoint=None --env.model_path="$task_assets/model.two_fighter_arena.xml"
    --env.physics_export_path=/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json
    --env.assets_path="$task_assets" --env.motion_features_path=/home/spark-advantage/rek-training/gpu-runtime-20260910/motion-foot-features)
  printf '%q ' "${task_command[@]}" > "$task_output/command.txt"
  printf '\n' >> "$task_output/command.txt"
  { date -u --iso-8601=seconds;hostname;id;uname -m;nvidia-smi -L;
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv;
    sha256sum "$task_build/puffer-rek-native5" "$task_assets/model.two_fighter_arena.xml"; } > "$task_output/provenance.txt"
  cp "$task_build/config/default.ini" "$task_build/config/rek_native5.ini" "$task_build/fast-build.txt" "$task_build/elf-dependencies.txt" "$task_output/"
  cd "$task_build"
  ulimit -c 0
  set +e
  /usr/bin/time -v -o "$task_output/process-timing.txt" timeout --signal=TERM --kill-after=10s 300s \
    "${task_command[@]}" > "$task_output/stdout.txt" 2> "$task_output/stderr.txt"
  task_status=$?
  set -e
  printf '%s\n' "$task_status" > "$task_output/exit-code.txt"
  sed -n 's/^native5_round_summary=//p' "$task_output/stdout.txt" > "$task_output/round-summary.json"
  if [[ -d "$task_output/checkpoints" ]];then
    find "$task_output/checkpoints" -type f -name '*.bin' -exec sha256sum {} + > "$task_output/checkpoint-hashes.txt"
  fi
  tail -20 "$task_output/stdout.txt"
  cat "$task_output/stderr.txt"
  printf 'training_exit_code=%s\noutput=%s\n' "$task_status" "$task_output"
  exit "$task_status"
fi
