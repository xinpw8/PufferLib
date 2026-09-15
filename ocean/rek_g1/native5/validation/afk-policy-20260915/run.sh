#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 ]] || exit 2
task_base=$(realpath "$1")
task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_output=$task_base/afk-policy-20260915
mkdir "$task_output"
task_native=$task_base/source-v3/ocean/rek_g1/native5
task_mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
unset REK_EVAL_ROUND_FEATURE
for task_version in 2 3;do
  task_build=$task_base/build-v$task_version
  task_dest=$task_output/build-v$task_version
  mkdir "$task_dest"
  exec 9>"$task_dest/build-commands.txt"
  BASH_XTRACEFD=9
  set -x
  for task_mode in neutral scripted;do
    task_input=$task_source/afk_policy_eval.cu
    if [[ "$task_mode" == scripted ]];then task_input=$task_native/fast_policy_eval.cu;fi
    /usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_121 -I"$task_native" -I"$task_base/source-v3/vendor" \
      "$task_input" "$task_build/fast_runtime.o" "$task_build/fast_assets.o" "$task_build/native_policy.o" "$task_build/cJSON.o" \
      -L"$task_mujoco" -Xlinker=-rpath -Xlinker="$task_mujoco" -l:libmujoco.so.3.7.0 -lcrypto -lcublas -lcurand \
      -o "$task_dest/eval-$task_mode" > "$task_dest/$task_mode.build.stdout.txt" 2> "$task_dest/$task_mode.build.stderr.txt"
  done
  sha256sum "$task_source/afk_policy_eval.cu" "$task_native/fast_policy_eval.cu" "$task_build/fast_runtime.o" "$task_build/fast_assets.o" "$task_build/native_policy.o" "$task_build/cJSON.o" "$task_dest/eval-neutral" "$task_dest/eval-scripted" > "$task_dest/build-hashes.txt"
  readelf -d "$task_dest/eval-neutral" > "$task_dest/elf-dependencies.txt"
  set +x
  if rg -qi '(libpython|libtorch)' "$task_dest/elf-dependencies.txt";then exit 2;fi
done
for task_version in 2 3;do
  if [[ "$task_version" == 2 ]];then
    task_checkpoint=$task_base/train-v2-33m/checkpoints/rek_native5/semantic-cuda-512-16/0000000033554432.bin
    task_sha=0fa325324083023d69f6e5b489792880e423b5b7c7bd06387eaf2c6007018fe2
  else
    task_checkpoint=$task_base/train-v3-33m/checkpoints/rek_native5/semantic-cuda-v3-512-16/0000000033554432.bin
    task_sha=e209edd0d1301170c5253f622f5f11aabccb57daf7fd4117983942bccb989fc6
  fi
  [[ "$(sha256sum "$task_checkpoint" | cut -d' ' -f1)" == "$task_sha" ]]
  for task_seconds in 20 300;do
    for task_mode in neutral scripted;do
      task_dest=$task_output/v$task_version-$task_mode-$task_seconds
      mkdir "$task_dest"
      node -e 'const fs=require("fs");const d=JSON.parse(fs.readFileSync(process.argv[1],"utf8"));d.round_seconds=Number(process.argv[3]);fs.writeFileSync(process.argv[2],JSON.stringify(d,null,2)+"\n",{flag:"wx"});' "$task_base/eval-run-v$task_version/semantic_cuda-worker.json" "$task_dest/worker.private.json" "$task_seconds"
      task_command=("$task_output/build-v$task_version/eval-$task_mode" "$task_dest/worker.private.json" "$task_checkpoint" "$task_sha" 128 4 10001 sampled bf16 "$task_dest/matches.private.jsonl")
      printf '%q ' "${task_command[@]}" > "$task_dest/command.txt"
      printf '\n' >> "$task_dest/command.txt"
      { date -u --iso-8601=seconds;hostname;id;nvidia-smi -L;sha256sum "$task_dest/worker.private.json" "$task_checkpoint" "$task_output/build-v$task_version/eval-$task_mode"; } > "$task_dest/provenance.txt"
      set +e
      /usr/bin/time -v -o "$task_dest/process-timing.txt" timeout --signal=TERM --kill-after=10s 180s "${task_command[@]}" > "$task_dest/summary.jsonl" 2> "$task_dest/stderr.txt"
      task_status=$?
      set -e
      printf '%s\n' "$task_status" > "$task_dest/exit-code.txt"
      printf 'case=v%s-%s-%s status=%s\n' "$task_version" "$task_mode" "$task_seconds" "$task_status"
      cat "$task_dest/summary.jsonl"
      [[ "$task_status" == 0 ]] || exit "$task_status"
    done
  done
done
