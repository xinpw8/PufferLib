#!/usr/bin/env bash
# Build the pinned PufferLib 5.0 native trainer and link caller-owned modules.
set -euo pipefail

if (( $# < 2 )); then
    printf 'Usage: %s NEW_BUILD_DIRECTORY --compile-trainer-only\n' "$0" >&2
    printf '       %s NEW_BUILD_DIRECTORY --build-runtime\n' "$0" >&2
    printf '       %s NEW_BUILD_DIRECTORY RUNTIME_OBJECT [MODULE_OBJECT ...]\n' "$0" >&2
    exit 2
fi

task_source=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_build=$1
shift
task_compile_only=0
task_build_runtime=0
task_objects=()
if [[ "$1" == --compile-trainer-only ]]; then
    (( $# == 1 )) || { printf 'Unexpected arguments after --compile-trainer-only\n' >&2; exit 2; }
    task_compile_only=1
elif [[ "$1" == --build-runtime ]]; then
    (( $# == 1 )) || { printf 'Unexpected arguments after --build-runtime\n' >&2; exit 2; }
    task_build_runtime=1
else
    for task_object in "$@"; do
        [[ "$task_object" == *.o && -f "$task_object" ]] || {
            printf 'Expected an existing native object: %s\n' "$task_object" >&2
            exit 2
        }
        task_objects+=("$(realpath "$task_object")")
    done
fi

task_commit=773f923d80e73bdc255a2ba730c918b28e416aa1
task_equivalent_local_commit=84a89728fafd4034ae3c386a9e0665d62310780b
task_public_repository=https://github.com/xinpw8/PufferLib.git
task_repository=${PUFFER5_GIT_SOURCE:-/home/spark-advantage/pufferlib-5.0-wr64}
task_stage=${PUFFER5_SOURCE_STAGE:-/home/spark-advantage/rek-training/native5-rek-20260913-v1/pufferlib5}
task_raylib=${PUFFER5_RAYLIB:-/home/spark-advantage/pufferlib-5.0-wr64/raylib-5.5_linux_aarch64}
task_nccl=${PUFFER5_NCCL:-/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl}
task_cuda=${REK_NATIVE5_CUDA:-/usr/local/cuda}
task_nvcc=${NVCC:-$task_cuda/bin/nvcc}
task_arch=${REK_CUDA_ARCH:-sm_121}
task_mujoco=${REK_NATIVE5_MUJOCO:-/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco}
task_g1=$(cd "$task_source/.." && pwd)
task_root=$(cd "$task_source/../../.." && pwd)

if (( task_build_runtime )); then
    for task_module in runtime measurement motion_assets physics sonic_controller robot_state native_policy; do
        [[ -f "$task_source/$task_module.cu" ]] || {
            printf 'Native runtime module missing: %s.cu\n' "$task_module" >&2
            exit 2
        }
    done
fi
if (( ! task_compile_only )); then
    for task_required in "$task_mujoco/include/mujoco/mujoco.h" "$task_mujoco/libmujoco.so.3.7.0"; do
        [[ -f "$task_required" ]] || { printf 'Native MuJoCo dependency missing: %s\n' "$task_required" >&2; exit 2; }
    done
fi

[[ -x "$task_nvcc" ]] || { printf 'CUDA compiler missing: %s\n' "$task_nvcc" >&2; exit 2; }
for task_required in "$task_raylib/include/raylib.h" "$task_raylib/lib/libraylib.a" \
        "$task_nccl/include/nccl.h" "$task_nccl/lib/libnccl.so"; do
    [[ -f "$task_required" ]] || { printf 'Build dependency missing: %s\n' "$task_required" >&2; exit 2; }
done
if ! git -C "$task_repository" cat-file -e "$task_commit^{commit}" 2>/dev/null; then
    # Never fetch into an existing caller checkout. Use a private temporary
    # repository when its object database lacks the published source pin.
    task_repository=$(mktemp -d "${TMPDIR:-/tmp}/rek-native5-git.XXXXXX")
    git -C "$task_repository" init -q
    git -C "$task_repository" fetch --quiet --filter=blob:none --no-tags --depth=1 \
        "$task_public_repository" "$task_commit"
fi
[[ "$(git -C "$task_repository" rev-parse "$task_commit^{commit}")" == "$task_commit" ]]

if [[ ! -e "$task_stage" ]]; then
    mkdir -p "$task_stage"
    git -C "$task_repository" archive "$task_commit" src config/default.ini \
        | tar -xf - -C "$task_stage"
fi
# Compare every staged trainer source to its pinned Git object. Local changes
# in the source checkout never enter this build.
while IFS= read -r task_relative; do
    task_expected=$(git -C "$task_repository" rev-parse "$task_commit:$task_relative")
    [[ -f "$task_stage/$task_relative" ]] || {
        printf 'Pinned source missing: %s\n' "$task_relative" >&2
        exit 2
    }
    task_actual=$(git -C "$task_repository" hash-object "$task_stage/$task_relative")
    [[ "$task_actual" == "$task_expected" ]] || {
        printf 'Pinned source differs: %s\n' "$task_relative" >&2
        exit 2
    }
done < <(git -C "$task_repository" ls-tree -r --name-only "$task_commit" -- src config/default.ini)

# A fresh build directory preserves previous binaries and validation evidence.
mkdir "$task_build"
task_build=$(cd "$task_build" && pwd)
mkdir "$task_build/config"
cp "$task_stage/config/default.ini" "$task_build/config/default.ini"
cp "$task_source/native5.ini" "$task_build/config/rek_native5.ini"
mkdir "$task_build/trainer"
git -C "$task_repository" archive "$task_commit" src config/default.ini \
    | tar -xf - -C "$task_build/trainer"
git -C "$task_build/trainer" apply --check "$task_source/pufferlib5_action_mask.patch"
git -C "$task_build/trainer" apply "$task_source/pufferlib5_action_mask.patch"
task_runner="$task_build/trainer/src/pufferl.cu"

if (( task_build_runtime )); then
    task_module_flags=(-std=c++17 -O2 "-arch=$task_arch" --fmad=false
        --prec-div=true --prec-sqrt=true --ftz=false -Xcompiler=-fPIC
        -Xcompiler=-ffp-contract=off -I"$task_g1" -I"$task_source"
        -I"$task_mujoco/include" -I"$task_cuda/include/cccl")
    for task_module in runtime measurement motion_assets physics sonic_controller robot_state native_policy; do
        printf 'Compiling native module %s\n' "$task_module"
        if [[ "$task_module" == physics || "$task_module" == native_policy ]]; then
            # Preserve the existing Puffysics kernel's default FMA behavior.
            "$task_nvcc" -std=c++17 -O3 "-arch=$task_arch" -Xcompiler=-fPIC \
                -I"$task_g1" -I"$task_source" -I"$task_mujoco/include" \
                -I"$task_cuda/include/cccl" -c "$task_source/$task_module.cu" \
                -o "$task_build/$task_module.o"
        else
            "$task_nvcc" "${task_module_flags[@]}" -c "$task_source/$task_module.cu" \
                -o "$task_build/$task_module.o"
        fi
        task_objects+=("$task_build/$task_module.o")
    done
    for task_module in native_motion_routes g1_strike_catalog; do
        "${CC:-gcc}" -std=c11 -O2 -fPIC -ffp-contract=off -I"$task_g1" \
            -c "$task_g1/$task_module.c" -o "$task_build/$task_module.host.o"
        task_objects+=("$task_build/$task_module.host.o")
    done
    "${CC:-gcc}" -std=c11 -O2 -fPIC -c "$task_root/vendor/cJSON.c" -o "$task_build/cJSON.o"
    task_objects+=("$task_build/cJSON.o")
    for task_module in puffer_action_adapter g1_semantic_action_table \
            native_locomotion_command sonic_motion_composer_native \
            sonic_motion_composer_libm_candidate sonic_motion_entry_matcher_native \
            g1_combat_tick g1_fight_state g1_fall_state g1_hit_detector; do
        printf 'Compiling recovered device module %s\n' "$task_module"
        "$task_nvcc" "${task_module_flags[@]}" -dc \
            "-DREK_G1_CUDA_SOURCE=\"$task_module.c\"" "$task_g1/g1_cuda_device.cu" \
            -o "$task_build/$task_module.device.o"
        task_objects+=("$task_build/$task_module.device.o")
    done
    for task_module in g1_motion_cuda g1_semantic_scheduler_cuda g1_combat_cuda g1_native_combat_cuda; do
        printf 'Compiling recovered CUDA module %s\n' "$task_module"
        "$task_nvcc" "${task_module_flags[@]}" -dc "$task_g1/$task_module.cu" \
            -o "$task_build/$task_module.o"
        task_objects+=("$task_build/$task_module.o")
    done
fi

task_flags=(-std=c++17 -O2 "-arch=$task_arch" --threads 0
    -I"$task_build/trainer" -I"$task_build/trainer/src" -I"$task_source"
    -I"$task_raylib/include" -I"$task_cuda/include" -I"$task_cuda/include/cccl"
    -I"$task_nccl/include" -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing
    --diag-suppress=2361 -DPLATFORM_DESKTOP -DPUFFERLIB_BUILD_MAIN
    -DENV_NAME=rek_native5 '-DPUFFER_ENV_NAME="rek_native5"'
    "-DENV_HEADER=\"$task_source/puffer_env.cu\"")

printf 'Compiling PufferLib 5.0 %s for %s\n' "$task_commit" "$task_arch"
"$task_nvcc" "${task_flags[@]}" -c "$task_runner" \
    -o "$task_build/pufferl.o"

{
    printf 'pufferlib_commit=%s\narchitecture=%s\n' "$task_commit" "$task_arch"
    printf 'public_repository=%s\n' "$task_public_repository"
    printf 'equivalent_local_src_and_default_ini_commit=%s\n' "$task_equivalent_local_commit"
    printf 'trainer_source=%s\n' "$task_runner"
    if (( task_compile_only )); then
        printf 'validation=compile_only\n'
    else
        printf 'validation=compile_and_link_only\n'
    fi
    printf 'runtime_validation=not_run\n'
    sha256sum "$task_stage/src/pufferl.cu" "$task_runner" "$task_source/pufferlib5_action_mask.patch" \
        "$task_stage/src/algo.cu" \
        "$task_source/puffer_env.cu" "$task_source/runtime_api.h" \
        "$task_source/native5.ini" "$task_build/pufferl.o"
} > "$task_build/build-source-manifest.txt"

if (( task_compile_only )); then
    printf 'Compiled trainer object: %s/pufferl.o\n' "$task_build"
    printf 'The runtime remains unresolved; no executable or GPU test was produced.\n'
    exit 0
fi

"$task_nvcc" "-arch=$task_arch" -Xcompiler=-fopenmp \
    "$task_build/pufferl.o" "${task_objects[@]}" "$task_raylib/lib/libraylib.a" \
    -L"$task_cuda/lib64" -L"$task_nccl/lib" -L"$task_mujoco" \
    -Xlinker=-rpath -Xlinker="$task_nccl/lib" \
    -Xlinker=-rpath -Xlinker="$task_cuda/lib64" \
    -Xlinker=-rpath -Xlinker="$task_mujoco" \
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
    -l:libmujoco.so.3.7.0 -lcrypto -lGL -lm -lpthread -lomp5 \
    -o "$task_build/puffer-rek-native5"

readelf -d "$task_build/puffer-rek-native5" > "$task_build/elf-dependencies.txt"
if rg -qi '(libpython|libtorch)' "$task_build/elf-dependencies.txt"; then
    printf 'Unexpected interpreter/framework link dependency; inspect %s\n' \
        "$task_build/elf-dependencies.txt" >&2
    exit 1
fi
sha256sum "${task_objects[@]}" "$task_build/puffer-rek-native5" \
    >> "$task_build/build-source-manifest.txt"
printf 'link_validation=passed\n' >> "$task_build/build-source-manifest.txt"
printf 'Built native executable: %s/puffer-rek-native5\n' "$task_build"
printf 'Run from %s so the native config loader finds config/.\n' "$task_build"
printf 'Compilation and linking do not validate environment behavior or training.\n'
