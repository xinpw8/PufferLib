#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir=${1:?Usage: test_g1_combat_cuda.sh BUILD_DIRECTORY}
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
bash "$source_dir/build_g1_combat_cuda.sh" "$build_dir"
build_dir=$(cd "$build_dir" && pwd)
host_objects=()
for module in g1_combat_tick g1_fight_state g1_fall_state g1_hit_detector; do
    gcc -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror -pedantic \
        -I"$source_dir" -c "$source_dir/$module.c" \
        -o "$build_dir/$module.host.o"
    host_objects+=("$build_dir/$module.host.o")
done
gcc -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror -pedantic \
    -Dmain=rek_g1_combat_fixture_main \
    -Drek_g1_combat_arena_init=rek_g1_test_cuda_init \
    -Drek_g1_combat_arena_substep=rek_g1_test_cuda_substep \
    -Drek_g1_combat_arena_apply_spawn_reset=rek_g1_test_cuda_reset \
    -c "$source_dir/test_g1_combat_tick.c" -o "$build_dir/combat.fixture.o"
gcc -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror -pedantic \
    -Dmain=rek_g1_fall_fixture_main \
    -Drek_g1_fall_state_step=rek_g1_test_cuda_fall_step \
    -c "$source_dir/test_g1_fall_state.c" -o "$build_dir/fall.fixture.o"
gcc -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror -pedantic \
    -Dmain=rek_g1_hit_fixture_main \
    -Drek_g1_hit_detector_process=rek_g1_test_cuda_hit_process \
    -c "$source_dir/test_g1_hit_detector.c" -o "$build_dir/hit.fixture.o"
"$nvcc" -std=c++20 -O2 -Xcompiler=-ffp-contract=off -I"$source_dir" \
    "$source_dir/test_g1_combat_cuda.cpp" "${host_objects[@]}" \
    "$build_dir/combat.fixture.o" "$build_dir/fall.fixture.o" \
    "$build_dir/hit.fixture.o" -L"$build_dir" -lrek_g1_combat_cuda \
    -Xlinker "-rpath=$build_dir" -o "$build_dir/test_g1_combat_cuda"
"$build_dir/test_g1_combat_cuda"
