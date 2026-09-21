#!/usr/bin/env bash
set -euo pipefail
if (( $# != 4 )); then
    echo "usage: test_critic_fixed_scale.sh BINARY CHECKPOINT SHA NEW_TEST_DIRECTORY" >&2
    exit 2
fi
binary=$(realpath -- "$1")
checkpoint=$(realpath -- "$2")
digest=$3
output=$(realpath -m -- "$4")
test ! -e "$output"
mkdir -p -- "$output"
"$binary" --cpu-self-test
for scale in 0 0.01 1; do
    "$binary" --fixed-scale "$checkpoint" "$digest" "$scale" "$output/scale-$scale.bin"
    # Independent byte checks for the pinned registered value row [65536,65792).
    cmp --bytes=262144 -- "$checkpoint" "$output/scale-$scale.bin"
    cmp --ignore-initial=263168 -- "$checkpoint" "$output/scale-$scale.bin"
done
cmp -- "$checkpoint" "$output/scale-1.bin"
expect_rejected() {
    local name=$1 sha=$2 scale=$3 destination=$4 status=0
    if "$binary" --fixed-scale "$checkpoint" "$sha" "$scale" "$destination" > "$output/$name.stdout" 2> "$output/$name.stderr"; then
        echo "unexpected success: $name" >&2
        exit 1
    else
        status=$?
    fi
    test "$status" -eq 1
    printf '{"rejection":"%s","exit":%d}\n' "$name" "$status"
}
for scale in -0.01 nan inf -inf 1x 1e999; do
    expect_rejected "invalid-scale-$scale" "$digest" "$scale" "$output/rejected.bin"
    test ! -e "$output/rejected.bin"
done
expect_rejected invalid-sha-format invalid 0.01 "$output/rejected.bin"
test ! -e "$output/rejected.bin"
expect_rejected wrong-sha 0000000000000000000000000000000000000000000000000000000000000000 0.01 "$output/rejected.bin"
test ! -e "$output/rejected.bin"
expect_rejected existing-output "$digest" 0.01 "$output/scale-1.bin"
cmp -- "$checkpoint" "$output/scale-1.bin"
sha256sum -- "$checkpoint" "$output/scale-0.bin" "$output/scale-0.01.bin" "$output/scale-1.bin"
printf '{"fixed_scale_cli_test":"passed","native_cpu_only":true,"outside_value_bytes_unchanged":true,"scale_one_byte_identity":true,"rejections":9}\n'
