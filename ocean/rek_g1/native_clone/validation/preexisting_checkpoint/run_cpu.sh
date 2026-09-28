#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Daniel/codex-rek-puffysics-training-profile
out=/mnt/c/rekagent/work/rek-github-checkpoint-20260928-r1/preexisting-tests/cpu
test ! -e "$out"
mkdir "$out"
exec > "$out/stdout.txt" 2> "$out/stderr.txt"
trap 'code=$?; printf "%s\n" "$code" > "$out/exit-code.txt"' EXIT
src=ocean/rek_g1/native5
set -x
gcc -std=c11 -O2 -c vendor/cJSON.c -o "$out/cJSON.o"
gcc -std=c99 -O2 "$src/test_will_connect.c" -lm -o "$out/test-will-connect"
"$out/test-will-connect" > "$out/will-connect.txt"
gcc -std=c99 -O2 "$src/will_connect_regression_test.c" -lm -o "$out/test-will-connect-regression"
"$out/test-will-connect-regression" > "$out/will-connect-regression.txt"
g++ -std=c++17 -O2 "$src/will_connect_diagnostics_test.cpp" "$out/cJSON.o" -lm -o "$out/test-will-connect-diagnostics"
"$out/test-will-connect-diagnostics" > "$out/will-connect-diagnostics.txt"
g++ -std=c++17 -O2 -Wall -Wextra -Werror "$src/observable_prev_action_test.cpp" -o "$out/test-previous-action"
"$out/test-previous-action" > "$out/previous-action.json"
g++ -std=c++17 -O2 -DREK_LIVE_PROTOCOL_TEST -x c++ "$src/live_policy_worker.cu" -x none "$out/cJSON.o" -o "$out/live-protocol"
for schema in rek.native5.scaled_polar_xy.v1 rek.native5.observable_balance.v1 rek.native5.observable_balance_prev_action.v1; do
 REK_OBSERVATION_SCHEMA="$schema" node "$src/live_policy_worker.test.cjs" protocol "$out/live-protocol" > "$out/protocol-$schema.json"
done
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -Wno-misleading-indentation "$src/live_transfer/encode_observable_balance.cpp" "$out/cJSON.o" -lcrypto -o "$out/encoder"
# The encoder uses this file only as a hash identity. A tiny explicit fixture
# avoids depending on private XML assets; geometry samples come from the tests.
printf '<mujoco model="cpu-identity-fixture"/>\n' > "$out/identity-fixture.xml"
for schema in rek.native5.observable_balance.v1 rek.native5.observable_balance_prev_action.v1; do
 REK_OBSERVATION_SCHEMA="$schema" node "$src/live_transfer/observable_encoder_test.cjs" "$out/encoder" "$out/identity-fixture.xml" > "$out/encoder-$schema.json"
done
sha256sum "$out"/test-* "$out/live-protocol" "$out/encoder" > "$out/binary-hashes.sha256"
