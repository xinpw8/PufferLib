#!/bin/bash
# Build and train rek_fight on the DGX Spark.
set -euo pipefail
HOST="${SPARK_HOST:-spark}"
REMOTE_DIR="${REK_REMOTE_DIR:-~/PufferLib}"
MJCF_REL="ocean/rek/evidence/evidence_out/t800_t800_factory_arena.diagnostic.xml"

ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" true

scp -q \
    ocean/rek_fight/rek_fight.h \
    ocean/rek_fight/binding.c \
    ocean/rek_fight/test_rek_fight.c \
    ocean/rek_fight/README.md \
    "$HOST:$REMOTE_DIR/ocean/rek_fight/"
scp -q config/rek_fight.ini "$HOST:$REMOTE_DIR/config/rek_fight.ini"

ssh "$HOST" "bash -lc $(printf '%q' "
set -euo pipefail
cd $REMOTE_DIR
export MUJOCO_HOME=\${MUJOCO_HOME:-\$HOME/mujoco-3.7.0}
export MUJOCO_LIB=\${MUJOCO_LIB:-\$MUJOCO_HOME/lib/libmujoco.so.3.7.0}
export REK_MJCF_PATH=\$PWD/$MJCF_REL
cc -std=c11 -O2 -I\"\$MUJOCO_HOME/include\" \
    ocean/rek_fight/test_rek_fight.c -o /tmp/test_rek_fight \
    \"\$MUJOCO_LIB\" -Wl,-rpath,\"\$(dirname \"\$MUJOCO_LIB\")\" -lm
/tmp/test_rek_fight
./build.sh rek_fight
python -m pufferlib.pufferl train rek_fight
")"
