set -eu
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
s=$b/bot1-source/ocean/rek_g1/native5
m=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
mkdir "$b/viewer-bot1"
/usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_121 -Xcompiler=-fPIC '-DREK_EVAL_BACKEND="semantic_cuda"' -I"$s/.." -I"$s" -I"$m/include" -I/usr/local/cuda/include/cccl -c "$s/eval_worker.cpp" -o "$b/viewer-bot1/eval_worker.o" > "$b/viewer-bot1/compile.stdout.txt" 2> "$b/viewer-bot1/compile.stderr.txt"
printf 'viewer_consumer_compile=passed; deployed=false\n'
