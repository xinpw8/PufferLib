set -eu
b=/home/spark-advantage/rek-training/policy-quality-20260916-r1
s=$b/bot1-source/ocean/rek_g1/native5
mkdir "$b/eval-bot1"
mkdir "$b/fast-eval-bot1"
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_121 -Xcompiler=-Wformat=2 -I"$s" -c "$s/diverse_policy_eval.cu" -o "$b/eval-bot1/evaluator.o" > "$b/eval-bot1/compile.stdout.txt" 2> "$b/eval-bot1/compile.stderr.txt"
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_121 -Xcompiler=-Wformat=2 -I"$s" -c "$s/fast_policy_eval.cu" -o "$b/fast-eval-bot1/evaluator.o" > "$b/fast-eval-bot1/compile.stdout.txt" 2> "$b/fast-eval-bot1/compile.stderr.txt"
printf 'both_evaluator_compiles=passed\n'
