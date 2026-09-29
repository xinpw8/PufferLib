#!/usr/bin/env bash
# Start the online server detached, pinned to the performance cores, with a disk
# cap on its log directory. Usage: start.sh CONFIG.json RUN_DIR
set -euo pipefail
test "$#" -eq 2 || { echo 'Usage: start.sh CONFIG.json RUN_DIR' >&2; exit 2; }
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config="$(realpath "$1")"; run="$2"
test ! -e "$run" || { echo "Fresh run directory required: $run" >&2; exit 1; }
mkdir -m 700 "$run"
port="$(node -e 'console.log(JSON.parse(require("fs").readFileSync(process.argv[1])).port)' "$config")"
if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then echo "Port $port already in use" >&2; exit 1; fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
setsid taskset -c "${REK_CPUS:-5-9,15-19}" node "$here/server.cjs" "$config" \
  > "$run/server.stdout.log" 2> "$run/server.stderr.log" < /dev/null &
pid=$!
start_ticks="$(awk '{print $22}' "/proc/$pid/stat")"
printf '{"pid":%s,"start_ticks":%s,"config":"%s","port":%s,"utc":"%s"}\n' "$pid" "$start_ticks" "$config" "$port" "$(date -u +%FT%TZ)" > "$run/server-process.json"
# Stops only this server's session if its run and log directories grow past 2 GB.
log_dir="$(node -e 'console.log(JSON.parse(require("fs").readFileSync(process.argv[1])).logDir)' "$config")"
setsid python3 "$here/disk_cap.py" --pid "$pid" --start-ticks "$start_ticks" --run "$log_dir" \
  --cap-bytes $((2*1024*1024*1024)) --disk-fraction 0.78 > "$run/disk-cap.log" 2>&1 < /dev/null &
for _ in $(seq 1 120); do
  if grep -q '"ready":true' "$run/server.stdout.log" 2>/dev/null; then cat "$run/server-process.json"; exit 0; fi
  kill -0 "$pid" 2>/dev/null || { echo 'Server exited during startup' >&2; cat "$run/server.stderr.log" >&2; exit 1; }
  sleep 0.5
done
echo 'Server did not report ready' >&2; exit 1
