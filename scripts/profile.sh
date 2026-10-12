#!/usr/bin/env bash
# Profile the inference path.
#
#   scripts/profile.sh [bench args...]
#
# 1. Built-in scoped-timer breakdown (always; portable).
# 2. A sampling profile if a system profiler is available:
#    perf (Linux) or valgrind --tool=callgrind.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/profile-out"; mkdir -p "$OUT"

cmake -S "$ROOT" -B "$ROOT/build-prof" -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
cmake --build "$ROOT/build-prof" --target mmwm_bench_latency -j >/dev/null
BENCH="$ROOT/build-prof/mmwm_bench_latency"

echo "== built-in profiler =="
"$BENCH" --profile "$@" >/dev/null

if command -v perf >/dev/null; then
  echo "== perf =="
  perf record -g -o "$OUT/perf.data" "$BENCH" --iters 10 "$@" >/dev/null
  perf report -i "$OUT/perf.data" --stdio --no-children 2>/dev/null | head -40
elif command -v valgrind >/dev/null; then
  echo "== callgrind =="
  valgrind --tool=callgrind --callgrind-out-file="$OUT/callgrind.out" \
           "$BENCH" --iters 2 --batch 8 "$@" >/dev/null
  callgrind_annotate "$OUT/callgrind.out" | head -40
else
  echo "(no perf/valgrind found; on macOS use: xcrun xctrace record --template 'Time Profiler' --launch -- $BENCH)"
fi
