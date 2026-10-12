#!/usr/bin/env bash
# Build-regression release gate. Fails (non-zero) unless every stage passes:
#
#   1. clean Release build, warnings-as-errors
#   2. C++ tests (ctest)
#   3. memory / UB / race checks   (scripts/memcheck.sh; STRICT=1 makes skips fatal)
#   4. Python tests
#   5. latency benchmark gate      (scripts/compare_bench.py)
#   6. package dist/mmwm-<version>-<git sha>-<os>-<arch>.tar.gz + gate report
#
#   scripts/regression_release.sh [--update-baseline]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BUILD="$ROOT/build-release"
DIST="$ROOT/dist"
BASELINE="$ROOT/bench/baseline.json"
VERSION="$(sed -n 's/^project(.*VERSION \([0-9.]*\).*/\1/p' CMakeLists.txt)"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
STAMP="${VERSION}-${SHA}-$(uname -s | tr A-Z a-z)-$(uname -m)"
UPDATE=0
if [ "${1:-}" = "--update-baseline" ]; then UPDATE=1; fi
PY="${PYTHON:-python3}"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

step "1/6 clean release build (-Werror)"
rm -rf "$BUILD"
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DMMWM_WERROR=ON >/dev/null
cmake --build "$BUILD" -j

step "2/6 C++ tests"
ctest --test-dir "$BUILD" --output-on-failure

step "3/6 memory / UB / race checks"
"$ROOT/scripts/memcheck.sh" all

step "4/6 Python tests"
MMWM_LIB="$(ls "$BUILD"/libmmwm.dylib "$BUILD"/libmmwm.so 2>/dev/null | head -1)" \
PYTHONPATH="$ROOT/python" "$PY" -m unittest discover -s python/tests

step "5/6 latency benchmark gate"
mkdir -p "$DIST"
"$BUILD/mmwm_bench_latency" --out "$DIST/bench.json" >/dev/null
if [ "$UPDATE" = 1 ]; then
  "$PY" scripts/compare_bench.py "$DIST/bench.json" --baseline "$BASELINE" --update-baseline
fi
"$PY" scripts/compare_bench.py "$DIST/bench.json" --baseline "$BASELINE" \
      --min-reduction 40 --report "$DIST/gate_report.json"

step "6/6 package"
PKG="$DIST/mmwm-$STAMP"
rm -rf "$PKG"; mkdir -p "$PKG/lib" "$PKG/bin"
cp "$BUILD"/libmmwm.* "$PKG/lib/"
cp "$BUILD/mmwm_bench_latency" "$PKG/bin/"
cp -R python configs "$PKG/"
cp "$DIST/bench.json" "$DIST/gate_report.json" "$PKG/"
tar -C "$DIST" -czf "$PKG.tar.gz" "mmwm-$STAMP"
echo "release artifact: $PKG.tar.gz"
