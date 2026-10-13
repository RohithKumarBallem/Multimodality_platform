#!/usr/bin/env bash
# Build-regression release gate. Fails (non-zero) unless every stage passes:
#
#   1. clean Release build with CMake, warnings-as-errors
#   2. C++ tests (ctest)
#   3. Bazel build + tests, warnings-as-errors   (BAZEL=path to bazel/bazelisk)
#   4. memory / UB / race checks   (scripts/memcheck.sh)
#   5. Python tests
#   6. latency benchmark gate      (scripts/compare_bench.py)
#   7. package dist/mmwm-<version>-<git sha>-<os>-<arch>.tar.gz + gate report
#   8. (--tag) annotated git tag v<version> on a clean tree, only after 1-7 pass
#
# STRICT=1 turns "tool unavailable on this host" skips (Bazel, sanitizers,
# valgrind) into failures. CI sets it.
#
#   scripts/regression_release.sh [--update-baseline] [--tag]
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
TAG=0
for arg in "$@"; do
  case "$arg" in
    --update-baseline) UPDATE=1 ;;
    --tag) TAG=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
PY="${PYTHON:-python3}"
STRICT="${STRICT:-0}"
BAZEL="${BAZEL:-$(command -v bazel || command -v bazelisk || true)}"

if [ "$TAG" = 1 ]; then  # fail fast: do not spend minutes building first
  git rev-parse --git-dir >/dev/null 2>&1 || { echo "--tag needs a git checkout" >&2; exit 2; }
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "--tag needs a clean working tree" >&2; exit 2
  fi
  if git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null; then
    echo "tag v$VERSION already exists" >&2; exit 2
  fi
fi

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

step "1/7 clean release build (-Werror)"
rm -rf "$BUILD"
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DMMWM_WERROR=ON >/dev/null
cmake --build "$BUILD" -j

step "2/7 C++ tests"
ctest --test-dir "$BUILD" --output-on-failure

step "3/7 Bazel build + tests (-Werror)"
if [ -n "$BAZEL" ]; then
  "$BAZEL" test --config=opt --config=ci //...
else
  echo "SKIP  bazel/bazelisk not found"
  if [ "$STRICT" = 1 ]; then exit 1; fi
fi

step "4/7 memory / UB / race checks"
"$ROOT/scripts/memcheck.sh" all

step "5/7 Python tests"
MMWM_LIB="$(ls "$BUILD"/libmmwm.dylib "$BUILD"/libmmwm.so 2>/dev/null | head -1)" \
PYTHONPATH="$ROOT/python" "$PY" -m unittest discover -s python/tests

step "6/7 latency benchmark gate"
mkdir -p "$DIST"
"$BUILD/mmwm_bench_latency" --out "$DIST/bench.json" >/dev/null
if [ "$UPDATE" = 1 ]; then
  "$PY" scripts/compare_bench.py "$DIST/bench.json" --baseline "$BASELINE" --update-baseline
fi
"$PY" scripts/compare_bench.py "$DIST/bench.json" --baseline "$BASELINE" \
      --min-reduction 40 --min-reduction-vs-v1 10 --report "$DIST/gate_report.json"

step "7/7 package"
PKG="$DIST/mmwm-$STAMP"
rm -rf "$PKG"; mkdir -p "$PKG/lib" "$PKG/bin"
cp "$BUILD"/libmmwm.* "$PKG/lib/"
cp "$BUILD/mmwm_bench_latency" "$PKG/bin/"
cp -R python configs "$PKG/"
cp "$DIST/bench.json" "$DIST/gate_report.json" "$PKG/"
tar -C "$DIST" -czf "$PKG.tar.gz" "mmwm-$STAMP"
echo "release artifact: $PKG.tar.gz"

if [ "$TAG" = 1 ]; then
  step "tag v$VERSION"
  git tag -a "v$VERSION" -m "mmwm $VERSION

$("$PY" - "$DIST/bench.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"latency: {r['reference']['p50_ms']:.1f} ms reference -> "
      f"{r['optimized']['p50_ms']:.1f} ms optimized "
      f"({r['latency_reduction_pct']:.1f}% lower, {r['config']['threads']} threads)")
PYEOF
)
gates: build -Werror, ctest, bazel, memcheck, python tests, latency gate"
  echo "created tag v$VERSION (push with: git push origin v$VERSION)"
fi
