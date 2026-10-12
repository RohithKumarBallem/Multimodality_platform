#!/usr/bin/env bash
# Run the C++ test-suite under memory / UB / race checkers.
#
#   scripts/memcheck.sh [asan|ubsan|tsan|valgrind|all]    (default: all)
#
# A sanitizer whose runtime cannot execute a trivial program on this host is
# SKIPPED with a warning (some macOS toolchains ship broken ASan/TSan runtimes).
# Set STRICT=1 (CI does) to turn a skip into a failure.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-all}"
STRICT="${STRICT:-0}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
CXX_BIN="${CXX:-c++}"
status=0

works() {  # does -fsanitize=$1 run a trivial program here?
  local t; t="$(mktemp -d)"
  printf 'int main(){return 0;}\n' > "$t/t.cc"
  local ok=1
  if "$CXX_BIN" -fsanitize="$1" "$t/t.cc" -o "$t/t" >/dev/null 2>&1; then
    # subshell so the shell's "Illegal instruction" report is not printed
    if ( "$t/t" ) >/dev/null 2>&1; then ok=0; fi
  fi
  rm -rf "$t"
  return "$ok"
}

run_sanitizer() {
  local name="$1" flag="$2"
  if ! works "$flag"; then
    echo "SKIP  $name: -fsanitize=$flag does not run on this host"
    if [ "$STRICT" = 1 ]; then status=1; fi
    return 0
  fi
  local dir="$ROOT/build-$name"
  echo "==== $name (-fsanitize=$flag) ===="
  cmake -S "$ROOT" -B "$dir" -DMMWM_SANITIZE="$flag" -DMMWM_BUILD_BENCH=OFF \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
  cmake --build "$dir" -j "$JOBS" >/dev/null
  local leaks=1
  if [ "$(uname -s)" = Darwin ]; then leaks=0; fi  # LeakSanitizer is unavailable on macOS
  ASAN_OPTIONS="detect_leaks=$leaks:abort_on_error=0:halt_on_error=1" \
  UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" \
  TSAN_OPTIONS="halt_on_error=1" \
    ctest --test-dir "$dir" --output-on-failure || status=1
}

run_valgrind() {
  if ! command -v valgrind >/dev/null; then
    echo "SKIP  valgrind: not installed"
    if [ "$STRICT" = 1 ]; then status=1; fi
    return 0
  fi
  local dir="$ROOT/build-valgrind"
  echo "==== valgrind memcheck ===="
  cmake -S "$ROOT" -B "$dir" -DMMWM_BUILD_BENCH=OFF -DCMAKE_BUILD_TYPE=Debug >/dev/null
  cmake --build "$dir" -j "$JOBS" >/dev/null
  for t in test_runtime test_model; do
    valgrind --error-exitcode=1 --leak-check=full --errors-for-leak-kinds=definite \
             --quiet "$dir/$t" || status=1
  done
}

case "$MODE" in
  asan)     run_sanitizer asan address ;;
  ubsan)    run_sanitizer ubsan undefined ;;
  tsan)     run_sanitizer tsan thread ;;
  valgrind) run_valgrind ;;
  all)      run_sanitizer asan address; run_sanitizer ubsan undefined
            run_sanitizer tsan thread; run_valgrind ;;
  *) echo "usage: $0 [asan|ubsan|tsan|valgrind|all]" >&2; exit 2 ;;
esac

if [ "$status" = 0 ]; then echo "memcheck: OK"; else echo "memcheck: FAILED"; fi
exit "$status"
