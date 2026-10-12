#!/usr/bin/env python3
"""Gate a benchmark result.

Gates (all must pass):
  1. latency_reduction_pct >= --min-reduction        (optimized vs reference path)
  2. optimized/reference p50 ratio did not regress by more than
     --max-regression-pct versus the baseline file (skipped when the baseline
     was recorded with a different thread count: the ratio is only comparable
     on like-for-like hardware).

Exit status 0 = pass, 1 = gate failed, 2 = usage / input error.
"""
import argparse
import json
import sys
from pathlib import Path


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        sys.exit(2)


def ratio(r):
    return r["optimized"]["p50_ms"] / r["reference"]["p50_ms"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--min-reduction", type=float, default=40.0)
    ap.add_argument("--max-regression-pct", type=float, default=15.0)
    ap.add_argument("--update-baseline", action="store_true",
                    help="write RESULT to --baseline and exit 0")
    ap.add_argument("--report", help="write the gate report as JSON here")
    a = ap.parse_args()

    res = load(a.result)
    report = {"result": res, "gates": []}
    ok = True

    def gate(name, passed, detail):
        nonlocal ok
        ok = ok and passed
        report["gates"].append({"name": name, "passed": passed, "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    if a.update_baseline:
        if not a.baseline:
            print("error: --update-baseline needs --baseline", file=sys.stderr)
            return 2
        Path(a.baseline).write_text(json.dumps(res, indent=2) + "\n")
        print(f"baseline written to {a.baseline}")
        return 0

    red = res["latency_reduction_pct"]
    gate("latency-reduction", red >= a.min_reduction,
         f"{red:.1f}% (required >= {a.min_reduction:.1f}%)")

    if a.baseline and Path(a.baseline).exists():
        base = load(a.baseline)
        if base["config"]["threads"] != res["config"]["threads"]:
            print(f"[SKIP] regression: baseline used {base['config']['threads']} threads, "
                  f"this run {res['config']['threads']}")
            report["gates"].append({"name": "regression", "passed": True, "detail": "skipped"})
        else:
            b, r = ratio(base), ratio(res)
            change = 100.0 * (r / b - 1.0)
            gate("no-regression", change <= a.max_regression_pct,
                 f"optimized/reference p50 ratio {r:.4f} vs baseline {b:.4f} "
                 f"({change:+.1f}%, allowed +{a.max_regression_pct:.0f}%)")
    else:
        print("[SKIP] regression: no baseline")

    report["passed"] = ok
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
