"""Reproduce every artifact backing the paper's claims, in dependency order.

Usage:
    python reproduce_all.py [--skip-heavy]

Run with ``--skip-heavy`` to skip the two long pipelines (temporal sweep and
the full v2 tuning) and still produce the core ceiling + label-noise outputs.

Determinism: every pipeline uses seed 42 and single-threaded estimators, so
repeated runs are bit-identical for a fixed environment.  Set SEARCH_JOBS=-1
to parallelise CV folds without changing results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


STEPS: list[dict[str, str]] = [
    {"name": "run_ceiling.py",
     "desc": "ceiling analysis (Table 1 zoo, MI spectrum, PCA, learning "
             "curve, 21-pair McNemar)"},
    {"name": "run_label_noise_audit.py",
     "desc": "label-noise audit (rho in {0,0.05,...,0.20})"},
    {"name": "main_v2.py", "heavy": True,
     "desc": "leakage-free v2 tuning + validation-based threshold/calibration"},
    {"name": "run_temporal_ceiling.py", "heavy": True,
     "desc": "temporal sweep + fixed-cohort + lead-padded controls"},
    {"name": "main.py", "heavy": True,
     "desc": "legacy v1 pipeline (validation-based threshold, nested CV, "
             "pairwise tests)"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-heavy", action="store_true",
                        help="skip temporal sweep + full v2 + v1 tuning")
    args = parser.parse_args()

    started = time.time()
    failures: list[str] = []
    for step in STEPS:
        if args.skip_heavy and step.get("heavy"):
            print(f"[skip] {step['name']} ({step['desc']})\n")
            continue
        print(f"[run ] {step['name']} ({step['desc']})...")
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-u", step["name"]],
                              stdout=sys.stdout, stderr=sys.stderr)
        elapsed = time.time() - t0
        if proc.returncode == 0:
            print(f"[done] {step['name']} in {elapsed:.1f}s\n")
        else:
            print(f"[FAIL] {step['name']} (exit {proc.returncode}, "
                  f"{elapsed:.1f}s)", file=sys.stderr)
            failures.append(step["name"])

    if failures:
        print(f"\nReproduction incomplete; failed: {', '.join(failures)}",
              file=sys.stderr)
        sys.exit(1)

    total = time.time() - started
    print(f"\nReproduction complete in {total:.1f}s "
          f"({total / 60:.1f} min).")
    print("Artifacts: plots_ceiling/*, plots_temporal/*, plots_v2/*, plots/*")


if __name__ == "__main__":
    main()