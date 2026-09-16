#!/usr/bin/env python
"""
Benchmarks DarkLens against a list of URLs.

What this script actually measures, live, no invented numbers: wall-
clock latency, peak RSS memory, and CPU time for a real scan of each
URL you give it. That's it. It does NOT print a precision/recall/F1/mAP
number, because computing those requires a labeled ground-truth dataset
(URLs with human-verified "these are the real dark patterns present")
that does not exist in this repo — inventing one to make this script's
output look more complete would violate the project's own "never invent
metrics" rule. See the [USER TASK] block below for what's needed to
fill that gap in for real.

Usage:
    python scripts/benchmark.py https://example.com https://another-site.com
    python scripts/benchmark.py --urls-file urls.txt --output docs/benchmark_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import tracemalloc
from pathlib import Path

logger = logging.getLogger("benchmark")

# ============================================================
# [USER TASK]
#
# Measure
# Latency
# Precision
# Recall
# F1
# mAP
# CPU Usage
# Memory Usage
# False Positives
# False Negatives
# Compare with baseline
# Replace placeholders
# ------------------------------------------------------------
# What this script covers automatically, per URL, on every run:
#   Latency          -> wall-clock scan time, printed and saved to JSON
#   CPU Usage        -> process CPU time via os.times(), before/after scan
#   Memory Usage     -> peak Python-allocated memory via tracemalloc
#
# What still requires real work before it means anything:
#   Precision / Recall / F1 / False Positives / False Negatives
#     -> Requires a labeled benchmark set: a list of URLs (ideally
#        real sites, or the fixture pages used in integration tests)
#        each paired with a human-verified list of "these dark patterns
#        are actually present, at these locations". Build this as
#        `data/benchmark/ground_truth.json` — schema: {url: [{category,
#        title, dom_selector_or_region}, ...]}. Then a finding is a true
#        positive if it matches an entry (by category + rough location),
#        false positive if it doesn't match anything in ground truth,
#        and any ground-truth entry with no matching finding is a false
#        negative. This script does not attempt that matching logic
#        itself — write it once you have real ground truth to match
#        against, since the matching tolerance (e.g. "same DOM subtree"
#        vs "exact selector") is a judgment call that shouldn't be
#        guessed at without real data to validate it against.
#   mAP
#     -> Comes for free from `scripts/train_cv_detector.py`'s own
#        `model.val()` call once the CV model is trained — that's the
#        standard object-detection metric, not something this
#        end-to-end script should reimplement.
#   Compare with baseline
#     -> Once precision/recall/F1 exist for this system, the meaningful
#        baseline to compare against is "rule engine only, no NLP/CV" —
#        toggle those detectors off (unset the model paths in config)
#        and rerun the same ground-truth comparison. Report the delta,
#        not just DarkLens's own numbers in isolation.
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="*", help="URLs to scan")
    parser.add_argument("--urls-file", type=Path, help="File with one URL per line")
    parser.add_argument("--output", type=Path, default=Path("docs/benchmark_results.json"))
    return parser


async def _benchmark_one(url: str) -> dict:
    # Imported here, not at module top level, so `--help` doesn't need
    # the full app importable if run in a minimal environment.
    import os

    from darklens.container import build_scan_use_case

    use_case = build_scan_use_case()

    tracemalloc.start()
    cpu_before = os.times()
    wall_start = time.monotonic()

    result = await use_case.execute(url)

    wall_ms = (time.monotonic() - wall_start) * 1000
    cpu_after = os.times()
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cpu_seconds = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)

    return {
        "url": url,
        "latency_ms": round(wall_ms, 1),
        "cpu_seconds": round(cpu_seconds, 3),
        "peak_memory_mb": round(peak_memory_bytes / (1024 * 1024), 2),
        "findings_count": len(result.findings),
        "trust_score": result.trust_score,
        "risk_score": result.risk_score,
        # Ground-truth-dependent metrics: deliberately absent, not zero
        # or None-as-placeholder — see [USER TASK] above for why they
        # can't be computed yet, and null-ing them out would look like
        # a measured "we checked and found nothing to report".
    }


async def main_async(urls: list[str]) -> list[dict]:
    results = []
    for url in urls:
        logger.info("Benchmarking %s", url)
        results.append(await _benchmark_one(url))
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()

    urls = list(args.urls)
    if args.urls_file:
        urls.extend(
            line.strip() for line in args.urls_file.read_text().splitlines() if line.strip()
        )
    if not urls:
        raise SystemExit("Provide at least one URL, or --urls-file.")

    results = asyncio.run(main_async(urls))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))

    for r in results:
        print(
            f"{r['url']}: {r['latency_ms']}ms, {r['cpu_seconds']}s CPU, "
            f"{r['peak_memory_mb']}MB peak, {r['findings_count']} finding(s)"
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
