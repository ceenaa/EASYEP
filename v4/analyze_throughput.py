#!/usr/bin/env python3
"""Recover decode throughput from an EASY-EP V4 run.

Runs produced before the pipeline recorded per-forward timings only have, per
row, ``prompt_tokens``, ``completion_tokens`` and a ``seconds`` that spans the
whole generate() call. ``completion_tokens / seconds`` is therefore not a decode
rate: it folds in a prefill whose cost grows with prompt length. This estimates
the two separately by least squares over the rows,

    seconds ~= a * prompt_tokens + b * completion_tokens + k

so 1/b is the decode rate and 1/a an average prefill rate. Rows that already
carry the measured ``decode`` block are reported directly and never fitted.

    python analyze_throughput.py --run /path/to/run_ID
    python analyze_throughput.py --jsonl questions/answers_full.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    out = [0.0] * n
    for row in reversed(range(n)):
        total = aug[row][n] - sum(aug[row][c] * out[c] for c in range(row + 1, n))
        out[row] = total / aug[row][row]
    return out


def correlation(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def fit(rows: list[dict]) -> dict:
    """Least-squares split of wall time into prefill and decode components."""
    prompts = [float(r.get("prompt_tokens") or 0) for r in rows]
    completions = [float(r.get("completion_tokens") or 0) for r in rows]
    seconds = [float(r.get("seconds") or 0) for r in rows]
    n = len(rows)
    if n < 4:
        return {"ok": False, "why": f"only {n} rows; need at least 4 to fit 3 terms"}

    columns = [prompts, completions, [1.0] * n]
    matrix = [[sum(a * b for a, b in zip(ci, cj)) for cj in columns] for ci in columns]
    rhs = [sum(c * s for c, s in zip(ci, seconds)) for ci in columns]
    coeffs = solve(matrix, rhs)
    if coeffs is None:
        return {"ok": False, "why": "design matrix is singular (no variation in inputs)"}

    a, b, k = coeffs
    predicted = [a * p + b * c + k for p, c in zip(prompts, completions)]
    mean = sum(seconds) / n
    ss_res = sum((s - p) ** 2 for s, p in zip(seconds, predicted))
    ss_tot = sum((s - mean) ** 2 for s in seconds)
    result = {
        "ok": True,
        "rows": n,
        "r_squared": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
        "seconds_per_decoded_token": round(b, 6),
        "decode_tokens_per_second": round(1 / b, 2) if b > 0 else None,
        "seconds_per_prompt_token": round(a, 6),
        "prefill_tokens_per_second": round(1 / a, 1) if a > 0 else None,
        "fixed_overhead_seconds": round(k, 3),
        "naive_completion_over_wallclock_tps": round(
            sum(completions) / sum(seconds), 3) if sum(seconds) > 0 else None,
        "prompt_completion_correlation": correlation(prompts, completions),
    }
    warnings = []
    if b <= 0:
        warnings.append("negative decode coefficient: the fit is not usable")
    if a <= 0:
        warnings.append("negative prefill coefficient: prompt lengths vary too little")
    corr = result["prompt_completion_correlation"]
    if corr is not None and abs(corr) > 0.9:
        warnings.append(
            f"prompt and completion lengths are correlated (r={corr:.2f}); the two "
            "coefficients are barely identifiable and the split is unreliable")
    # Prefill is quadratic in prompt length (the Indexer scores it unblocked), so
    # a linear term understates long prompts and leaks into the intercept.
    warnings.append("prefill is modelled linearly but is quadratic in prompt length; "
                    "treat the prefill rate as indicative and the decode rate as the result")
    result["warnings"] = warnings
    return result


def measured(rows: list[dict]) -> dict | None:
    """Aggregate the per-row decode block that new runs record directly."""
    blocks = [r["decode"] for r in rows if isinstance(r.get("decode"), dict)]
    if not blocks:
        return None
    steps = sum(b.get("decode_steps_measured") or 0 for b in blocks)
    seconds = sum(b.get("decode_seconds_measured") or 0.0 for b in blocks)
    prefill_tokens = sum(b.get("prefill_tokens") or 0 for b in blocks)
    prefill_seconds = sum(b.get("prefill_seconds") or 0.0 for b in blocks)
    return {
        "rows_with_timing": len(blocks),
        "decode_steps": steps,
        "decode_tokens_per_second": round(steps / seconds, 3) if seconds > 0 else None,
        "prefill_tokens_per_second": (
            round(prefill_tokens / prefill_seconds, 1) if prefill_seconds > 0 else None),
    }


def report(path: Path) -> None:
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"\n=== {path.name}  ({len(rows)} rows) ===")
    if not rows:
        print("  empty")
        return
    direct = measured(rows)
    if direct is not None:
        print("  measured per-forward timings present:")
        for key, value in direct.items():
            print(f"    {key:34s} {value}")
        return
    print("  no per-forward timings (run predates the instrumentation); estimating")
    result = fit(rows)
    if not result.pop("ok"):
        print(f"    cannot estimate: {result['why']}")
        return
    for key in ("rows", "r_squared", "decode_tokens_per_second",
                "seconds_per_decoded_token", "prefill_tokens_per_second",
                "fixed_overhead_seconds", "naive_completion_over_wallclock_tps",
                "prompt_completion_correlation"):
        value = result[key]
        if isinstance(value, float):
            value = round(value, 4)
        print(f"    {key:34s} {value}")
    for warning in result["warnings"]:
        print(f"    ! {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run_ID directory; scans questions/ and pairs/")
    parser.add_argument("--jsonl", nargs="*", default=[], help="explicit jsonl files")
    args = parser.parse_args()

    paths: list[Path] = [Path(p) for p in args.jsonl]
    if args.run:
        root = Path(args.run)
        paths += sorted(root.glob("questions/answers_*.jsonl"))
        paths += sorted(root.glob("pairs/pairs_*.jsonl"))
    if not paths:
        raise SystemExit("nothing to analyse: pass --run or --jsonl")
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"not a file: {path}")
        report(path)


if __name__ == "__main__":
    main()
