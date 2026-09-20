"""Repeatability study: separate answer-order sensitivity from run-to-run variation.

The cyclic experiment (run_circular.py) confounds two sources of disagreement. When an item
is answered differently under two rotations, that may be because the option order changed,
or simply because the endpoint is not deterministic. This script measures the second source
directly so the two can be told apart.

Two designs, both writing to raw/:

  identical   Every item is queried k times with a BYTE-IDENTICAL prompt (rotation 0, the
              canonical option order). With k = 4 this matches the four-rotation budget
              exactly, so "average 4 identical calls" and "average 4 rotations" can be
              compared at equal cost. -> raw/repeat_<subject>.json

  tokens      A small sample per dataset, queried once each, purely to record the `usage`
              the endpoint reports. The original runs predate usage capture, so per-dataset
              token rates have to be measured separately to cost them.
              -> raw/tokens_<subject>.json

  factorial   A stratified subsample is queried k times at EACH of the n rotations, giving
              a two-factor design (rotation x repeat) and hence a variance decomposition
              into between-rotation and within-rotation components.
              -> raw/repeatfac_<subject>.json

In both designs the task list is shuffled before dispatch, so repeats of the same item are
never issued back to back. Without this, any server-side caching or locality effect would
masquerade as determinism.

Usage:
  TYPESAFE_API_KEY=... python3 run_repeat.py identical cyber bio --repeats 4
  TYPESAFE_API_KEY=... python3 run_repeat.py factorial cyber bio --repeats 3 --sample 500
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import RAW
from jev_client import (
    build_choice_payload,
    labels_for,
    make_session,
    parse_choice,
    post,
    utc_now,
    write_manifest,
)

MAX_WORKERS = 16
DISPATCH_SEED = 505050   # shuffles dispatch order only; does not touch prompt content
SAMPLE_SEED = 4242       # selects the factorial subsample


def load_items(subject):
    """Questions and gold indices from the existing single-pass run."""
    items = []
    with open(os.path.join(RAW, f"{subject}.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if r["error"] is None:
                items.append({"index": r["index"], "question": r["question"],
                              "choices": r["choices"], "gold": r["gold_idx"]})
    return items


def _call(session, item, rot):
    """One query. `rot` shifts which original option appears at each prompt position."""
    ch = item["choices"]
    n = len(ch)
    rotated = [ch[(p + rot) % n] for p in range(n)]
    labels = labels_for(n, "positional")
    data, err = post(session, build_choice_payload(item["question"], rotated, labels))
    if err is not None:
        return None
    parsed = parse_choice(data, labels)
    if parsed["chosen_pos"] < 0:
        return None
    # Map the choice and the whole vector back to original option indices.
    probs = [0.0] * n
    for pos in range(n):
        probs[(pos + rot) % n] = parsed["probs_by_position"][pos]
    return {"choice": (parsed["chosen_pos"] + rot) % n, "probs": probs,
            "usage": parsed["usage"], "model_version": parsed["model_version"]}


def _dispatch(session, items, tasks, label):
    """Run (item_idx, rotation, repeat) triples concurrently in shuffled order."""
    rng = random.Random(DISPATCH_SEED)
    rng.shuffle(tasks)
    out, done = {}, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_call, session, items[i], rot): (i, rot, rep)
                   for i, rot, rep in tasks}
        for fut in as_completed(futures):
            key = futures[fut]
            out[key] = fut.result()
            done += 1
            if done % 1000 == 0:
                print(f"[{label}] {done}/{len(tasks)}", file=sys.stderr)
    return out


def run_identical(subject, k):
    items = load_items(subject)
    session = make_session()
    started = utc_now()
    tasks = [(i, 0, rep) for i in range(len(items)) for rep in range(k)]
    print(f"[{subject}/identical] {len(items)} items x {k} repeats "
          f"= {len(tasks)} calls", file=sys.stderr)
    out = _dispatch(session, items, tasks, f"{subject}/identical")

    recs, dropped, usage, version = [], 0, defaultdict(int), None
    for i, it in enumerate(items):
        per = [out.get((i, 0, rep)) for rep in range(k)]
        if any(p is None for p in per):
            dropped += 1
            continue
        for p in per:
            version = p["model_version"] or version
            for fld in ("input_tokens", "output_tokens"):
                usage[fld] += (p["usage"] or {}).get(fld, 0)
        recs.append({"index": it["index"], "gold": it["gold"],
                     "n_options": len(it["choices"]),
                     "choices_by_rep": [p["choice"] for p in per],
                     "probs_by_rep": [p["probs"] for p in per]})
    path = os.path.join(RAW, f"repeat_{subject}.json")
    with open(path, "w") as fh:
        json.dump(recs, fh)
    write_manifest(os.path.join(RAW, f"manifest_repeat_{subject}.json"),
                   f"identical:{subject}", len(tasks), started, version,
                   {"repeats": k, "dropped_items": dropped, "usage_totals": dict(usage)})
    print(f"[{subject}/identical] wrote {len(recs)} items ({dropped} dropped) -> {path}")
    print(f"    tokens: {dict(usage)}")


def run_factorial(subject, k, sample):
    items = load_items(subject)
    rng = random.Random(SAMPLE_SEED)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    idx = sorted(idx[:sample])
    session = make_session()
    started = utc_now()
    tasks = [(i, rot, rep) for i in idx
             for rot in range(len(items[i]["choices"])) for rep in range(k)]
    print(f"[{subject}/factorial] {len(idx)} items x rotations x {k} repeats "
          f"= {len(tasks)} calls", file=sys.stderr)
    out = _dispatch(session, items, tasks, f"{subject}/factorial")

    recs, dropped, version = [], 0, None
    for i in idx:
        n_rot = len(items[i]["choices"])
        grid = [[out.get((i, rot, rep)) for rep in range(k)] for rot in range(n_rot)]
        if any(p is None for row in grid for p in row):
            dropped += 1
            continue
        version = grid[0][0]["model_version"] or version
        recs.append({"index": items[i]["index"], "gold": items[i]["gold"],
                     "n_options": n_rot,
                     # [rotation][repeat]
                     "choices": [[p["choice"] for p in row] for row in grid],
                     "probs": [[p["probs"] for p in row] for row in grid]})
    path = os.path.join(RAW, f"repeatfac_{subject}.json")
    with open(path, "w") as fh:
        json.dump(recs, fh)
    write_manifest(os.path.join(RAW, f"manifest_repeatfac_{subject}.json"),
                   f"factorial:{subject}", len(tasks), started, version,
                   {"repeats": k, "sample": sample, "dropped_items": dropped})
    print(f"[{subject}/factorial] wrote {len(recs)} items ({dropped} dropped) -> {path}")


def run_tokens(subject, sample):
    """Measure the token cost of a representative sample of this dataset's prompts."""
    items = load_items(subject)
    rng = random.Random(SAMPLE_SEED)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    idx = sorted(idx[:min(sample, len(items))])
    session = make_session()
    started = utc_now()
    out = _dispatch(session, items, [(i, 0, 0) for i in idx], f"{subject}/tokens")

    rows, version = [], None
    for i in idx:
        r = out.get((i, 0, 0))
        if r is None or not r["usage"]:
            continue
        version = r["model_version"] or version
        rows.append({"index": items[i]["index"],
                     "input_tokens": r["usage"].get("input_tokens"),
                     "output_tokens": r["usage"].get("output_tokens"),
                     "n_chars": len(items[i]["question"])
                               + 2 * sum(len(str(c)) for c in items[i]["choices"])})
    path = os.path.join(RAW, f"tokens_{subject}.json")
    with open(path, "w") as fh:
        json.dump(rows, fh)
    write_manifest(os.path.join(RAW, f"manifest_tokens_{subject}.json"),
                   f"tokens:{subject}", len(idx), started, version, {"sample": len(rows)})
    mean_in = sum(r["input_tokens"] for r in rows) / len(rows)
    print(f"[{subject}/tokens] n={len(rows)} mean input tokens/call = {mean_in:.1f} -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", choices=["identical", "factorial", "tokens"])
    ap.add_argument("subjects", nargs="+")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--sample", type=int, default=500,
                    help="items per dataset for the factorial design")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items for a smoke test")
    args = ap.parse_args()

    for subject in args.subjects:
        if args.design == "tokens":
            run_tokens(subject, args.limit or args.sample)
        elif args.design == "identical":
            if args.limit:
                run_identical_limited(subject, args.repeats, args.limit)
            else:
                run_identical(subject, args.repeats)
        else:
            run_factorial(subject, args.repeats, args.limit or args.sample)


def run_identical_limited(subject, k, limit):
    """Smoke test: same path as run_identical but on the first `limit` items."""
    items = load_items(subject)[:limit]
    session = make_session()
    tasks = [(i, 0, rep) for i in range(len(items)) for rep in range(k)]
    out = _dispatch(session, items, tasks, f"{subject}/smoke")
    agree = 0
    for i in range(len(items)):
        per = [out.get((i, 0, rep)) for rep in range(k)]
        if all(p is not None for p in per):
            agree += len({p["choice"] for p in per}) == 1
    print(f"[{subject}/smoke] {agree}/{len(items)} items agreed across {k} identical calls")
    print(f"    sample usage: {out[next(iter(out))]['usage']}")


if __name__ == "__main__":
    main()
