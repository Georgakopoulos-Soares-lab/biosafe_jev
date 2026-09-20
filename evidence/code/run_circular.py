"""Cyclic (circular) evaluation: present every item in all n rotations of its option list.

This is the experiment that supplies the control arm a one-directional reshuffle test
lacks. For every rotation the returned distribution is mapped back to ORIGINAL option
indices, so rotations are directly comparable and can be averaged.

Generalises the original four-option-only script to arbitrary n, which is required for the
LAB-Bench subsets (2 to 10 options). --max-rotations caps the per-item cost for wide items.

Usage:
  TYPESAFE_API_KEY=... python3 run_circular.py cyber bio
  TYPESAFE_API_KEY=... python3 run_circular.py bio-robust --max-rotations 4
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import RAW
from jev_client import (
    build_choice_payload,
    labels_for,
    make_session,
    parse_choice,
    post,
)

MAX_WORKERS = 16


def load_items(subject):
    """Take questions and gold indices from an existing single-pass run."""
    path = os.path.join(RAW, f"{subject}.jsonl")
    items = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r["error"] is not None:
                continue
            items.append({"index": r["index"], "question": r["question"],
                          "choices": r["choices"], "gold": r["gold_idx"]})
    return items


def run(subject, max_rot, label_mode):
    items = load_items(subject)
    session = make_session()
    tasks = []
    for i, it in enumerate(items):
        n_rot = min(len(it["choices"]), max_rot) if max_rot else len(it["choices"])
        tasks += [(i, rot) for rot in range(n_rot)]
    print(f"[{subject}] {len(items)} items, {len(tasks)} calls", file=sys.stderr)
    out = {}

    def work(i, rot):
        ch = items[i]["choices"]
        n = len(ch)
        # Prompt position p shows the option originally at (p + rot) % n.
        rotated = [ch[(p + rot) % n] for p in range(n)]
        labels = labels_for(n, label_mode)
        data, err = post(session, build_choice_payload(items[i]["question"], rotated, labels))
        if err is not None:
            return i, rot, None
        p = parse_choice(data, labels)
        if p["chosen_pos"] < 0:
            return i, rot, None
        # Invert the rotation: map both the choice and the whole vector back to originals.
        probs_by_orig = [0.0] * n
        for pos in range(n):
            probs_by_orig[(pos + rot) % n] = p["probs_by_position"][pos]
        return i, rot, ((p["chosen_pos"] + rot) % n, probs_by_orig)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(work, i, rot) for i, rot in tasks]
        done = 0
        for fut in as_completed(futures):
            i, rot, val = fut.result()
            out[(i, rot)] = val
            done += 1
            if done % 500 == 0:
                print(f"[{subject}] {done}/{len(tasks)}", file=sys.stderr)

    recs, dropped = [], 0
    for i, it in enumerate(items):
        n_rot = min(len(it["choices"]), max_rot) if max_rot else len(it["choices"])
        per = [out.get((i, rot)) for rot in range(n_rot)]
        if any(p is None for p in per):
            dropped += 1
            continue
        recs.append({"index": it["index"], "gold": it["gold"],
                     "n_options": len(it["choices"]),
                     "choices_by_rot": [p[0] for p in per],
                     "probs_by_rot": [p[1] for p in per]})
    path = os.path.join(RAW, f"perm_{subject}.json")
    with open(path, "w") as fh:
        json.dump(recs, fh)
    print(f"[{subject}] wrote {len(recs)} complete items ({dropped} dropped) -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="+")
    ap.add_argument("--max-rotations", type=int, default=None,
                    help="cap rotations per item; default is one per option")
    ap.add_argument("--label-mode", default="positional",
                    choices=["positional", "symbolic"])
    args = ap.parse_args()
    for s in args.subjects:
        run(s, args.max_rotations, args.label_mode)


if __name__ == "__main__":
    main()
