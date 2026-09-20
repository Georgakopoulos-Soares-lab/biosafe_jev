"""Single-pass evaluation runner.

Reproduces the primary runs and adds the two ablation arms the original harness lacked:

  --label-mode {positional,shuffled,symbolic}
      The original prompt assigned letters in positional order, making letter identity and
      prompt position perfectly collinear, so a first-option preference could not be
      distinguished from a letter-A preference. `shuffled` and `symbolic` break that.

  --format-mode {both,state,criteria}
      The original serialised every option twice (state text and category keys). Given the
      documented sensitivity of models to prompt formatting, this needs an ablation.

Usage:
  TYPESAFE_API_KEY=... python3 run_eval.py bio cyber
  TYPESAFE_API_KEY=... python3 run_eval.py cyber --label-mode symbolic --tag sym
"""
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from common import RAW
from jev_client import (
    build_choice_payload,
    labels_for,
    make_session,
    parse_choice,
    post,
)

DATA = os.path.join(os.path.dirname(RAW), "..", "data")
MAX_WORKERS = 16
SHUFFLE_SEED = 1234        # matches the seed used for the primary LAB-Bench runs
LABEL_SEED = 99991         # independent stream, so label mode cannot perturb option order

QUOTED_STR = re.compile(r"'((?:\\.|[^'\\])*)'|\"((?:\\.|[^\"\\])*)\"")

LABBENCH_CONFIGS = {
    "labbench-cloning": "CloningScenarios", "labbench-protocol": "ProtocolQA",
    "labbench-seq": "SeqQA", "labbench-db": "DbQA",
    "labbench-litqa2": "LitQA2", "labbench-supp": "SuppQA",
}


def parse_numpy_repr_choices(s):
    """WMDP-Bio-Robust stores `choices` as a numpy array repr that mixes quote styles:
    entries containing an apostrophe are double-quoted. Match both."""
    out = []
    for m in QUOTED_STR.finditer(s):
        val = m.group(1) if m.group(1) is not None else m.group(2)
        out.append(val.replace("\\'", "'").replace('\\"', '"'))
    return out


def load_labbench_rows(config_name):
    df = pd.read_parquet(os.path.join(DATA, f"labbench-{config_name}.parquet"))
    rows, skipped, deduped = [], 0, 0
    rng = random.Random(SHUFFLE_SEED)
    for rec in df.to_dict("records"):
        ideal = str(rec["ideal"])
        distractors = [str(d) for d in rec["distractors"]]
        options = list(dict.fromkeys([ideal] + distractors))
        if len(options) < 1 + len(distractors):
            deduped += 1          # a distractor duplicated the reference answer
        if len(options) < 2:
            skipped += 1
            continue
        order = list(range(len(options)))
        rng.shuffle(order)
        shuffled = [options[j] for j in order]
        rows.append({"question": rec["question"], "choices": shuffled,
                     "answer": shuffled.index(ideal),
                     "config": rec.get("subtask", config_name)})
    if skipped or deduped:
        print(f"[{config_name}] skipped {skipped} (<2 options), "
              f"de-duplicated {deduped} rows", file=sys.stderr)
    return rows


def load_rows(subject):
    if subject == "bio-robust":
        df = pd.read_parquet(os.path.join(DATA, "wmdp-bio-robust.parquet"))
        rows = []
        for rec in df.to_dict("records"):
            rec["choices"] = parse_numpy_repr_choices(rec["choices"])
            rows.append(rec)
        return rows
    if subject in LABBENCH_CONFIGS:
        return load_labbench_rows(LABBENCH_CONFIGS[subject])
    df = pd.read_parquet(os.path.join(DATA, f"wmdp-{subject}.parquet"))
    return df.to_dict("records")


def run(subject, label_mode, fmt, out_path):
    rows = load_rows(subject)
    session = make_session()
    results = [None] * len(rows)

    def work(i, row):
        choices = list(row["choices"])
        n = len(choices)
        # Per-item rng keyed on the index, so a rerun reproduces the same label assignment.
        rng = random.Random(LABEL_SEED + i)
        labels = labels_for(n, label_mode, rng)
        payload = build_choice_payload(row["question"], choices, labels, fmt)
        data, err = post(session, payload)
        gold_idx = int(row["answer"])
        rec = {"subject": subject, "index": i, "question": row["question"],
               "choices": choices, "gold_idx": gold_idx,
               "labels": labels, "label_mode": label_mode, "format_mode": fmt,
               "error": err}
        if err is None:
            p = parse_choice(data, labels)
            rec.update({
                "pred_idx": p["chosen_pos"],          # position, which indexes `choices`
                "pred_label": p["chosen_label"],
                "gold_label": labels[gold_idx] if gold_idx < n else None,
                "correct": p["chosen_pos"] == gold_idx,
                "confidence": p["confidence"],
                # keyed by option index, not by label, so arms stay comparable
                "probabilities": {str(j): v for j, v in enumerate(p["probs_by_position"])},
            })
        else:
            rec.update({"pred_idx": -1, "pred_label": None, "gold_label": None,
                        "correct": False, "confidence": None, "probabilities": None})
        if "config" in row:
            rec["config"] = row["config"]
        return i, rec

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(work, i, r) for i, r in enumerate(rows)]
        done = 0
        for fut in as_completed(futures):
            i, rec = fut.result()
            results[i] = rec
            done += 1
            if done % 200 == 0 or done == len(rows):
                print(f"[{subject}/{label_mode}/{fmt}] {done}/{len(rows)}", file=sys.stderr)

    with open(out_path, "w") as fh:
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
    ok = [r for r in results if r["error"] is None]
    acc = sum(r["correct"] for r in ok) / len(ok) if ok else 0.0
    print(f"{subject} [{label_mode}/{fmt}]: {acc:.4f} over {len(ok)} "
          f"({len(results) - len(ok)} errors) -> {out_path}")
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="+")
    ap.add_argument("--label-mode", default="positional",
                    choices=["positional", "shuffled", "symbolic"])
    ap.add_argument("--format-mode", default="both", choices=["both", "state", "criteria"])
    ap.add_argument("--tag", default=None, help="suffix for the output filename")
    args = ap.parse_args()

    for subject in args.subjects:
        suffix = f"_{args.tag}" if args.tag else ""
        out = os.path.join(RAW, f"{subject}{suffix}.jsonl")
        if os.path.exists(out) and not args.tag:
            print(f"refusing to overwrite {out} without --tag", file=sys.stderr)
            continue
        run(subject, args.label_mode, args.format_mode, out)


if __name__ == "__main__":
    main()
