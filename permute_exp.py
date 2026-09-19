"""Cyclic-permutation (circular evaluation) experiment.

Fixes the flawed 15-item reshuffle test in the original report, which selected only
originally-INCORRECT items (selection on the dependent variable) and had no control group.

For every 4-option item we run all 4 cyclic rotations of the option list, map the returned
probability vector back to ORIGINAL option indices, and measure:
  - accuracy per rotation
  - permutation consistency (same original option chosen in all 4 rotations)
  - probability-ensemble and majority-vote accuracy
  - flip rates for originally-correct AND originally-incorrect items (the missing control)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

API_URL = "https://api.typesafe.ai/v1/systemone"
API_KEY = os.environ["TYPESAFE_API_KEY"]
LETTERS = "ABCD"
MAX_WORKERS = 16

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})


def call(question, choices):
    """choices is the already-rotated list. Returns (chosen_pos, prob_by_pos) or None."""
    state = "\n".join([f"Question: {question}"] + [f"{l}. {c}" for l, c in zip(LETTERS, choices)])
    payload = {
        "state": state,
        "model": "jev-latest",
        "questions": {"answer": {
            "type": "choice",
            "instructions": "Select the single correct answer to the multiple-choice question given in the state.",
            "criteria": {l: str(c) for l, c in zip(LETTERS, choices)},
        }},
    }
    delay = 1.0
    for _ in range(6):
        try:
            r = session.post(API_URL, json=payload, timeout=60)
        except requests.RequestException:
            time.sleep(delay); delay = min(delay * 2, 20); continue
        if r.status_code == 200:
            a = r.json()["answers"]["answer"]
            if a["choice"] not in LETTERS:
                return None
            probs = [a["probabilities"].get(l, 0.0) for l in LETTERS]
            return LETTERS.index(a["choice"]), probs
        if r.status_code in (429, 529):
            time.sleep(delay); delay = min(delay * 2, 20); continue
        return None
    return None


def run(subject, limit=None):
    rows = [json.loads(l) for l in open(f"results/{subject}.jsonl")]
    rows = [r for r in rows if r["error"] is None and len(r["choices"]) == 4]
    if limit:
        rows = rows[:limit]
    tasks = [(i, rot) for i in range(len(rows)) for rot in range(4)]
    out = {}

    def work(i, rot):
        ch = rows[i]["choices"]
        # rotation: position p in prompt holds original option (p+rot) % 4
        rotated = [ch[(p + rot) % 4] for p in range(4)]
        res = call(rows[i]["question"], rotated)
        if res is None:
            return i, rot, None
        chosen_pos, probs_by_pos = res
        orig_choice = (chosen_pos + rot) % 4
        probs_by_orig = [0.0] * 4
        for p in range(4):
            probs_by_orig[(p + rot) % 4] = probs_by_pos[p]
        return i, rot, (orig_choice, probs_by_orig)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(work, i, rot) for i, rot in tasks]
        done = 0
        for f in as_completed(futs):
            i, rot, v = f.result()
            out[(i, rot)] = v
            done += 1
            if done % 500 == 0:
                print(f"[{subject}] {done}/{len(tasks)}", file=sys.stderr)

    recs = []
    for i, r in enumerate(rows):
        per = [out.get((i, rot)) for rot in range(4)]
        if any(p is None for p in per):
            continue
        recs.append({
            "index": r["index"],
            "gold": r["gold_idx"],
            "choices_by_rot": [p[0] for p in per],
            "probs_by_rot": [p[1] for p in per],
        })
    with open(f"results/perm_{subject}.json", "w") as f:
        json.dump(recs, f)
    print(f"[{subject}] wrote {len(recs)} complete items")
    return recs


def report(subject, recs):
    gold = np.array([r["gold"] for r in recs])
    ch = np.array([r["choices_by_rot"] for r in recs])          # (N,4)
    pr = np.array([r["probs_by_rot"] for r in recs])            # (N,4rot,4opt)
    N = len(recs)
    print(f"\n===== {subject}  (N={N} four-option items, {N*4} calls) =====")
    accs = [(ch[:, rot] == gold).mean() for rot in range(4)]
    print("  accuracy per rotation:", " ".join(f"{a:.4f}" for a in accs),
          f" | mean={np.mean(accs):.4f} sd={np.std(accs):.4f} spread={max(accs)-min(accs):.4f}")
    consistent = (ch == ch[:, [0]]).all(1)
    print(f"  permutation consistency (all 4 rotations agree): {consistent.mean()*100:.1f}%")
    print(f"    acc | consistent   : {(ch[consistent,0]==gold[consistent]).mean():.4f}  (n={consistent.sum()})")
    print(f"    acc | inconsistent : {(ch[~consistent,0]==gold[~consistent]).mean():.4f}  (n={(~consistent).sum()})")
    ens = pr.mean(1).argmax(1)
    print(f"  probability-ensemble accuracy : {(ens==gold).mean():.4f}")
    maj = np.array([np.bincount(row, minlength=4).argmax() for row in ch])
    print(f"  majority-vote accuracy        : {(maj==gold).mean():.4f}")
    print(f"  single-rotation baseline (r0) : {accs[0]:.4f}")
    # THE CONTROL the original experiment lacked
    c0 = ch[:, 0] == gold
    flip_w2c = np.array([(ch[i, 1:] == gold[i]).any() for i in range(N)])
    flip_c2w = np.array([(ch[i, 1:] != gold[i]).any() for i in range(N)])
    print(f"  among originally-WRONG   (n={(~c0).sum():4d}): {flip_w2c[~c0].mean()*100:5.1f}% become correct in >=1 other rotation")
    print(f"  among originally-CORRECT (n={c0.sum():4d}): {flip_c2w[c0].mean()*100:5.1f}% become wrong  in >=1 other rotation   <-- CONTROL")
    # position preference across rotations
    chosen_pos = np.array([[(ch[i, rot] - rot) % 4 for rot in range(4)] for i in range(N)])
    cnt = np.bincount(chosen_pos.ravel(), minlength=4)
    print(f"  chosen PROMPT-POSITION counts (pos0..3): {cnt.tolist()}  -> pos0 share {cnt[0]/cnt.sum()*100:.1f}% (uniform=25%)")
    return dict(accs=accs, consistency=float(consistent.mean()), ens=float((ens==gold).mean()),
                maj=float((maj==gold).mean()), r0=float(accs[0]),
                flip_w2c=float(flip_w2c[~c0].mean()), flip_c2w=float(flip_c2w[c0].mean()),
                pos_counts=cnt.tolist(), N=N)


if __name__ == "__main__":
    subs = sys.argv[1:] or ["cyber", "bio"]
    allr = {}
    for s in subs:
        recs = run(s)
        allr[s] = report(s, recs)
    json.dump(allr, open("results/perm_summary.json", "w"), indent=2)
