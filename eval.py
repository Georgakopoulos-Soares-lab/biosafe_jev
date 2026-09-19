import json
import os
import random
import re
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

API_URL = "https://api.typesafe.ai/v1/systemone"
API_KEY = os.environ["TYPESAFE_API_KEY"]
MODEL = "jev-latest"
SUBJECTS = ["bio", "chem", "cyber"]
LETTERS = string.ascii_uppercase
MAX_WORKERS = 16
MAX_RETRIES = 6

QUOTED_STR = re.compile(r"'((?:\\.|[^'\\])*)'|\"((?:\\.|[^\"\\])*)\"")


def parse_numpy_repr_choices(s):
    out = []
    for m in QUOTED_STR.finditer(s):
        val = m.group(1) if m.group(1) is not None else m.group(2)
        out.append(val.replace("\\'", "'").replace('\\"', '"'))
    return out


LABBENCH_CONFIGS = {
    "labbench-cloning": "CloningScenarios",
    "labbench-protocol": "ProtocolQA",
    "labbench-seq": "SeqQA",
    "labbench-db": "DbQA",
    "labbench-litqa2": "LitQA2",
    "labbench-supp": "SuppQA",
}


def load_labbench_rows(config_name):
    df = pd.read_parquet(f"data/labbench-{config_name}.parquet")
    rows = []
    skipped = 0
    rng = random.Random(1234)
    for rec in df.to_dict("records"):
        ideal = str(rec["ideal"])
        distractors = [str(d) for d in rec["distractors"]]
        options = list(dict.fromkeys([ideal] + distractors))  # dedup, ideal guaranteed present once
        if len(options) < 2:
            skipped += 1
            continue
        order = list(range(len(options)))
        rng.shuffle(order)
        shuffled = [options[j] for j in order]
        rows.append({
            "question": rec["question"],
            "choices": shuffled,
            "answer": shuffled.index(ideal),
            "config": rec.get("subtask", config_name),
        })
    if skipped:
        print(f"[labbench-{config_name}] skipped {skipped}/{len(df)} rows with <2 unique options", file=sys.stderr)
    return rows


def load_rows(subject):
    if subject == "bio-robust":
        df = pd.read_parquet("data/wmdp-bio-robust.parquet")
        rows = []
        for rec in df.to_dict("records"):
            choices = parse_numpy_repr_choices(rec["choices"])
            rec["choices"] = choices
            rows.append(rec)
        return rows
    if subject in LABBENCH_CONFIGS:
        return load_labbench_rows(LABBENCH_CONFIGS[subject])
    df = pd.read_parquet(f"data/wmdp-{subject}.parquet")
    return df.to_dict("records")

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
})


def build_payload(question, choices):
    letters = LETTERS[: len(choices)]
    state_lines = [f"Question: {question}"]
    for letter, choice in zip(letters, choices):
        state_lines.append(f"{letter}. {choice}")
    state = "\n".join(state_lines)
    criteria = {letter: str(choice) for letter, choice in zip(letters, choices)}
    return {
        "state": state,
        "model": MODEL,
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "Select the single correct answer to the multiple-choice question given in the state.",
                "criteria": criteria,
            }
        },
    }, letters


def call_jev(question, choices):
    payload, letters = build_payload(question, choices)
    delay = 1.0
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.post(API_URL, json=payload, timeout=60)
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(delay)
            delay = min(delay * 2, 20)
            continue
        if resp.status_code == 200:
            data = resp.json()
            ans = data["answers"]["answer"]
            choice_letter = ans["choice"]
            pred_idx = letters.index(choice_letter) if choice_letter in letters else -1
            return {
                "pred_letter": choice_letter,
                "pred_idx": pred_idx,
                "confidence": ans.get("confidence"),
                "probabilities": ans.get("probabilities"),
                "usage": data.get("usage"),
                "error": None,
            }
        if resp.status_code in (429, 529):
            time.sleep(delay)
            delay = min(delay * 2, 20)
            last_err = f"HTTP {resp.status_code}"
            continue
        last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        break
    return {
        "pred_letter": None,
        "pred_idx": -1,
        "confidence": None,
        "probabilities": None,
        "usage": None,
        "error": last_err,
    }


def eval_subject(subject):
    rows = load_rows(subject)
    results = [None] * len(rows)

    def work(i, row):
        choices = list(row["choices"])
        r = call_jev(row["question"], choices)
        gold_idx = int(row["answer"])
        gold_letter = LETTERS[gold_idx] if gold_idx < len(LETTERS) else None
        correct = (r["pred_idx"] == gold_idx) if r["error"] is None else False
        rec = {
            "subject": subject,
            "index": i,
            "question": row["question"],
            "choices": choices,
            "gold_idx": gold_idx,
            "gold_letter": gold_letter,
            "pred_letter": r["pred_letter"],
            "pred_idx": r["pred_idx"],
            "correct": correct,
            "confidence": r["confidence"],
            "probabilities": r["probabilities"],
            "error": r["error"],
        }
        if "config" in row:
            rec["config"] = row["config"]
        return i, rec

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(work, i, row) for i, row in enumerate(rows)]
        done = 0
        for fut in as_completed(futures):
            i, rec = fut.result()
            results[i] = rec
            done += 1
            if done % 100 == 0 or done == len(rows):
                print(f"[{subject}] {done}/{len(rows)}", file=sys.stderr)

    return results


def main():
    only = sys.argv[1:] or SUBJECTS
    all_results = {}
    for subject in only:
        print(f"=== Evaluating {subject} ===", file=sys.stderr)
        results = eval_subject(subject)
        all_results[subject] = results
        with open(f"results/{subject}.jsonl", "w") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")
        n = len(results)
        errors = sum(1 for r in results if r["error"] is not None)
        correct = sum(1 for r in results if r["correct"])
        acc = correct / n if n else 0.0
        print(f"{subject}: {correct}/{n} correct = {acc:.4f} (errors: {errors})")

    summary = {}
    for subject, results in all_results.items():
        n = len(results)
        correct = sum(1 for r in results if r["correct"])
        errors = sum(1 for r in results if r["error"] is not None)
        summary[subject] = {"n": n, "correct": correct, "accuracy": correct / n if n else 0.0, "errors": errors}
    total_n = sum(s["n"] for s in summary.values())
    total_correct = sum(s["correct"] for s in summary.values())
    total_errors = sum(s["errors"] for s in summary.values())
    summary["overall"] = {
        "n": total_n,
        "correct": total_correct,
        "accuracy": total_correct / total_n if total_n else 0.0,
        "errors": total_errors,
    }
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
