"""Shared client for the TypeSafe System-One endpoint.

Consolidates the request construction and retry logic that the evaluation runners share.
The API key is read from TYPESAFE_API_KEY and is never written to disk or to a result file.
"""
import json
import os
import time
from datetime import datetime, timezone

import requests

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
MAX_RETRIES = 6
RETRY_STATUS = (429, 500, 502, 503, 504, 529)

ALPHA_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Deliberately non-alphabetic, so that no ordinal prior over labels is available.
SYMBOL_LABELS = "◆▲●■★✚✦❖◗⬟"


def make_session():
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is not set")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    return s


def labels_for(n, mode="positional", rng=None):
    """Return the label shown at each of the n prompt positions.

    positional : A, B, C, ...        -- letter identity and position are collinear
    shuffled   : a permutation of A..  -- breaks that collinearity, keeps the alphabet
    symbolic   : non-alphabetic glyphs -- removes the ordinal prior entirely
    """
    if mode == "positional":
        return list(ALPHA_LABELS[:n])
    if mode == "shuffled":
        if rng is None:
            raise ValueError("shuffled label mode needs a seeded rng")
        lab = list(ALPHA_LABELS[:n])
        rng.shuffle(lab)
        return lab
    if mode == "symbolic":
        return list(SYMBOL_LABELS[:n])
    raise ValueError(f"unknown label mode {mode!r}")


def build_choice_payload(question, choices, labels, fmt="both",
                         instructions="Select the single correct answer to the "
                                      "multiple-choice question given in the state."):
    """Build a `choice` request.

    fmt controls where the options appear, which the original harness never ablated:
      both     -- enumerated in the state text AND as category keys (the original)
      state    -- enumerated in the state text only; categories carry bare labels
      criteria -- categories only; the state carries just the question
    """
    if fmt == "criteria":
        state = f"Question: {question}"
    else:
        lines = [f"Question: {question}"]
        lines += [f"{lab}. {ch}" for lab, ch in zip(labels, choices)]
        state = "\n".join(lines)
    if fmt == "state":
        criteria = {lab: f"option {lab}" for lab in labels}
    else:
        criteria = {lab: str(ch) for lab, ch in zip(labels, choices)}
    return {"state": state, "model": MODEL,
            "questions": {"answer": {"type": "choice", "instructions": instructions,
                                     "criteria": criteria}}}


def post(session, payload, timeout=60):
    """POST with exponential backoff. Returns (json, None) or (None, error_string).

    The response carries `model` (the alias resolved to a concrete version) and `usage`
    (input/output token counts); callers should persist both, so that cost and provenance
    are measured rather than reconstructed.
    """
    delay, last = 1.0, None
    for _ in range(MAX_RETRIES):
        try:
            resp = session.post(API_URL, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last = str(exc)
            time.sleep(delay)
            delay = min(delay * 2, 20)
            continue
        if resp.status_code == 200:
            return resp.json(), None
        if resp.status_code in RETRY_STATUS:
            last = f"HTTP {resp.status_code}"
            time.sleep(delay)
            delay = min(delay * 2, 20)
            continue
        return None, f"HTTP {resp.status_code}: {resp.text[:300]}"
    return None, last


def parse_choice(data, labels, key="answer"):
    """Extract the selected position and the probability vector ordered by PROMPT POSITION,
    together with the provenance and usage the endpoint reports."""
    ans = data["answers"][key]
    chosen = ans.get("choice")
    pos = labels.index(chosen) if chosen in labels else -1
    probs = ans.get("probabilities") or {}
    return {
        "chosen_label": chosen,
        "chosen_pos": pos,
        "probs_by_position": [float(probs.get(lab, 0.0)) for lab in labels],
        "confidence": ans.get("confidence"),
        "model_version": data.get("model"),
        "usage": data.get("usage"),
    }


def utc_now():
    """Timestamp for provenance, as an ISO-8601 string in UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(path, run_name, n_calls, started, model_version, extra=None):
    """Record what was run, against which endpoint and model version, and when."""
    manifest = {
        "run": run_name,
        "endpoint": API_URL,
        "model_alias": MODEL,
        "model_version": model_version,
        "started_utc": started,
        "finished_utc": utc_now(),
        "n_calls": n_calls,
    }
    if extra:
        manifest.update(extra)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
