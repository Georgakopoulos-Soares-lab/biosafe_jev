r"""Paper/code parity check.

`check_build.py` verifies that every macro the text cites is defined. This goes further and
verifies that the numbers actually *rendered in the PDF* are the numbers the analysis
pipeline produced, by re-deriving each macro's value from derived/stats.json and confirming
the string appears in the extracted PDF text.

It also flags claims that are easy to leave stale when results change: dataset sizes, call
counts, and any prose assertion of the form "X is larger than Y" that the code can check.

Exit code is non-zero if any macro value is missing from the rendered text.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "evidence", "code")
sys.path.insert(0, CODE)

import export_numbers  # noqa: E402

fails, notes = [], []

# --- 1. macros are regenerated from stats.json and must match numbers.tex on disk
fresh = export_numbers.build()
on_disk = dict(re.findall(r"\\newcommand\{\\(stat[A-Za-z]+)\}\{(.*)\}",
                          open(os.path.join(HERE, "numbers.tex")).read()))
drift = sorted(k for k in fresh if on_disk.get(k) != fresh[k])
if drift:
    fails.append(f"numbers.tex is stale for {len(drift)} macro(s): {', '.join(drift[:8])}"
                 + (" ..." if len(drift) > 8 else ""))

# --- 2. every macro the text cites must appear, rendered, in the PDF
main = open(os.path.join(HERE, "main.tex")).read()
body = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in main.splitlines())
used = sorted(set(re.findall(r"\\(stat[A-Za-z]+)", body)))
pdf = subprocess.run(["pdftotext", os.path.join(HERE, "main.pdf"), "-"],
                     capture_output=True, text=True).stdout
# pdftotext inserts line breaks and uses Unicode minus; normalise before searching.
flat = re.sub(r"\s+", " ", pdf).replace("−", "-").replace("–", "-")

def rendered(value):
    """The literal a macro produces, as it would appear in the text layer."""
    v = value.replace("\\%", "%").replace("{,}", ",").replace("\\,", " ")
    v = re.sub(r"\\mbox\{([^}]*)\}", r"\1", v)
    v = re.sub(r"\\!|\\\\|\$", "", v)
    v = re.sub(r"\\times", "x", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v

missing = []
for k in used:
    if k not in on_disk:
        continue
    val = rendered(on_disk[k])
    if not val or re.search(r"10\^|\\", val):     # scientific notation is typeset, skip
        continue
    if val not in flat:
        missing.append(f"{k}={val}")
if missing:
    fails.append(f"{len(missing)} macro value(s) cited but not found in the rendered PDF: "
                 + ", ".join(missing[:10]))

# --- 3. cross-checks between prose claims and the data
S = json.load(open(os.path.join(HERE, "..", "evidence", "derived", "stats.json")))
checks = [
    ("pooled item count matches the dataset total",
     S["headline"]["totals"]["n"] == sum(v["n"] for v in S["headline"]["per_dataset"].values())),
    ("call total equals single-pass plus circular",
     S["a10_cost"]["n_calls_total"]
     == S["a10_cost"]["n_calls_single_pass"] + S["a10_cost"]["n_calls_circular"]),
    ("ECE on p_max is better than on the vendor field",
     S["a2_calibration"]["pooled"]["on_pmax"]["ece_fixed"]
     < S["a2_calibration"]["pooled"]["on_confidence_field"]["ece_fixed"]),
    ("rotation instability exceeds identical-call instability on both suites",
     all(S["a11_repeatability"][k]["inconsistency_rotations"]
         > S["a11_repeatability"][k]["inconsistency_identical"] for k in ("bio", "cyber"))),
    ("rotation averaging beats identical averaging on cyber",
     S["a11_repeatability"]["cyber"]["rotation_beats_identical"]),
    ("Bio cascade advantage is NOT claimed significant",
     S["a6_cascade"]["bio"]["frac_random_draws_beating_router_at_0.4"] > 0.05),
    ("observed stability/error ratio does not exceed the null",
     not S["a5_circular"]["cyber"]["exceeds_null"]),
    # Retained accuracy is reported twice -- in Table 1 and in the prose. They come from
    # different analysis blocks, so guard against them drifting apart again.
    ("retained accuracy agrees between Table 1 and the risk-coverage curve",
     all(abs(S["a7_breakeven"][k]["by_deferral"]["0.4"]["retained_accuracy"]
             - S["a4_selective_prediction"]["risk_coverage"][k]["acc_at_coverage"]["0.6"])
         < 1e-9 for k in S["a7_breakeven"])),
]
for label, ok in checks:
    if not ok:
        fails.append(f"consistency check failed: {label}")

# --- 4. claims that must NOT appear (scope discipline)
banned = [
    ("no usage field", "contradicted: the endpoint does report usage"),
    ("screening accuracy", "WMDP accuracy must not be described as screening performance"),
    ("hazard detection performance", "not evaluated"),
]
for phrase, why in banned:
    if phrase.lower() in flat.lower():
        fails.append(f"banned phrase present ({why}): '{phrase}'")

for n in notes:
    print(f"note: {n}")
if fails:
    print("\nPARITY CHECK FAILED")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print(f"parity check passed: {len(used)} macros cited, {len(checks)} consistency checks")
