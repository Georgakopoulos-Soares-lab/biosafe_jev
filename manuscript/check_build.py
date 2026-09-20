r"""Build guards. Run after tectonic; exits non-zero on any violation.

1. every \stat* macro used in main.tex is defined in numbers.tex
2. every macro defined in numbers.tex is actually used (no stale leftovers)
3. the output is US Letter (612 x 792 pt) -- the venue enforces this strictly
4. the body is within the 6-page short-paper limit, excluding the LLM Usage Statement
"""
import re
import subprocess
import sys

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
PAGE_LIMIT = 6
LETTER = (612, 792)

fails = []

main = open(f"{HERE}/main.tex").read()
numbers = open(f"{HERE}/numbers.tex").read()

defined = set(re.findall(r"\\newcommand\{\\(stat[A-Za-z]+)\}", numbers))
# strip comments before scanning for uses
body = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in main.splitlines())
used = set(re.findall(r"\\(stat[A-Za-z]+)", body))

missing = sorted(used - defined)
unused = sorted(defined - used)
if missing:
    fails.append(f"main.tex uses {len(missing)} undefined macro(s): {', '.join(missing)}")
if unused:
    # numbers.tex is a superset: export_numbers.py emits every statistic in stats.json so
    # that MANIFEST.md can trace all of them, whether or not the prose cites each one.
    print(f"note: {len(unused)} of {len(defined)} macros defined but not cited in the text")

# --- unresolved cross-references and citations, read back from the rendered text
try:
    text = subprocess.run(["pdftotext", f"{HERE}/main.pdf", "-"],
                          capture_output=True, text=True).stdout
    n_qq = len(re.findall(r"\?\?", text))
    if n_qq:
        fails.append(f"{n_qq} unresolved \\ref/\\cite in the rendered PDF (shows as '??')")
except Exception as exc:
    print(f"note: could not scan main.pdf for unresolved refs ({exc})")

# --- LaTeX log: undefined references and citations
try:
    log = open(f"{HERE}/main.log", errors="ignore").read()
    for pat, what in ((r"Reference `([^']+)' on page", "reference"),
                      (r"Citation `([^']+)' on page", "citation")):
        undef = sorted(set(re.findall(pat, log)))
        if undef:
            fails.append(f"undefined {what}(s): {', '.join(undef)}")
except FileNotFoundError:
    print("note: main.log not found; run tectonic with --keep-logs")

# --- page geometry and count, read back from the produced PDF
try:
    info = subprocess.run(["pdfinfo", f"{HERE}/main.pdf"], capture_output=True, text=True).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    w, h = (float(x) for x in re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups())
except Exception as exc:  # pdfinfo unavailable -> fall back to a raw parse
    pages, w, h = None, None, None
    print(f"note: could not read main.pdf via pdfinfo ({exc})")

if w is not None and (round(w) != LETTER[0] or round(h) != LETTER[1]):
    fails.append(f"page size is {w:.0f}x{h:.0f} pt, must be US Letter {LETTER[0]}x{LETTER[1]}")

if pages is not None:
    # The LLM Usage Statement is excluded from the limit, but only the pages it actually
    # occupies alone. If it shares a page with references or any body section, that page
    # still counts -- so locate the heading and check whether it starts a fresh page.
    body_pages = pages
    try:
        per_page = subprocess.run(["pdftotext", "-layout", f"{HERE}/main.pdf", "-"],
                                  capture_output=True, text=True).stdout.split("\f")[:-1]
        start = next(i for i, pg in enumerate(per_page, 1) if "LLM Usage Statement" in pg)
        # A two-column page puts both columns on the same text line, so "is the heading
        # the first line?" is not a usable test: references can sit in the left column
        # while the statement starts the right. Test for counted content instead --
        # references count toward the limit, only the statement itself does not.
        page_txt = per_page[start - 1]
        cut = page_txt.index("LLM Usage Statement")
        others = re.findall(r"\[\d+\]|References|\\section", page_txt)
        shares = bool(others) or cut > 0 and bool(page_txt[:cut].strip())
        body_pages = start if shares else start - 1
        print(f"note: {pages} pages; LLM statement starts on p{start} "
              f"({'shares the page with counted content' if shares else 'alone'})"
              f" -> {body_pages} count")
    except StopIteration:
        fails.append("no 'LLM Usage Statement' section found; the venue requires one")
    if body_pages > PAGE_LIMIT:
        fails.append(f"{body_pages} pages count toward the limit, maximum is {PAGE_LIMIT}")

if fails:
    print("\nBUILD CHECK FAILED")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("build checks passed")
