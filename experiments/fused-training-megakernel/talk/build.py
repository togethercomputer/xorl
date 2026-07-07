#!/usr/bin/env python3
"""Inline facts.json (+ fresh sweep scores, if a sweep log exists) into index.html.

Idempotent: replaces the <script id="mkgap-facts" type="application/json">...</script>
blob in place. The regex replacement is a FUNCTION (lambda) — a plain replacement
string would have its \\n / \\u escapes interpreted by the regex engine and silently
corrupt the JSON (see design-principles pitfall).

Usage: python3 build.py [sweep.log ...]
Sweep logs are score_shape.py output: lines of the form  SCORE {"shape": ...}.
"""

import glob
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_sweep(paths):
    fresh = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.startswith("SCORE "):
                row = json.loads(line[len("SCORE "):])
                fresh[row["shape"]] = {
                    "mk": row["megakernel_us"],
                    "base": row["compile_cudagraph_plus_us"],
                    "ratio": row["ratio"],
                }
    return fresh


def main():
    facts = json.loads((HERE / "facts.json").read_text())

    # drift asserts: the shape registry and every matrix/bucket row must agree
    names = [s["name"] for s in facts["shapes"]]
    assert len(names) == 12, names
    assert set(facts["matrix"]) == set(names)
    assert facts["buckets"]["order"] == names
    for k, v in facts["buckets"].items():
        if k in ("order", "note"):
            continue
        assert len(v["mk"]) == 12 and len(v["base"]) == 12, k

    sweep_paths = sys.argv[1:] or sorted(
        glob.glob("/tmp/claude-0/-home-apanda-xorl-oss/*/scratchpad/sweep-*.log")
    )
    fresh = load_sweep(sweep_paths)
    if fresh:
        unknown = set(fresh) - set(names)
        assert not unknown, unknown
        facts["fresh"] = fresh
        print(f"merged fresh scores for: {', '.join(n for n in names if n in fresh)}")
    else:
        print("no sweep log found — keeping facts.json fresh block as-is")

    blob = json.dumps(facts, separators=(",", ":"))
    html_path = HERE / "index.html"
    html = html_path.read_text()
    pat = re.compile(
        r'(<script id="mkgap-facts" type="application/json">).*?(</script>)', re.S
    )
    assert pat.search(html), "mkgap-facts blob not found in index.html"
    html = pat.sub(lambda m: m.group(1) + blob + m.group(2), html)
    html_path.write_text(html)
    print(f"inlined {len(blob)} bytes into index.html")


if __name__ == "__main__":
    main()
