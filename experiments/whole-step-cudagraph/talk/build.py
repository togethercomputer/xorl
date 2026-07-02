#!/usr/bin/env python3
"""Inline facts.json into index.html's <script id="facts-data"> blob (idempotent).

Static measured facts (throughput / memory / regime), not a wandb export — same
data-inlining pattern as the reference talk sites. Run after editing facts.json:
    python build.py
If facts.json is missing, the blob is left as-is and the page's JS fallback (the
same numbers, hardcoded) carries it.
"""
import json
import pathlib
import re

HERE = pathlib.Path(__file__).parent
html = HERE / "index.html"
facts = HERE / "facts.json"
if not facts.exists():
    print("no facts.json — leaving blob untouched (JS fallback carries it)")
    raise SystemExit(0)

blob = json.dumps(json.loads(facts.read_text()), separators=(",", ":"))
text = html.read_text()
pat = re.compile(
    r'(<script id="facts-data" type="application/json">)(.*?)(</script>)', re.S
)
new, n = pat.subn(lambda m: m.group(1) + blob + m.group(3), text)
if n != 1:
    raise SystemExit(f"expected exactly one facts-data blob, found {n}")
html.write_text(new)
print(f"inlined {len(blob)} bytes into {html}")
