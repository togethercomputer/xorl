#!/usr/bin/env python3
"""Build label-doctored traces: replace decision K's token with an anchor id.

Decision K's producing forward consumes positions < prompt+K, all unchanged,
so the doctored replay's trainer logprob AT decision K is the trainer's
production logprob of the anchor token under the identical forward. Later
decisions are garbage and ignored.
"""

from __future__ import annotations

import argparse
import copy
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--decision", type=int, default=39)
    parser.add_argument("--token", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trace = json.load(open(args.trace))
    doctored = copy.deepcopy(trace)
    for cap in doctored["captures"]:
        cap["output_ids"][args.decision] = args.token
        prompt_len = len(cap["prompt_ids"])
        cap["full_ids"][prompt_len + args.decision] = args.token
    doctored["label"] = f"{trace['label']}-doctored-d{args.decision}-t{args.token}"
    json.dump(doctored, open(args.output, "w"), indent=1)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
