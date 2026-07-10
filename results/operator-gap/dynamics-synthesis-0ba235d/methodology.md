# Hypothesis-evolution study: megakernel trial corpus

Goal (user, 2026-07-09): complete explanation of the underlying dynamics of what is and
is not efficient in the training megakernel; validate hypotheses by their ability to
predict PROMOTED vs NO-GO across the full trial history; time-slice the corpus, build
per-era hypothesis sets, track how they evolve, project the evolution forward, and
derive the future trial portfolio from that projection.

## Corpus (3.26MB)
- NOTES.md part1 (lines 1-3700): v1 -> P4 era (pre Jul-5). Agent: notes1
- NOTES.md part2 (3700-7444): P4 -> opgap transition. Agent: notes2
- ARCHIVE-20260707 part1/2/3 (1-5500/5500-11100/11100-end): Jul 5-7. Agents: a707-1/2/3
- ARCHIVE-20260708 part1/2 (1-4700/4700-end): Jul 8. Agents: a708-1/2
- Live board part1/2 (1-444/444-888): Jul 8 late - Jul 9. Agents: live1/2
- IMPROVEMENTS.md (all 2126 lines, ~223 entries): distilled ledger w/ stated laws. Agent: ledger

## Timeline anchors (sha -> date)
c5af0ff/f6b1f85 = Jul-5 matrix era; 94cb1ef/2a41f6a = Jul-6; a00c1dd/bb199ea = Jul-7 meter;
589ee3d = pdf executor Jul-7; 3d37664 = Jul-7 NT store + PDR nogo; 3f21e1e/6b137f5 = Jul-7 eve;
9550a7b/6b9218df = Jul-7 certs; 785d048/05e9e97 = Jul-8 burst + post-burst cert;
6d4e86d = Jul-8; 0ba235d = Jul-8/9 tip (fresh series qwen-l1 1.028x .. s8192 1.942x).

## Current certified state @0ba235d (fresh series, talk/facts.json)
qwen-l1 1.028, qwen-l2 1.109, deep 1.267 (floor), nano 1.376, s256 1.385, s128 1.416,
small 1.533, s1024 1.593, s2048 1.641, s3072 1.704, s4096 1.849, s8192 1.942 (frontier).
Absolute gaps (mk-base us): s8192 2996, s4096 1350, small 1020, qwen-l2 972, s3072 930,
s2048 672, deep 472, s1024 463, nano 239, s256 214, s128 204, qwen-l1 205.

## Residual exposure @05e9e97 (on-path us)
s8192: DQ wait 1751 + attn-fwd 1024 + CE 285 + lm-head fwd 250 + lm-head dX 253 + RMS dX 187
s4096: DKV wait 465 + attn-fwd 288 + CE 150 + lm-head fwd 137 + RMS dX 107 + SKR 78
s3072: DQ wait 278 + attn-fwd 184 + CE 115 + lm-head 106 + RMS dX 91
s2048: DQ wait 144 + attn-fwd 125 + lm-head 78 + CE 76
small: attn-fwd 219 + RMS dX 103 + lm-head fwd/dX 101/100 + DQ 54
deep: attn-fwd 191 + RMS dX 79 + lm-head 40+29 (mostly itax floor: 213 cp hops x ~3.5us)
s1024: attn-fwd 123 + lm-head fwd/dX 57/56 + RMS dX 43
qwen-l1: closed to 1.028 (NT supertile PDF route); l2 residual ~1.11 (head-dX pdfonly
candidate -27..-37us pending kind6 stack landing)

## Wave plan
Wave 1 (RUNNING): 10 extraction agents (above). Output: structured trial inventories.
Wave 2: sequential era-hypothesis agents. Era order: notes1 -> notes2 -> a707(1+2+3) ->
  a708(1+2) -> live(1+2). Each gets: prior era's law set + own era inventory. Must output:
  laws kept / refined (with what evidence forced the refinement) / overturned / added.
Wave 3: validation agents: each final law scored vs FULL inventory (write compiled
  inventory to scratchpad/inventory-full.md): confirming trials, counterexamples,
  precision, and "what future result would falsify/refine this law".
Wave 4 (me): convergence analysis (stable vs moving laws), projection (how law set
  shifts under candidate future trials), future-trial portfolio = info-gain on moving
  laws + EV on stable laws. Deliverables: results/operator-gap synthesis note (house
  style) + chat summary: (a) gaps + why, (b) stack rank, (c) dynamics/principles.

## Candidate laws pre-registered (scratchpad/candidate-laws.md): H1-H12
Wave 3 must score these too, not just wave-2 output.
