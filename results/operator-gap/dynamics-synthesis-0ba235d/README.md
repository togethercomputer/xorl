# Evidence package for dynamics-synthesis-0ba235d.md

## Scope and honest claim

This is a HISTORICAL evidence package frozen against analysis head `0ba235d`
(2026-07-09), packaged at checkout `3b5701e`, with the row-level corpus extraction
completed later the same day (corpus state at extraction time: live board 1918 lines,
IMPROVEMENTS.md 2232 lines — both had grown past the analysis snapshot; the archives
and NOTES.md are static).

**`trials_full.csv` is the complete row-level record**: 2,313 rows, one per
verdict-bearing entry across the five source files, each with `source_file` +
`source_line` (verified by spot-check against the sources). Extracted by 11 agents over
exact line ranges; the raw per-range shards with their coverage reports are kept in
`trials_full/`. Duplication semantics: the same experiment legitimately appears in
multiple source files (a board CLOSE, an IMPROVEMENTS ledger entry, and a NOTES record
are three distinct evidence records of one trial), and double-posted board closes by
concurrent sessions are marked `verdict=DUPLICATE`. Verdict-class distribution:
643 NO-GO, 527 PROMOTED, 401 MEASUREMENT, 102 CLOSED-BY-ANALYSIS, 91 SASS-PROOF,
83 NEUTRAL, 69 VALIDATED-PARKED, 47 FLIP-PROMOTED, 23 REFUTED, 21 NO-GO-CORRECTNESS,
plus NOTE/SCAFFOLD/LAW/ROUTE-AWAY/SUPERSEDED/INVALID/WITHDRAWN/ACTIVE/PARKED tails.

`trial_ledger.csv` (126 rows) remains the CURATED, deduplicated view that
`law_trial_links.csv` (177 links) points at; `evidence_events.csv` (32 rows) holds the
curated non-terminal evidence. Precision figures in `laws.csv` are the adversarial
validation reports' claims (`validation/*.md`); the mechanical audit path is the join
`laws.csv <- law_trial_links.csv -> {trial_ledger,evidence_events}.csv`, and any deeper
question can now be resolved against `trials_full.csv` and its source lines directly.

## Files

- `laws.csv` — 39 laws (37 original, C1/C2 split out of the old P5 on 2026-07-09):
  id, domain, name, status, precision, predicts_n, true_counterexamples, refinement,
  info_gain_probe.
- `trial_ledger.csv` — one row per verdict-bearing trial, terminal verdict normalized
  (PROMOTED / NO-GO / REFUTED / NO-GO-CORRECTNESS), with landed status and source file.
  Split-by-shape outcomes are separate rows (e.g. T054/T055).
- `evidence_events.csv` — non-terminal evidence: measurements, standalone probes,
  SASS-only proofs, scaffolds, law declarations, retractions, floor declarations,
  validated-but-parked candidates.
- `law_trial_links.csv` — law_id, ref_id (T### or E###), role (confirm | counterexample
  | boundary | probe), expected vs observed verdict, source_file, source_line
  (line numbers filled only where line-accurate; blank otherwise — the inventories were
  extracted over named line RANGES of the source corpus, recorded in each inventory
  header).
- `inventory/` — the 8 era-slice extraction reports. These are records of what each era
  slice contained AND believed; they are deliberately not retro-edited. Known
  superseded statement: inventory/08's "ANY WGMMA body reachable via function call =>
  singleton drain" is refined by C1/C2 in laws.csv (attention drains are
  body-structural/toolchain — see supersession map).
- `validation/` — the 4 per-domain adversarial validation reports (the source of the
  precision claims; cite counts and strongest examples, not full row sets).
- `laws-evolution.md`, `known-evolution-episodes.md` — era-by-era belief trajectory.
- `candidate-laws.md` — pre-registered H1-H12 (written before corpus contact); scoring
  in `validation/validation-V4-epistemics.md`.
- `trials_full.csv` — the complete row-level corpus record (2,313 rows with source
  line numbers; see honest claim above). Schema: source_file, source_line, date,
  category, trial, shape, verdict, delta, note (internal commas replaced by `;`).
- `trials_full/` — the 11 raw extraction shards with per-shard coverage ranges and the
  extractors' recorded judgment calls (in the session record).
- `methodology.md` — the original wave plan. Its promised full-corpus row table is now
  realized as `trials_full.csv` (added in the follow-up pass); the `inventory/`
  directory remains the era-summary view.

## Supersession map (as of packaging, checkout 3b5701e)

| This package says | Superseded / resolved by |
|---|---|
| Portfolio ranking (note section 4) | `results/operator-gap/frontier-20260709-postsession.md` |
| Image partitioning "IN MOTION" for attention | REFUTED: `attn-splitentry-sass-0ba235d-refuted.md` (CALL=0 still {1:12}); mechanism: `attn-hgmma-drain-mechanism-0ba235d.md` (ptxas 13.1 descriptor-slot; toolkit-bump gate) |
| Band-preserving one-pass probe (item 2 frame) | CLOSED: `onepass-band-pdf-audit-0ba235d.md` |
| CE pair op-body "open lane" | CE_FWD warprow LANDED `3b5701e` (-105us s8192); CE_BWD floor-closed |
| SwiGLU fold "in flight" | PROMOTED nano+deep+small: `dagdepth-swbf-0ba235d.md`; composed cert @`e91f729` (deep 1.234 new floor) |
| qwen sidecar "contract in hand" | launch ABI + split plan proven, runnable_now=false: `qwen-nt-sidecar-{launchabi,splitplan}-0ba235d.md` |
| Precision lattice item 7 | executed: `precision-lattice-s2048-0ba235d.md` + payload v2.1 (~-235us / 5 shapes) |
| Gap table (@05e9e97/6d4e86d rows) | post-payload/post-landing residuals in the Jul-9 profile refreshes and `frontier-20260709-postsession.md` |

## Provenance of the gap table in the parent note

Non-qwen rows: certified @`05e9e97` (`postburst-scoreboard-05e9e97.md`,
`talk-fresh-sweep-jul8-postburst-20260708T1020Z.log`). Qwen rows: @`6d4e86d`
(`qwen-postntpdf-score-profile-6d4e86d.md`). Adopted into the `0ba235d` fresh series
(`talk/facts.json`).
