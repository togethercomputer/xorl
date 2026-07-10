# qwen NT Sidecar Manual Dirty-Frontier Port @7abb355

Date: 2026-07-09 UTC

Status: PROMOTE-CANDIDATE. The manual port of the validated qwen NT sidecar/topembed stack onto the dirty-frontier qwen source passed route, rollback, policy-off, step parity, graph parity, and timing gates on both qwen4b-l1 and qwen4b-l2. Both the local GPU run and an independent remote isolated H100 run reproduced positive timings.

## Source

- Isolated source worktree: `/tmp/xorl-oss-qwen-nt-sidecar-manual-be55098-codex`
- remote-visible source worktree: `<repo-root>`
- Base dirty-frontier snapshot: `be55098`
- Commit: `7abb355` (`port qwen nt sidecar to dirty frontier`)
- Shared checkout source edits: none
- Worktree status after validation: clean in both source worktrees

Replay patch:

- `results/qwen-nt-sidecar-manual-dirty-7abb355-20260709T2145Z.patch`
- sha256: `2397027d2052b9dd2dacadaa66a8b0ae729f899b822f142ee312a8861e9b9405`
- size: `7240` lines / `261371` bytes

## Static And Route Gates

Static gates passed before GPU work:

- `git diff --check HEAD^..HEAD`
- `py_compile` for `mk.py`
- `py_compile` for `model.py`
- `py_compile` for all `results/qwen_nt*.py` helpers in the port worktree

Apply-check passed from clean `be55098` in `/tmp/xorl-oss-qwen-nt-sidecar-manual-applycheck-be55098-7abb355`:

- patch `--check`
- patch apply
- `py_compile`
- `git diff --check`
- host route contract

Host route contract:

- log: `results/qwen-nt-sidecar-manual-route-7abb355-20260709T2145Z.log`
- sha256: `ebcd3a8451a5275f0001b21eff973010c4780efee84df5132534fae00ca07bff`
- size: `11` lines / `17080` bytes

Route results:

- qwen4b-l1 default sidecar route: `n_instr=47`, `critical_path=26`, `gated=9`, one boundary row, one cutpoint, sidecar tile range `[0,4748]`, prefix/post `22/24`, valid split plan.
- qwen4b-l2 default sidecar route: `n_instr=78`, `critical_path=44`, `gated=14`, one boundary row, one cutpoint, sidecar tile range `[0,4748]`, prefix/post `37/40`, valid split plan.
- Forced-old rollback restores the direct target GEMM row and removes sidecar API/suffix.
- `MK_QWEN_NT_SIDECAR_STEP=0` keeps the boundary API available but sets `step_requested=false`.
- Small and prerequisite-off negative cases expose no sidecar metadata.

## Canonical Local GPU Validation

Command class:

- `CUDA_VISIBLE_DEVICES=<local-gpu>`
- `MKAB_TREE=/tmp/xorl-oss-qwen-nt-sidecar-manual-be55098-codex`
- private cache: `/tmp/torchext-qwen-nt-sidecar-manual-7abb355-20260709T2147Z`
- helper: `results/qwen_nt_sidecar_default_policy_0ba235d.py --device 0 --reps 12 --warmup 4`

Artifacts:

- log: `results/qwen-nt-sidecar-manual-defaultpolicy-localgpu-7abb355-20260709T2147Z.log`
- log sha256: `f5d8774952d29785335e804d2567a405af2adf8d16ff06109b1150a716b1f24c`
- log size: `45` lines / `48124` bytes
- summary: `results/qwen-nt-sidecar-manual-defaultpolicy-summary-7abb355-20260709T2147Z.json`
- summary sha256: `8f4c39a1e3d081888b930a56e26d89d20324fb1b6e6a49d66ab6a2f5f263c72a`
- summary size: `1303` lines / `37458` bytes

Summary:

- `pass=true`
- `sha=7abb355`
- cases: `4`
- qwen4b-l1 old_first: step median `7377.696us -> 6996.208us`, delta `-381.488us`, wins `12/12`; graph median `7405.184us -> 6988.480us`, delta `-416.704us`, wins `12/12`; worst step/graph grad rel `0.0015723270440251573`.
- qwen4b-l1 promoted_first: step median `7370.816us -> 7010.704us`, delta `-360.112us`, wins `12/12`; graph median `7404.848us -> 6965.120us`, delta `-439.728us`, wins `12/12`; worst step/graph grad rel `0.0015723270440251573`.
- qwen4b-l2 old_first: step median `9582.032us -> 9189.696us`, delta `-392.336us`, wins `12/12`; graph median `9583.312us -> 9141.328us`, delta `-441.984us`, wins `12/12`; worst step/graph grad rel `0.004807692307692308`.
- qwen4b-l2 promoted_first: step median `9613.248us -> 9258.480us`, delta `-354.768us`, wins `12/12`; graph median `9573.728us -> 9141.200us`, delta `-432.528us`, wins `12/12`; worst step/graph grad rel `0.004807692307692308`.

Parity and guard details:

- L1 step equivalence passed; loss diff <= `2.86102294921875e-06`.
- L2 step equivalence passed; loss diff <= `6.67572021484375e-06`.
- Step parity passed in all four order/shape cells.
- Graph capture passed for old and promoted in all four order/shape cells.
- Graph parity passed in all four order/shape cells.
- Policy-off guard passed for step and graph in all four order/shape cells.
- local GPU lock was released; final local GPU check was `0 MiB, 0%`.
- Other local GPU ordinals were not used.

## Independent remote isolated Replication

remote isolated job:

- job: `qwen-ntscdirty-7abb355`
- manifest: `results/remote isolated-qwen-nt-sidecar-dirty-7abb355-20260709T2147Z.yaml`
- manifest sha256: `250ed768bfa69ed53d7f418d0c28f83b358561dda756f6792f9e7e6a0e8abd20`
- manifest size: `139` lines / `5344` bytes
- remote worker: `<redacted-remote-node>`
- current remote isolated resources: none found for `app=qwen-ntscdirty`

Artifacts:

- log: `results/qwen-nt-sidecar-dirty-defaultpolicy-remote isolated-7abb355-20260709T2147Z.log`
- log sha256: `98b2e2270417ea7c361731b51c1430b3759ba6a3fc40416e00bffcf2e97608a1`
- log size: `72` lines / `67158` bytes
- routecheck log: `results/qwen-nt-sidecar-dirty-defaultpolicy-routecheck-7abb355-20260709T2147Z.log`
- routecheck sha256: `ebcd3a8451a5275f0001b21eff973010c4780efee84df5132534fae00ca07bff`
- routecheck size: `11` lines / `17080` bytes
- summary: `results/qwen-nt-sidecar-dirty-defaultpolicy-summary-7abb355-20260709T2147Z.json`
- summary sha256: `446e8cfc8340b4d598ce178133d7ad34ef4b4f88085035a194d96f1fc54bfebb`
- summary size: `1303` lines / `37427` bytes

remote isolated summary:

- `pass=true`
- `sha=7abb355`
- qwen4b-l1 old_first: step delta `-415.392us`, wins `12/12`; graph delta `-483.968us`, wins `12/12`.
- qwen4b-l1 promoted_first: step delta `-395.952us`, wins `12/12`; graph delta `-439.312us`, wins `12/12`.
- qwen4b-l2 old_first: step delta `-421.632us`, wins `12/12`; graph delta `-436.096us`, wins `12/12`.
- qwen4b-l2 promoted_first: step delta `-344.944us`, wins `12/12`; graph delta `-347.120us`, wins `12/12`.
- Worst step grad rel: L1 `0.0015723270440251573`, L2 `0.004807692307692308`.
- Worst graph grad rel: L1 `0.0031446540880503146`, L2 `0.004807692307692308`.

## Verdict

This is the first dirty-frontier qwen NT lane in this batch that is both behavior-clean and materially faster. It should be treated as a promotion candidate for the dirty-frontier qwen source:

- route contract is explicit and rollbackable;
- policy-off preserves an emergency runtime disable;
- step and graph parity pass for L1 and L2;
- both construction orders win every timing replicate;
- local and remote isolated H100 runs independently reproduce the result.

Do not confuse this with the earlier inline-only attempts. `6c556a5` and `cdaebe0` changed SASS but were cadence-neutral/call-only. The sidecar port changes the issue contract by splitting the NT target into a boundary row and sidecar execution path, and that is what produced the timing win.

Remaining before landing into a shared branch:

- rebase or replay `results/qwen-nt-sidecar-manual-dirty-7abb355-20260709T2145Z.patch` onto the intended integration branch;
- rerun the same helper after replay;
- run the broader qwen integrated/trainer smoke expected by the branch owner;
- keep rollback envs documented: `MK_GEMM_N256_NT_SUPERTILE_SIDECAR_BOUNDARY=0` and `MK_QWEN_NT_SIDECAR_STEP=0`.
