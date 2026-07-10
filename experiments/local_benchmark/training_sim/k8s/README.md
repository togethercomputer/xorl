# Simulator Calibration K8s Notes

Calibration jobs should be normal XoRL training benchmark jobs with these trainer flags enabled:

```yaml
train:
  enable_step_phase_timing: true
  enable_per_component_timing: true
  enable_step_memory_profiling: true
```

Any pod requesting GPUs on the research-common-h100 cluster must set `team: turbo` on the pod template labels.
Keep the run short, preserve the trainer-head log, and feed that log to `collect_calibration.py`.
