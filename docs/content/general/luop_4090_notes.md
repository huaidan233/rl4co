# LUOP 4090 Stability Notes

## GPU Memory Guardrail

Default to the RTX 4090 for LUOP validation unless there is a concrete reason
not to. Use the largest already-verified batch for routine probes:

- LUOP50: `batch_size=2048`
- LUOP100: `batch_size=1024`
- LUOP200: `batch_size=256`

Treat LUOP200 as the special memory-bound case. Do not raise it above `256`
unless the run is an explicit guarded memory probe.

The LUOP200 stability probe must not use `batch_size=512` on the RTX 4090 host. A real run reached about `24073 / 24564 MiB` device memory, and with shared memory pressure this is effectively a 36 GB-scale allocation path that can exhaust the machine.

Use `batch_size=256` when resuming LUOP200 validation on the 4090:

```bash
python scripts/luop_4090_stability.py --suite --log-dir logs/luop_4090_stability_resume
```

The standard suite now emits `LUOP200 batch 256->128`. Do not resume or relaunch stale `LUOP200 batch=512` commands from old logs.

Measured RTX 4090 probes:

- `LUOP200 batch=128`: stable, peak `10692 MiB`.
- `LUOP200 batch=256`: stable, peak `20127 MiB`.
- `LUOP200 batch=512`: unsafe, reached about `24073 MiB` device memory and can exhaust shared memory.

Guarded suite scan on 2026-07-01 passed with no stability failures and no runtime memory-cap hits:

- `LUOP50 batch=2048`, linear scalarization: peak `12280 MiB`.
- `LUOP100 batch=1024`, linear scalarization: peak `20852 MiB`.
- `LUOP200 batch=256`, linear scalarization: peak `20128 MiB`.
- `LUOP50 batch=2048`, Chebyshev scalarization: peak `12265 MiB`.

The verified remote log directory is `logs/luop_4090_stability_guarded_suite_20260701`.
After the scan, no matching LUOP training process was running and the 4090 was idle at about `430 MiB` used.

The runner builds commands with `sys.executable`, not a hard-coded `python`, so
the remote `.venv4090/bin/python` environment works even when `python` is not on
the shell `PATH`.

The runner rejects `LUOP200 batch_size > 256` before launch and terminates any probe that samples more than `22000 MiB` GPU memory. If that runtime cap is hit, the adaptive runner backs off to the next smaller batch. For a cautious memory check, run a single probe rather than the full suite:

```bash
python scripts/luop_4090_stability.py --num-loc 200 --target-batch 256 --min-batch 128 --log-dir logs/luop_4090_memory_probe
```

## Evaluation Stability Guardrail

Do not treat the previous normalized 5-epoch curve as stable:

```text
val/reward: 0.5365 -> 0.5114 -> 0.5883 -> 0.5703 -> 0.5294
```

That run was GPU-safe, but the metric still bounced enough that it should not be used as evidence of training stability. The likely cause was evaluation-time objective-weight noise: `sample_objective_weights=true` used to materialize random `objective_weights` for train, validation, and test when generated datasets had no saved weights.

Current behavior:

- Training still samples objective weights when `sample_objective_weights=true`.
- Validation and test use fixed `eval_objective_weights` by default, currently `[0.5, 0.5]`.
- Set `sample_eval_objective_weights=true` only for an explicit noisy-evaluation experiment.
- Cityplan AM checkpoints now monitor `val/checkpoint_score`, the mean of scalar reward, accessibility reward, and Pareto hypervolume, instead of scalar `val/reward` or pure `val/pareto_hypervolume`.
- The 4090 runner now fails a run when `val/reward`, `val/compatibility_reward`, `val/accessibility_reward`, `val/pareto_hypervolume`, or `val/checkpoint_score` has at least four validation points and shows a post-best drop or oscillation above `0.05`.
- Full (`--full`) stability runs require at least four finite validation points by default; smoke probes keep this gate disabled unless `--min-validation-points` is supplied.
- When a run directory is scanned, the runner also checks `.hydra/config.yaml` and rejects artifacts that did not actually use fixed eval weights or `val/checkpoint_score` checkpointing.
- Runner-launched probes set `log_every_n_steps=1` so `metrics.csv` contains train loss/reward signals even for short smoke runs.
- The rollout baseline now keeps one fixed challenge dataset when it accepts a stronger policy. Previously it regenerated that dataset on baseline updates, so the t-test and the new baseline mean could be measured on different generated LUOP instances.
- Rollout warmup now keeps the exponential/mixed baseline path until `alpha == 1`. Previously `n_epochs > 1` still wrapped training datasets with pure rollout `extra` as soon as `alpha > 0`, so partial warmup was bypassed in `calculate_loss`.

Current CPU diagnostics still do not prove training stability. On LUOP50, scalar `val/reward` drifted down mostly because `val/accessibility_reward` collapsed while compatibility stayed flat or improved. Even when Pareto hypervolume rose, that component-level regression is not acceptable stability evidence and must be treated as a failed probe until a longer guarded run shows both scalar and component curves are controlled.

## Rollout Baseline Train-Mode Fix

The LUOP training instability had a real baseline-state bug: `RolloutBaseline.rollout()` switched the live candidate policy to eval mode during epoch-end baseline challenges and did not restore its previous training mode. This could leave subsequent training epochs running with eval-mode encoder behavior.

The rollout path now restores the policy mode in a `finally` block, while the copied baseline policy remains in eval mode after baseline updates. The regression test is:

```bash
python -m pytest tests/test_luop_joint_multiobjective.py::test_luop_rollout_baseline_rollout_restores_policy_training_mode -q
```

Local RTX 4060 LUOP50 diagnostics after the fix, with fixed train/eval weights, `batch_size=32`, `val_data_size=512`, `lr=5e-5`, and `decode_type_first=false`, are materially smoother than the earlier run:

```text
val/reward: 0.4427 -> 0.4974 -> 0.5210 -> 0.5244 -> 0.5271 -> 0.5350 -> 0.5323 -> 0.5345 -> 0.5335 -> 0.5341 -> 0.5330 -> 0.5345 -> 0.5375 -> 0.5349 -> 0.5488 -> 0.5513
val/accessibility_reward: 0.2059 -> 0.2767 -> 0.3032 -> 0.3081 -> 0.3160 -> 0.3213 -> 0.3322 -> 0.3344 -> 0.3351 -> 0.3338 -> 0.3296 -> 0.3306 -> 0.3376 -> 0.3375 -> 0.3558 -> 0.3595
val/pareto_hypervolume: 0.2559 -> 0.2286 -> 0.2481 -> 0.2497 -> 0.2619 -> 0.2802 -> 0.2757 -> 0.2832 -> 0.2879 -> 0.2777 -> 0.2729 -> 0.2640 -> 0.2652 -> 0.2648 -> 0.2788 -> 0.2799
```

The same stability scanner no longer flags this 16-point local diagnostic, but this is still a local LUOP50 probe, not final LUOP200 evidence. Before resuming the remote 4090 run, keep using `batch_size=256` for LUOP200 and run the guarded scanner on the produced artifacts.

After moving LUOP-specific metrics out of generic `REINFORCE`, a local RTX 4060 smoke run verified that real training artifacts include `val/checkpoint_score` and that `callbacks.model_checkpoint.monitor=val/checkpoint_score` is honored:

```text
logs/luop50_checkpoint_score_smoke_20260703
val/checkpoint_score: 0.1506 -> 0.2617
```

This smoke run only verifies the metric path and scanner compatibility; it is too short to prove training stability.

An attempted 8-epoch local RTX 4060 LUOP50 diagnostic with `batch_size=16`, `train_data_size=256`, and `val_data_size=256` was stopped by the local command timeout after six completed validation points. The partial artifact still passed the stability scanner with `min_validation_points=4`:

```text
logs/luop50_checkpointscore_diag_8ep_20260703
val/reward: 0.4264 -> 0.4722 -> 0.5054 -> 0.5135 -> 0.5125 -> 0.5286
val/accessibility_reward: 0.1692 -> 0.2680 -> 0.2860 -> 0.2893 -> 0.2900 -> 0.3246
val/pareto_hypervolume: 0.2346 -> 0.2118 -> 0.2288 -> 0.2323 -> 0.2268 -> 0.2453
val/checkpoint_score: 0.2767 -> 0.3173 -> 0.3401 -> 0.3450 -> 0.3431 -> 0.3662
```

This is useful local evidence that the current checkpoint-score path does not reintroduce the previous early accessibility collapse, but it is still not a substitute for a full guarded LUOP200 run on the 4090.

To get a complete local 8-epoch run within the local tool timeout, a lighter RTX 4060 LUOP50 diagnostic used `batch_size=16`, `train_data_size=128`, and `val_data_size=128`. It completed all eight validation points and passed the stability scanner with `min_validation_points=8`:

```text
logs/luop50_checkpointscore_diag_val128_8ep_20260703
val/reward: 0.4247 -> 0.4894 -> 0.4999 -> 0.4940 -> 0.4866 -> 0.4895 -> 0.4979 -> 0.4996
val/accessibility_reward: 0.1388 -> 0.2621 -> 0.2994 -> 0.2731 -> 0.2589 -> 0.2719 -> 0.2786 -> 0.2811
val/pareto_hypervolume: 0.2157 -> 0.2364 -> 0.2404 -> 0.2261 -> 0.2162 -> 0.2186 -> 0.2277 -> 0.2264
val/checkpoint_score: 0.2597 -> 0.3293 -> 0.3466 -> 0.3311 -> 0.3206 -> 0.3267 -> 0.3347 -> 0.3357
```

The best local checkpoint for this run was `epoch_002.ckpt`, selected by `val/checkpoint_score`. The mild mid-run dip stayed below the scanner threshold and recovered by epoch 7.

The same lightweight diagnostic with `seed=4321` exposed an important scanner edge case rather than a clear model collapse:

```text
val/reward: 0.4627 -> 0.4956 -> 0.4892 -> 0.4838 -> 0.4840 -> 0.4982 -> 0.5160 -> 0.5304
val/compatibility_reward: 0.7547 -> 0.7456 -> 0.7475 -> 0.7481 -> 0.7401 -> 0.7320 -> 0.7326 -> 0.7157
val/accessibility_reward: 0.1707 -> 0.2456 -> 0.2308 -> 0.2195 -> 0.2278 -> 0.2643 -> 0.2994 -> 0.3450
val/pareto_hypervolume: 0.1854 -> 0.2141 -> 0.2291 -> 0.2648 -> 0.2783 -> 0.2825 -> 0.2863 -> 0.2900
val/checkpoint_score: 0.2730 -> 0.3184 -> 0.3164 -> 0.3227 -> 0.3300 -> 0.3483 -> 0.3672 -> 0.3885
```

Compatibility declined by about `0.039`, but accessibility improved by about `0.174`, scalar reward improved, Pareto hypervolume improved, checkpoint score improved, and the component gap narrowed from about `0.584` to `0.371`. Treat this as equal-weight objective rebalancing, not as the same failure mode as an uncompensated component collapse. The scanner now permits this narrow case only when scalar reward, Pareto hypervolume, and checkpoint score are themselves stable/improving and the declining component stays within the normal `0.05` swing budget. It still rejects uncompensated component drops and oscillatory curves.

For the next resumed/full LUOP200 run, keep `batch_size=256`, fixed eval weights, and the 22 GB runtime cap. Do not resume the old run as evidence of stability; start or resume only after confirming these overrides are present in the command/config.

## 2026-07-03 RTX 4090 Evidence

Current LUOP-specific architecture is isolated under
`rl4co/models/zoo/luop_am/`; generic `zoo/am` is not carrying LUOP behavior.

Focused local tests passed before remote execution:

```bash
python -m pytest tests/test_luop_joint_multiobjective.py tests/test_luop_4090_runner.py -q
```

The 4090 smoke suite passed with no scanner failures:

- LUOP50, `batch_size=2048`: peak `8169/8277 MiB`, scan `[]`
- LUOP100, `batch_size=1024`: peak `12521 MiB`, scan `[]`
- LUOP200, `batch_size=256`: peak `11873 MiB`, scan `[]`
- LUOP50 Chebyshev, `batch_size=2048`: peak `8169 MiB`, scan `[]`

Eight-epoch 4090 diagnostics also passed:

```text
logs/luop50_4090_bs2048_8ep_20260703
val/reward: 0.4876 -> 0.5880
val/checkpoint_score: 0.3322 -> 0.4439

logs/luop100_4090_bs1024_8ep_20260703
val/reward: 0.5178 -> 0.6146
val/checkpoint_score: 0.3372 -> 0.4642

logs/luop200_4090_bs256_8ep_20260703
val/reward: 0.5405 -> 0.6447
val/checkpoint_score: 0.3567 -> 0.4806
```

A longer LUOP200 full validation on the 4090 completed with status `0` and
scanner result `[]`:

```text
logs/luop200_4090_bs256_20ep_full_20260703
batch_size=256, train_data_size=8192, eval_data_size=2048
gpu_peak_memory_mib=11795
best checkpoint: epoch_019.ckpt

val/reward:
0.5543 -> 0.6172 -> 0.6504 -> 0.6443 -> 0.6614 -> 0.6752 -> 0.6761 -> 0.6937 -> 0.7000 -> 0.7131 -> 0.7138 -> 0.7162 -> 0.7312 -> 0.7317 -> 0.7254 -> 0.7350 -> 0.7199 -> 0.7257 -> 0.7355 -> 0.7418

val/checkpoint_score:
0.3653 -> 0.4410 -> 0.4870 -> 0.4784 -> 0.5028 -> 0.5293 -> 0.5253 -> 0.5533 -> 0.5601 -> 0.5794 -> 0.5818 -> 0.5843 -> 0.6120 -> 0.6105 -> 0.5998 -> 0.6142 -> 0.5903 -> 0.5970 -> 0.6137 -> 0.6233
```

The final LUOP200 test metrics were:

```text
test/reward: 0.7406
test/compatibility_reward: 0.8853
test/accessibility_reward: 0.5960
test/pareto_hypervolume: 0.5281
test/checkpoint_score: 0.6216
```
