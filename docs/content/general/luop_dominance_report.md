# LUOP Dominance Model Report

## Summary

This work adds a new LUOP model, `LUOPDominanceAttentionModel`, under
`rl4co/models/zoo/luop_dominance/`. The original weighted LUOP model
`LUOPAttentionModel` under `rl4co/models/zoo/luop_am/` is preserved.

The new model keeps LUOP multi-action decoding by reusing the existing
type-parcel policy stack. Its train-time reward is not the weighted
multi-objective scalarization. Instead, each LUOP instance is expanded into a
small candidate group, and candidates are scored with Pareto rank, 2D
hypervolume contribution, and crowding distance.

## Literature Basis

The main literature basis uses high-quality multi-objective optimization, MORL,
and neural combinatorial optimization sources:

- Deb et al., NSGA-II, IEEE TEVC 2002, DOI `10.1109/4235.996017`.
- Zhang and Li, MOEA/D, IEEE TEVC 2007, DOI `10.1109/TEVC.2007.892759`.
- Zitzler and Thiele, Strength Pareto, IEEE TEVC 1999,
  DOI `10.1109/4235.797969`.
- Zitzler, Thiele, and Bader, set-based multiobjective optimization,
  IEEE TEVC 2010, DOI `10.1109/TEVC.2009.2016569`.
- Beume, Naujoks, and Emmerich, SMS-EMOA, EJOR 2007,
  DOI `10.1016/j.ejor.2006.08.008`.
- Roijers et al., multi-objective sequential decision-making survey, JAIR 2013,
  DOI `10.1613/jair.3987`.
- Van Moffaert and Nowe, Pareto Q-learning, JMLR 2014.
- Kyriakis et al., Pareto Policy Adaptation, ICLR 2022.
- Lin, Yang, and Zhang, PMOCO, ICLR 2022.
- Chen et al., NHDE, NeurIPS 2023.
- Kool, van Hoof, and Welling, Attention Model, ICLR 2019.
- Kwon et al., POMO, NeurIPS 2020.
- Williams, REINFORCE, Machine Learning 1992, DOI `10.1007/BF00992696`.
- Sutton et al., policy-gradient theorem, NeurIPS 1999.

Application-only and low-impact papers are not used as primary support.

## Reward

For objective components `C` with shape `[batch, candidates, 2]`, objectives are
maximized. The model computes:

```text
dominance_reward =
    rank_reward_scale * rank_score
  + hv_reward_scale * normalized_hv_contribution
  + crowding_reward_scale * normalized_crowding
  - dominated_penalty * is_dominated
```

The default config uses:

- `num_dominance_candidates: 4`
- `rank_reward_scale: 1.0`
- `hv_reward_scale: 0.25`
- `crowding_reward_scale: 0.05`
- `dominated_penalty: 0.25`

The original environment still computes and exposes `reward_components` and the
legacy scalarized reward. The dominance model stores the legacy scalarized value
as `scalarized_reward`, then replaces train-time `reward` with
`dominance_reward` before calling the REINFORCE loss.

## Difference from Weighted LUOP AM

`LUOPAttentionModel`:

- Uses the environment scalar reward.
- The scalar reward is produced from `reward_components` and
  `objective_weights`.
- Remains available as `configs/model/luop_am.yaml`.

`LUOPDominanceAttentionModel`:

- Reuses the same LUOP policy and multi-action decoder.
- Expands each training instance into a candidate group.
- Scores candidate groups with Pareto rank, hypervolume contribution, and
  crowding distance.
- Uses `shared` baseline by default, matching the group reward structure.
- Is available as `configs/model/luop_dominance.yaml`.

## Local Validation

Run on the local Windows environment in `D:\Lab\ProjectRL4CO2\rl4co_fork_sync`:

```powershell
python -m pytest tests/test_luop_dominance.py -q
```

Result:

```text
8 passed
```

Targeted regression:

```powershell
python -m pytest tests/test_luop_dominance.py tests/test_luop_joint_multiobjective.py::test_luop_attention_model_has_dedicated_zoo_entrypoint tests/test_luop_joint_multiobjective.py::test_generic_attention_policy_stays_luop_module_free tests/test_luop_joint_multiobjective.py::test_generic_reinforce_stays_luop_metric_free -q
```

Result:

```text
11 passed
```

Minimal Hydra smoke:

```powershell
python run.py experiment=cityplan/dominance trainer.max_epochs=1 model.batch_size=2 model.val_batch_size=2 model.test_batch_size=2 model.train_data_size=4 model.val_data_size=2 model.test_data_size=2 model.generate_default_data=false env.generator_params.num_loc=4 env.generator_params.num_fixed=0 'env.min_type_ratios=[0,0,0,0,0,0,0,0]' env.val_file=null env.test_file=null model.num_dominance_candidates=2 +model.policy_kwargs.embed_dim=32 +model.policy_kwargs.num_encoder_layers=1 +model.policy_kwargs.num_heads=4 +model.policy_kwargs.feedforward_hidden=64 logger=csv test=false callbacks.rich_progress_bar=null +trainer.num_sanity_val_steps=0
```

Result: one train epoch and one validation pass completed with finite
`train/loss`, `train/dominance_reward`, `val/pareto_hypervolume`, and
`val/pareto_front_size`.

## 4090 Stability Protocol

Target machine:

- Host: `100.111.43.33`
- User: `dell`

Suggested smoke command:

```bash
python run.py experiment=cityplan/dominance trainer.max_epochs=3 trainer.accelerator=gpu +trainer.devices=1 trainer.precision=16-mixed model.batch_size=256 model.val_batch_size=256 model.test_batch_size=256 model.train_data_size=1024 model.val_data_size=128 model.test_data_size=128 model.generate_default_data=false env.val_file=null env.test_file=null callbacks.rich_progress_bar=null callbacks.learning_rate_monitor=null +trainer.log_every_n_steps=1 logger=csv
```

Required checks:

- Training loss stays finite.
- `train/dominance_reward` is logged and not all equal.
- `train/hv_contribution` is finite.
- Validation logs `val/pareto_hypervolume` and `val/pareto_front_size`.
- No LUOP action replay or infeasible action errors appear.

## 4090 Result

Remote smoke completed on `100.111.43.33`:

- Commit: `effbb070ee8222be6cab79bf3e845b3d1f77f7c8`.
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB.
- Python/Torch/CUDA: Python 3.10.20, Torch 2.5.1, CUDA 12.1.
- Lightning: 2.4.0.
- Command:

```bash
python scripts/luop_dominance_4090.py +trainer.num_sanity_val_steps=0
```

Metrics file:
`logs/train/runs/2026-07-08_20-25-16/csv/version_0/metrics.csv`.

Observed metrics:

- `train/loss`: final `-0.632765`, range `[-0.842533, -0.166422]`.
- `train/dominance_reward`: final `0.794702`, range `[0.746968, 0.794702]`.
- `train/hv_contribution`: final `0.006853`, range `[0.006212, 0.007703]`.
- `val/pareto_hypervolume`: final `0.251371`, range `[0.176043, 0.283247]`.
- `val/pareto_front_size`: final `2.335938`, range `[1.843750, 2.335938]`.
- `val/checkpoint_score`: final `0.351607`, range `[0.177777, 0.396943]`.

The run exited with code 0. `scripts/luop_dominance_4090.py --metrics` confirmed
finite dominance and Pareto metrics, and `train/dominance_reward` did not
collapse to a constant.
