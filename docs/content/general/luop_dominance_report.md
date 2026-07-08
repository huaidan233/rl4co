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

## 4090 Stability Protocol

Target machine:

- Host: `100.111.43.33`
- User: `dell`

Suggested smoke command:

```bash
python run.py experiment=cityplan/dominance trainer.max_epochs=3 model.batch_size=256 model.train_data_size=1024 model.val_data_size=128 model.test_data_size=128 logger=csv
```

Required checks:

- Training loss stays finite.
- `train/dominance_reward` is logged and not all equal.
- `train/hv_contribution` is finite.
- Validation logs `val/pareto_hypervolume` and `val/pareto_front_size`.
- No LUOP action replay or infeasible action errors appear.

## 4090 Result

Pending remote run.
