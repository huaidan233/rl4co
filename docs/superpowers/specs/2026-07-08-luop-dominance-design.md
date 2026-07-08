# LUOP Dominance Model Design

## Goal

Add a new LUOP model under `rl4co/models/zoo/` that keeps the existing
weighted `luop_am` model intact, reuses the LUOP multi-action policy, and trains
with a dominance-style reward instead of the weighted multi-objective scalar
reward.

## User Requirements

- Keep the existing weighted LUOP model and behavior.
- Do not make large model changes inside `rl4co/models/zoo/am`.
- Add a new zoo model directory.
- Support LUOP multi-action decoding: joint type-parcel actions and optional
  type-first decoding.
- Use a Pareto/dominance training signal similar to nondominated sorting, not
  `reward_components @ objective_weights`, as the main training reward.
- Use high-quality literature only for the main report.
- Write records and reports, test locally, then tune training stability on the
  4090 machine.
- Commit changes frequently.

## Architecture

The new model will live in `rl4co/models/zoo/luop_dominance/`. It will reuse
`LUOPAttentionModelPolicy`, `LUOPConstructivePolicy`, and
`LUOPAttentionModelDecoder` from `luop_am`, so the multi-action action space is
not duplicated or patched into generic AM.

The existing `luop_am` remains the weighted baseline. The new model
`LUOPDominanceAttentionModel` subclasses the LUOP model behavior where useful
but overrides train-time reward construction. During training, each original
LUOP instance is expanded into a group of `K` candidate rollouts. Candidate
solutions from the same instance are compared in objective space using
nondominated sorting, 2D hypervolume contribution, and crowding distance. The
resulting scalar dominance reward is passed to the normal REINFORCE loss.

Validation and test still report regular scalar reward, compatibility,
accessibility, Pareto hypervolume, and front size so existing reports remain
comparable. Weighted scalarization may remain available for evaluation,
conditioning, and legacy metrics, but it is not the main train reward of the new
model.

## Reward Design

For each group of candidates with objective components `C` shaped
`[batch, candidates, objectives]`, where objectives are maximized:

1. Compute nondominated rank within each instance group.
2. Convert rank into a bounded score, with rank 0 as the strongest front.
3. Compute 2D hypervolume contribution for each candidate.
4. Compute crowding distance inside each rank front as a diversity tie-breaker.
5. Combine the terms:

```text
dominance_reward =
    rank_reward_scale * rank_score
  + hv_reward_scale * normalized_hv_contribution
  + crowding_reward_scale * normalized_crowding
  - dominated_penalty * is_dominated
```

The default first implementation will use a small group size, such as 4, so the
extra sampling cost is controlled. All terms are detached scalar rewards for
policy-gradient training; no differentiable sorting is required.

## Literature Basis

Main report citations should use high-level sources:

- Deb et al., NSGA-II, IEEE TEVC 2002, DOI `10.1109/4235.996017`.
- Zhang and Li, MOEA/D, IEEE TEVC 2007, DOI `10.1109/TEVC.2007.892759`.
- Zitzler and Thiele, Strength Pareto and coverage, IEEE TEVC 1999,
  DOI `10.1109/4235.797969`.
- Zitzler, Thiele, and Bader, Set-based multiobjective optimization,
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

Application-only papers and low-impact venues should not be used as main
support. They can be mentioned only as implementation analogies if needed.

## Files

Create:

- `rl4co/models/zoo/luop_dominance/__init__.py`
- `rl4co/models/zoo/luop_dominance/model.py`
- `rl4co/models/zoo/luop_dominance/rewards.py`
- `configs/model/luop_dominance.yaml`
- `configs/experiment/cityplan/dominance.yaml`
- `tests/test_luop_dominance.py`
- `docs/content/general/luop_dominance_report.md`
- `scripts/luop_dominance_4090.py`

Modify:

- `rl4co/models/__init__.py`
- `rl4co/models/zoo/__init__.py`
- `.gitignore` to ignore `.ace-tool/`

Do not modify:

- `rl4co/models/zoo/am/*`
- Existing weighted LUOP model semantics in `rl4co/models/zoo/luop_am/*`

## Testing

Local tests must cover:

- Nondominated ranks for batched objective tensors.
- Crowding distance behavior on rank fronts.
- 2D hypervolume contribution behavior.
- Dominance reward finite values and shape preservation.
- New model train step returns finite loss and dominance metrics.
- Multi-action outputs still include valid flat, type, and parcel actions.
- Existing `luop_am` weighted train smoke test still passes.
- Generic AM remains LUOP-free.

4090 stability checks must record:

- Train loss finite over smoke epochs.
- Dominance reward not collapsed to all equal values.
- Pareto hypervolume and front size logged on validation.
- No infeasible LUOP action traces.

## Acceptance Criteria

- A new `LUOPDominanceAttentionModel` is importable from `rl4co.models`.
- Hydra can instantiate the new model config.
- Training uses dominance reward in the new model.
- Existing `LUOPAttentionModel` weighted path still works.
- No large edits are made to `zoo/am`.
- Local tests pass.
- A 4090 run record and report are written.
- Commits are pushed to the fork.
