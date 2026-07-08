# LUOP Dominance Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new LUOP dominance model in `zoo` that supports multi-action decoding and trains with group-level Pareto dominance rewards while preserving the existing weighted `luop_am` model.

**Architecture:** The new zoo module reuses LUOP multi-action policy and decoder classes from `luop_am`. Only the new `luop_dominance` model changes train-time reward construction by expanding each input instance into a small candidate group and applying nondominated sorting, hypervolume contribution, and crowding distance to `reward_components`.

**Tech Stack:** Python, PyTorch, TensorDict, Lightning, Hydra configs, pytest.

---

## File Structure

- Create `rl4co/models/zoo/luop_dominance/rewards.py`
  - Owns nondominated rank, crowding distance, 2D hypervolume contribution, and combined dominance reward.
- Create `rl4co/models/zoo/luop_dominance/model.py`
  - Defines `LUOPDominanceAttentionModel`, reusing `LUOPAttentionModelPolicy`.
- Create `rl4co/models/zoo/luop_dominance/__init__.py`
  - Exports the new model and reward helpers.
- Modify `rl4co/models/__init__.py` and `rl4co/models/zoo/__init__.py`
  - Register the new zoo model without touching generic AM.
- Create `configs/model/luop_dominance.yaml`
  - Hydra model target.
- Create `configs/experiment/cityplan/dominance.yaml`
  - Cityplan experiment using the new model.
- Create `tests/test_luop_dominance.py`
  - Reward utility tests and model smoke tests.
- Create `docs/content/general/luop_dominance_report.md`
  - Literature-backed report and experiment log.
- Create `scripts/luop_dominance_4090.py`
  - Remote training command generator and log checker.

---

### Task 1: Reward Utility Tests

**Files:**
- Create: `tests/test_luop_dominance.py`

- [ ] **Step 1: Write failing tests for reward helpers**

Add tests that import these not-yet-existing functions:

```python
import torch

from rl4co.models.zoo.luop_dominance.rewards import (
    crowding_distance,
    dominance_reward,
    hypervolume_contribution_2d,
    nondominated_rank,
)


def test_luop_dominance_rank_identifies_batched_fronts():
    points = torch.tensor(
        [
            [[0.9, 0.2], [0.7, 0.7], [0.2, 0.9], [0.4, 0.4]],
            [[0.1, 0.1], [0.2, 0.2], [0.3, 0.1], [0.1, 0.3]],
        ]
    )

    ranks = nondominated_rank(points)

    assert ranks.tolist() == [[0, 0, 0, 1], [2, 0, 1, 1]]


def test_luop_hypervolume_contribution_rewards_unique_contributors():
    points = torch.tensor([[[0.8, 0.2], [0.6, 0.6], [0.2, 0.8], [0.3, 0.3]]])

    contribution = hypervolume_contribution_2d(points, reference=torch.tensor([0.0, 0.0]))

    assert contribution.shape == (1, 4)
    assert torch.all(contribution[..., :3] > 0)
    assert contribution[..., 3].item() == 0


def test_luop_crowding_distance_prefers_front_boundaries():
    points = torch.tensor([[[0.8, 0.2], [0.6, 0.6], [0.2, 0.8], [0.3, 0.3]]])
    ranks = nondominated_rank(points)

    distance = crowding_distance(points, ranks)

    assert distance.shape == (1, 4)
    assert torch.isfinite(distance).all()
    assert distance[0, 0] > distance[0, 1]
    assert distance[0, 2] > distance[0, 1]


def test_luop_dominance_reward_is_finite_and_not_weighted_sum():
    points = torch.tensor([[[0.9, 0.2], [0.7, 0.7], [0.2, 0.9], [0.4, 0.4]]])
    weighted = points.matmul(torch.tensor([0.5, 0.5]))

    reward, info = dominance_reward(points, return_info=True)

    assert reward.shape == (1, 4)
    assert torch.isfinite(reward).all()
    assert not torch.allclose(reward, weighted)
    assert info["rank"].shape == (1, 4)
    assert info["hypervolume_contribution"].shape == (1, 4)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_luop_dominance.py -q
```

Expected: import failure because `rl4co.models.zoo.luop_dominance` does not exist.

---

### Task 2: Reward Utility Implementation

**Files:**
- Create: `rl4co/models/zoo/luop_dominance/__init__.py`
- Create: `rl4co/models/zoo/luop_dominance/rewards.py`

- [ ] **Step 1: Implement reward helpers**

Implement:

```python
def nondominated_rank(points: torch.Tensor) -> torch.Tensor:
    ...

def crowding_distance(points: torch.Tensor, ranks: torch.Tensor | None = None) -> torch.Tensor:
    ...

def hypervolume_contribution_2d(points: torch.Tensor, reference=None) -> torch.Tensor:
    ...

def dominance_reward(points: torch.Tensor, ..., return_info: bool = False):
    ...
```

Rules:

- `points` shape is `[..., candidates, objectives]`.
- Objectives are maximized.
- Rank 0 is nondominated.
- Hypervolume contribution supports exactly 2 objectives.
- All returned rewards are finite.
- No weighted sum is used in `dominance_reward`.

- [ ] **Step 2: Run reward tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_luop_dominance.py -q
```

Expected: reward helper tests pass.

- [ ] **Step 3: Commit reward helpers**

```powershell
git add tests/test_luop_dominance.py rl4co/models/zoo/luop_dominance
git commit -m "Add LUOP dominance reward utilities"
```

---

### Task 3: Model Tests

**Files:**
- Modify: `tests/test_luop_dominance.py`

- [ ] **Step 1: Add failing model smoke tests**

Add tests for:

```python
from rl4co.envs.urbanplan.cityplan.env import landuseOptEnv
from rl4co.models import LUOPAttentionModel, LUOPDominanceAttentionModel


def _small_env():
    return landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )


def test_luop_dominance_model_train_step_uses_dominance_reward():
    env = _small_env()
    model = LUOPDominanceAttentionModel(
        env,
        baseline="no",
        batch_size=2,
        train_data_size=2,
        val_data_size=2,
        test_data_size=2,
        num_dominance_candidates=3,
        policy_kwargs={
            "embed_dim": 32,
            "num_encoder_layers": 1,
            "num_heads": 4,
            "feedforward_hidden": 64,
        },
        metrics={"train": ["loss", "reward", "dominance_reward", "pareto_rank_mean"]},
    )
    batch = env.generator(batch_size=[2])

    out = model.shared_step(batch, 0, phase="train")

    assert torch.isfinite(out["loss"])
    assert "train/dominance_reward" in out
    assert "train/pareto_rank_mean" in out


def test_luop_dominance_policy_keeps_multi_action_outputs_valid():
    env = _small_env()
    model = LUOPDominanceAttentionModel(
        env,
        baseline="no",
        batch_size=2,
        train_data_size=2,
        val_data_size=2,
        test_data_size=2,
        policy_kwargs={
            "embed_dim": 32,
            "num_encoder_layers": 1,
            "num_heads": 4,
            "feedforward_hidden": 64,
        },
    )
    batch = env.generator(batch_size=[2])
    td = env.reset(batch)

    out = model.policy(td, env, phase="train", return_actions=True, return_plan=True)
    encoded = env.encode_action(out["type_actions"], out["parcel_actions"], td["locs"].size(-2))
    valid = out["actions"].ne(-1)

    assert torch.equal(encoded[valid], out["actions"][valid])
    assert out["current_plan"].shape == (2, 4)


def test_luop_weighted_model_remains_importable_and_trainable():
    env = _small_env()
    model = LUOPAttentionModel(
        env,
        baseline="no",
        batch_size=2,
        train_data_size=2,
        val_data_size=2,
        test_data_size=2,
        policy_kwargs={
            "embed_dim": 32,
            "num_encoder_layers": 1,
            "num_heads": 4,
            "feedforward_hidden": 64,
        },
        metrics={"train": ["loss", "reward"]},
    )
    batch = env.generator(batch_size=[2])

    out = model.shared_step(batch, 0, phase="train")

    assert torch.isfinite(out["loss"])
    assert "train/reward" in out
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/test_luop_dominance.py -q
```

Expected: import failure for `LUOPDominanceAttentionModel`.

---

### Task 4: New Zoo Model

**Files:**
- Create: `rl4co/models/zoo/luop_dominance/model.py`
- Modify: `rl4co/models/zoo/luop_dominance/__init__.py`
- Modify: `rl4co/models/__init__.py`
- Modify: `rl4co/models/zoo/__init__.py`

- [ ] **Step 1: Implement `LUOPDominanceAttentionModel`**

Model behavior:

- Inherits `LUOPAttentionModel`.
- Constructor defaults to `LUOPAttentionModelPolicy`.
- Adds `num_dominance_candidates`, rank/HV/crowding scale parameters, and optional reference.
- For train phase, expands batch to shape `[batch, num_dominance_candidates]`.
- Calls policy with `select_best=False`.
- Builds dominance reward from `reward_components`.
- Calls `calculate_loss(..., reward=dominance_reward)`.
- Logs `dominance_reward`, `pareto_rank_mean`, `pareto_front_size`, `hv_contribution`.
- For val/test, behaves like `LUOPAttentionModel` so existing Pareto evaluation remains usable.

- [ ] **Step 2: Run model tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_luop_dominance.py -q
```

Expected: all dominance tests pass.

- [ ] **Step 3: Commit model implementation**

```powershell
git add rl4co/models tests/test_luop_dominance.py
git commit -m "Add LUOP dominance attention model"
```

---

### Task 5: Hydra Configs

**Files:**
- Create: `configs/model/luop_dominance.yaml`
- Create: `configs/experiment/cityplan/dominance.yaml`
- Modify: `tests/test_luop_dominance.py`

- [ ] **Step 1: Add config test**

Add a test that composes `experiment=cityplan/dominance` and asserts:

- `_target_` is `rl4co.models.LUOPDominanceAttentionModel`.
- `model.num_dominance_candidates > 1`.
- `model.baseline` can be `"rollout"` in config.
- `env.sample_objective_weights` remains false unless explicitly enabled.

- [ ] **Step 2: Run config test to verify RED**

Run:

```powershell
python -m pytest tests/test_luop_dominance.py -q
```

Expected: config missing.

- [ ] **Step 3: Add configs**

`configs/model/luop_dominance.yaml`:

```yaml
_target_: rl4co.models.LUOPDominanceAttentionModel

baseline: "rollout"
num_dominance_candidates: 4
rank_reward_scale: 1.0
hv_reward_scale: 0.25
crowding_reward_scale: 0.05
dominated_penalty: 0.25
```

`configs/experiment/cityplan/dominance.yaml` should mirror
`configs/experiment/cityplan/am.yaml` but override model to
`luop_dominance.yaml` and add dominance metrics.

- [ ] **Step 4: Run config tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_luop_dominance.py -q
```

Expected: all dominance tests pass.

- [ ] **Step 5: Commit configs**

```powershell
git add configs tests/test_luop_dominance.py
git commit -m "Add LUOP dominance configs"
```

---

### Task 6: Local Regression

**Files:**
- No production edits unless a test exposes a bug.

- [ ] **Step 1: Run targeted tests**

```powershell
python -m pytest tests/test_luop_dominance.py tests/test_luop_joint_multiobjective.py::test_luop_attention_model_has_dedicated_zoo_entrypoint tests/test_luop_joint_multiobjective.py::test_generic_attention_policy_stays_luop_module_free tests/test_luop_joint_multiobjective.py::test_generic_reinforce_stays_luop_metric_free -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run diff check**

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 3: Commit any fixes**

Commit only if local regression required fixes.

---

### Task 7: Report and 4090 Runner

**Files:**
- Create: `docs/content/general/luop_dominance_report.md`
- Create: `scripts/luop_dominance_4090.py`

- [ ] **Step 1: Write report**

Report sections:

- Objective and user constraints.
- High-quality literature basis.
- Reward formula.
- Difference from weighted `luop_am`.
- Local test results.
- 4090 run protocol and results.

- [ ] **Step 2: Add runner**

The runner should print or execute a Hydra command for:

```powershell
python run.py experiment=cityplan/dominance trainer.max_epochs=3 model.batch_size=256 model.train_data_size=1024 model.val_data_size=128 model.test_data_size=128
```

It should also scan logs for finite loss and Pareto metrics.

- [ ] **Step 3: Commit report and runner**

```powershell
git add docs/content/general/luop_dominance_report.md scripts/luop_dominance_4090.py
git commit -m "Add LUOP dominance training report"
```

---

### Task 8: Remote Stability and Push

**Files:**
- Modify report only with observed remote results.

- [ ] **Step 1: Push branch to fork**

```powershell
git push origin main
```

- [ ] **Step 2: Run 4090 smoke training**

Connect to `100.111.43.33` with the provided account, pull latest fork, install
editable package in a clean environment, and run the dominance experiment smoke
command.

- [ ] **Step 3: Record results**

Update `docs/content/general/luop_dominance_report.md` with:

- Commit hash.
- CUDA/Torch versions.
- Command.
- Final train loss.
- Dominance reward range.
- Validation hypervolume/front size.
- Any instability and mitigation.

- [ ] **Step 4: Verify and push final**

```powershell
python -m pytest tests/test_luop_dominance.py -q
git diff --check
git status --short --branch
git push origin main
```

Expected: tests pass, no diff check errors, working tree clean except intentional
remote-run artifacts excluded by `.gitignore`.

---

## Self-Review

- Spec coverage: The plan includes new zoo model, multi-action reuse, dominance
  reward, preservation of weighted model, docs, local tests, remote 4090 record,
  commits, and push.
- Placeholder scan: The plan contains no `TBD`, broad placeholders, or missing
  paths.
- Type consistency: The planned public class is consistently
  `LUOPDominanceAttentionModel`; reward helper names match tests and exports.
