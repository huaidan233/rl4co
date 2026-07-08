from typing import Iterable, Optional

import torch

from tensordict import TensorDict


def normalize_weights(
    weights, device=None, dtype=torch.float32, name: str = "objective weights"
) -> torch.Tensor:
    """Return objective weights normalized along the last dimension."""
    weights = torch.as_tensor(weights, device=device, dtype=dtype)
    if weights.dim() == 0 or weights.size(-1) == 0:
        raise ValueError(f"{name} must have at least one objective dimension")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(f"{name} must contain non-negative finite values")
    weight_sum = weights.sum(dim=-1, keepdim=True)
    if (weight_sum <= 1e-8).any():
        raise ValueError(f"{name} must contain non-negative finite values with a positive row sum")
    return weights / weight_sum


def normalize_weight_grid(
    weights, device=None, dtype=torch.float32, name: str = "objective weights"
) -> torch.Tensor:
    """Return normalized Pareto evaluation weights with shape [num_weights, 2]."""
    weight_grid = normalize_weights(weights, device=device, dtype=dtype, name=name)
    if weight_grid.dim() == 1:
        weight_grid = weight_grid.unsqueeze(0)
    if weight_grid.dim() != 2 or weight_grid.size(-1) != 2:
        raise ValueError(f"{name} must be a weight grid with two objective dimensions")
    return weight_grid


def is_non_dominated(points: torch.Tensor) -> torch.Tensor:
    """Identify non-dominated points for a maximization Pareto front.

    Args:
        points: Tensor with shape ``[..., num_points, num_objectives]``.

    Returns:
        Boolean tensor with shape ``[..., num_points]`` where ``True`` marks
        Pareto-efficient candidates.
    """
    candidates = points.unsqueeze(-3)
    challengers = points.unsqueeze(-2)
    dominates = (challengers >= candidates).all(dim=-1) & (
        challengers > candidates
    ).any(dim=-1)
    return ~dominates.any(dim=-2)


def hypervolume_2d(
    points: torch.Tensor, reference: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute dominated hypervolume for 2-objective maximization fronts."""
    if points.size(-1) != 2:
        raise ValueError("hypervolume_2d expects exactly two objectives")

    if reference is None:
        reference = points.new_zeros(2)
    else:
        reference = torch.as_tensor(
            reference, device=points.device, dtype=points.dtype
        )

    flat_points = points.reshape(-1, points.size(-2), 2)
    flat_hv = []
    for row in flat_points:
        front = row[is_non_dominated(row)]
        front = torch.maximum(front, reference)
        front = front[front[:, 0].argsort()]

        hv = row.new_zeros(())
        prev_x = reference[0]
        for point in front:
            width = (point[0] - prev_x).clamp_min(0)
            height = (point[1] - reference[1]).clamp_min(0)
            hv = hv + width * height
            prev_x = torch.maximum(prev_x, point[0])
        flat_hv.append(hv)

    return torch.stack(flat_hv).reshape(points.shape[:-2])


def _masked_pareto_values(
    values: torch.Tensor, pareto_mask: torch.Tensor, fill_value=0
) -> tuple:
    """Return Pareto-filtered values padded to the largest front in the batch."""
    trailing_shape = values.shape[pareto_mask.dim() :]
    flat_values = values.reshape(-1, values.size(pareto_mask.dim() - 1), *trailing_shape)
    flat_mask = pareto_mask.reshape(-1, pareto_mask.size(-1))
    max_front_size = int(flat_mask.sum(dim=-1).max().item()) if flat_mask.numel() else 0
    padded = values.new_full(
        (flat_mask.size(0), max_front_size, *trailing_shape), fill_value
    )
    valid = torch.zeros(
        flat_mask.size(0), max_front_size, dtype=torch.bool, device=values.device
    )

    for row_idx, (row_values, row_mask) in enumerate(zip(flat_values, flat_mask)):
        selected = row_values[row_mask]
        count = selected.size(0)
        if count > 0:
            padded[row_idx, :count] = selected
            valid[row_idx, :count] = True

    return padded.reshape(
        *pareto_mask.shape[:-1], max_front_size, *trailing_shape
    ), valid.reshape(*pareto_mask.shape[:-1], max_front_size)


def _stack_padded_sequence_values(
    values: list[torch.Tensor], dim: int = 1, fill_value=0
) -> torch.Tensor:
    """Stack tensors after padding a variable-length sequence axis."""
    if not values:
        raise ValueError("values must contain at least one tensor")

    reference = values[0]
    max_length = max(value.size(-1) for value in values)
    padded_values = []
    for value in values:
        if value.shape[:-1] != reference.shape[:-1]:
            raise ValueError(
                "Pareto artifact tensors must share batch dimensions; "
                f"got {tuple(value.shape)} and {tuple(reference.shape)}"
            )
        padded = value.new_full((*value.shape[:-1], max_length), fill_value)
        if value.size(-1) > 0:
            padded[..., : value.size(-1)] = value
        padded_values.append(padded)
    return torch.stack(padded_values, dim=dim)


def _require_shape(name: str, value: torch.Tensor, expected_shape: torch.Size) -> None:
    if value.shape != expected_shape:
        raise ValueError(
            f"evaluate_pareto_front expected {name} shape "
            f"{tuple(expected_shape)}, got {tuple(value.shape)}"
        )


def _require_sequence_shape(
    name: str, value: torch.Tensor, batch_size: torch.Size
) -> None:
    if value.dim() != len(batch_size) + 1 or value.shape[:-1] != batch_size:
        raise ValueError(
            "evaluate_pareto_front expected "
            f"{name} shape {tuple(batch_size)} + (steps,), got {tuple(value.shape)}"
        )


def _require_integer_sequence_artifact(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"evaluate_pareto_front requires finite {name}")
    if value.is_floating_point() and not torch.equal(value, value.round()):
        raise ValueError(
            f"evaluate_pareto_front requires {name} to contain integer action ids"
        )
    if (value < -1).any():
        raise ValueError(
            f"evaluate_pareto_front requires {name} to use -1 only for padding"
        )


def _require_suffix_padding(name: str, value: torch.Tensor) -> None:
    padding = value.eq(-1)
    if padding.size(-1) <= 1:
        return
    non_suffix_padding = padding[..., :-1] & ~padding[..., 1:]
    if non_suffix_padding.any():
        raise ValueError(
            f"evaluate_pareto_front requires {name} padding to be a trailing suffix"
        )


def _require_sequence_value_range(
    name: str,
    value: torch.Tensor,
    upper_bound: int,
) -> None:
    if (value >= upper_bound).any():
        raise ValueError(
            f"evaluate_pareto_front requires {name} values in range "
            f"[-1, {upper_bound - 1}]"
        )


def _require_matching_padding(
    reference_name: str,
    reference: torch.Tensor,
    name: str,
    value: torch.Tensor,
) -> None:
    if not torch.equal(reference.eq(-1), value.eq(-1)):
        raise ValueError(
            "evaluate_pareto_front requires matching padding across "
            f"{reference_name} and {name}"
        )


def _normalize_padding_from_initial_plan(
    value: torch.Tensor,
    initial_plan: torch.Tensor,
) -> torch.Tensor:
    """Replace per-row replay padding beyond required LUOP steps with -1."""
    if value.size(-1) == 0:
        return value

    row_steps = initial_plan.lt(0).sum(dim=-1).to(device=value.device)
    step_idx = torch.arange(
        value.size(-1),
        device=value.device,
        dtype=row_steps.dtype,
    ).view((1,) * row_steps.dim() + (value.size(-1),))
    padding = step_idx >= row_steps.unsqueeze(-1)
    return torch.where(padding, value.new_full((), -1), value)


def _require_dual_actions_encode_flat_actions(
    actions: torch.Tensor,
    type_actions: torch.Tensor,
    parcel_actions: torch.Tensor,
    num_loc: int,
) -> None:
    valid = actions.ne(-1)
    encoded = type_actions.to(actions.device) * num_loc + parcel_actions.to(
        actions.device
    )
    if not torch.equal(encoded[valid], actions[valid]):
        raise ValueError(
            "evaluate_pareto_front requires type_actions and parcel_actions "
            "to encode the returned flat actions"
        )


def _require_complete_plan_artifact(
    current_plan: torch.Tensor,
    num_types: int,
) -> None:
    if not torch.isfinite(current_plan).all():
        raise ValueError("evaluate_pareto_front requires finite current_plan")
    if current_plan.is_floating_point() and not torch.equal(
        current_plan, current_plan.round()
    ):
        raise ValueError(
            "evaluate_pareto_front requires current_plan to contain integer land-use types"
        )
    if (current_plan < 0).any():
        raise ValueError(
            "evaluate_pareto_front requires current_plan to be complete with no "
            "unassigned parcels"
        )
    if (current_plan >= num_types).any():
        raise ValueError(
            "evaluate_pareto_front requires current_plan to contain known land-use "
            f"type ids in [0, {num_types - 1}]"
        )


def _replay_actions_through_env(
    actions: torch.Tensor,
    env,
    initial_td: TensorDict,
) -> TensorDict:
    replay_td = initial_td.clone()
    for step in range(actions.size(-1)):
        step_action = actions[..., step].long().to(replay_td.device)
        active = ~replay_td["done"].squeeze(-1)
        safe_action = torch.where(
            step_action.eq(-1),
            torch.zeros_like(step_action),
            step_action,
        )
        feasible = replay_td["action_mask"].gather(
            -1, safe_action.unsqueeze(-1)
        ).squeeze(-1)
        invalid = active & (step_action.eq(-1) | ~feasible)
        if invalid.any():
            rows = (
                invalid.reshape(-1)
                .nonzero(as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            )
            if len(rows) > 8:
                rows = rows[:8] + ["..."]
            raise ValueError(
                "evaluate_pareto_front requires returned actions to satisfy "
                "the current LUOP action_mask at each replay step; "
                f"infeasible batch rows at step {step}: {rows}"
            )
        replay_td.set("action", safe_action)
        replay_td = env.step(replay_td)["next"]
    return replay_td


def _require_complete_action_trace(
    actions: torch.Tensor,
    initial_plan: torch.Tensor,
) -> torch.Tensor:
    valid_actions = actions.ne(-1)
    required_steps = initial_plan.lt(0).sum(dim=-1)
    valid_counts = valid_actions.sum(dim=-1)
    complete_trace = valid_counts.eq(required_steps)
    incomplete_trace = ~complete_trace
    if incomplete_trace.any():
        rows = (
            incomplete_trace.reshape(-1)
            .nonzero(as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        if len(rows) > 8:
            rows = rows[:8] + ["..."]
        raise ValueError(
            "evaluate_pareto_front requires returned actions to include one "
            "valid action for each initially unassigned parcel; mismatched "
            f"batch rows: {rows}"
        )
    return complete_trace


def _require_actions_replay_to_current_plan(
    actions: torch.Tensor,
    initial_plan: torch.Tensor,
    current_plan: torch.Tensor,
    num_loc: int,
    env=None,
    initial_td: Optional[TensorDict] = None,
) -> None:
    complete_trace = _require_complete_action_trace(actions, initial_plan)
    if not complete_trace.any():
        return

    if env is not None and initial_td is not None and "action_mask" in initial_td.keys():
        replay_td = _replay_actions_through_env(actions, env, initial_td)
        if not torch.equal(replay_td["current_plan"].long(), current_plan.long()):
            raise ValueError(
                "evaluate_pareto_front requires returned actions to replay to the "
                "returned current_plan"
            )

    flat_actions = actions.reshape(-1, actions.size(-1)).long()
    flat_initial_plan = initial_plan.reshape(-1, num_loc).long()
    flat_current_plan = current_plan.reshape(-1, num_loc).long()
    flat_complete_trace = complete_trace.reshape(-1)
    invalid_rows = []

    for row in flat_complete_trace.nonzero(as_tuple=False).flatten().tolist():
        replayed_plan = flat_initial_plan[row].clone()
        row_actions = flat_actions[row][flat_actions[row].ne(-1)]
        valid_replay = True
        for action in row_actions:
            selected_type = action // num_loc
            parcel = action % num_loc
            if replayed_plan[parcel] >= 0:
                valid_replay = False
                break
            replayed_plan[parcel] = selected_type
        if not valid_replay or not torch.equal(replayed_plan, flat_current_plan[row]):
            invalid_rows.append(row)

    if invalid_rows:
        if len(invalid_rows) > 8:
            invalid_rows = invalid_rows[:8] + ["..."]
        raise ValueError(
            "evaluate_pareto_front requires returned actions to replay to the "
            f"returned current_plan; mismatched batch rows: {invalid_rows}"
        )


def evaluate_pareto_front(
    policy: torch.nn.Module,
    env,
    batch: TensorDict,
    weights: Iterable[Iterable[float]],
    reference: Optional[Iterable[float]] = None,
    phase: str = "test",
    decode_type: str = "greedy",
    return_actions: bool = True,
    return_plan: bool = True,
    **policy_kwargs,
) -> dict:
    """Evaluate one policy across a weight grid and summarize Pareto candidates."""
    weight_grid = normalize_weight_grid(
        weights, device=batch.device, dtype=batch["locs"].dtype
    )
    if reference is not None:
        reference_tensor = torch.as_tensor(
            reference,
            device=batch.device,
            dtype=batch["locs"].dtype,
        )
        if reference_tensor.shape != torch.Size([weight_grid.size(-1)]):
            raise ValueError(
                "evaluate_pareto_front reference must have shape "
                f"({weight_grid.size(-1)},), got {tuple(reference_tensor.shape)}"
            )
    else:
        reference_tensor = None
    components = []
    rewards = []
    actions = []
    type_actions = []
    parcel_actions = []
    plans = []

    for weight in weight_grid:
        candidate = batch.clone()
        expanded_weight = weight.expand(*candidate.batch_size, -1)
        candidate.set("objective_weights", expanded_weight)
        td = env.reset(candidate)
        eval_policy_kwargs = dict(policy_kwargs)
        if "multistart" in decode_type and "select_best" not in eval_policy_kwargs:
            eval_policy_kwargs["select_best"] = True
        out = policy(
            td,
            env,
            phase=phase,
            decode_type=decode_type,
            return_actions=return_actions,
            return_plan=return_plan,
            **eval_policy_kwargs,
        )

        if "reward_components" not in out:
            raise ValueError(
                "evaluate_pareto_front requires policy outputs to include "
                "'reward_components'. Ensure the environment reward exposes "
                "per-objective components and calc_reward=True."
            )
        _require_shape(
            "reward_components",
            out["reward_components"],
            torch.Size((*td.batch_size, weight_grid.size(-1))),
        )
        _require_shape("reward", out["reward"], torch.Size(td.batch_size))
        if not torch.isfinite(out["reward_components"]).all():
            raise ValueError(
                "evaluate_pareto_front requires finite reward_components"
            )
        if not torch.isfinite(out["reward"]).all():
            raise ValueError("evaluate_pareto_front requires finite rewards")
        components.append(out["reward_components"])
        rewards.append(out["reward"])
        normalized_actions = None
        if return_actions:
            if "actions" not in out:
                raise ValueError(
                    "evaluate_pareto_front requires policy outputs to include "
                    "'actions' when return_actions=True"
                )
            out_actions = out["actions"]
            _require_sequence_shape("actions", out_actions, td.batch_size)
            _require_integer_sequence_artifact("actions", out_actions)
            num_loc = td["locs"].shape[-2]
            num_types = getattr(env, "num_types", 8)
            _require_sequence_value_range(
                "actions",
                out_actions,
                num_types * num_loc,
            )
            out_actions = _normalize_padding_from_initial_plan(
                out_actions,
                td["current_plan"],
            )
            _require_suffix_padding("actions", out_actions)
            normalized_actions = out_actions
            actions.append(out_actions)
            has_type_actions = "type_actions" in out
            has_parcel_actions = "parcel_actions" in out
            if has_type_actions != has_parcel_actions:
                raise ValueError(
                    "evaluate_pareto_front requires type_actions and parcel_actions "
                    "to be returned together for dual-action policies"
                )
            if not has_type_actions and hasattr(env, "decode_action"):
                out_type_actions, out_parcel_actions = env.decode_action(
                    out_actions.long(),
                    num_loc,
                )
                out_type_actions = torch.where(
                    out_actions.eq(-1),
                    out_actions,
                    out_type_actions,
                )
                out_parcel_actions = torch.where(
                    out_actions.eq(-1),
                    out_actions,
                    out_parcel_actions,
                )
                type_actions.append(out_type_actions)
                parcel_actions.append(out_parcel_actions)
            elif has_type_actions:
                out_type_actions = out["type_actions"]
                out_parcel_actions = out["parcel_actions"]
                _require_shape("type_actions", out_type_actions, out_actions.shape)
                _require_shape(
                    "parcel_actions", out_parcel_actions, out_actions.shape
                )
                _require_integer_sequence_artifact("type_actions", out_type_actions)
                _require_integer_sequence_artifact("parcel_actions", out_parcel_actions)
                _require_sequence_value_range(
                    "type_actions",
                    out_type_actions,
                    num_types,
                )
                _require_sequence_value_range(
                    "parcel_actions",
                    out_parcel_actions,
                    num_loc,
                )
                out_type_actions = _normalize_padding_from_initial_plan(
                    out_type_actions,
                    td["current_plan"],
                )
                out_parcel_actions = _normalize_padding_from_initial_plan(
                    out_parcel_actions,
                    td["current_plan"],
                )
                _require_suffix_padding("type_actions", out_type_actions)
                _require_suffix_padding("parcel_actions", out_parcel_actions)
                _require_matching_padding(
                    "actions", out_actions, "type_actions", out_type_actions
                )
                _require_matching_padding(
                    "actions",
                    out_actions,
                    "parcel_actions",
                    out_parcel_actions,
                )
                _require_dual_actions_encode_flat_actions(
                    out_actions,
                    out_type_actions,
                    out_parcel_actions,
                    num_loc,
                )
                type_actions.append(out_type_actions)
                parcel_actions.append(out_parcel_actions)
        if return_plan:
            if "current_plan" not in out:
                raise ValueError(
                    "evaluate_pareto_front requires policy outputs to include "
                    "'current_plan' when return_plan=True"
                )
            _require_shape(
                "current_plan",
                out["current_plan"],
                torch.Size((*td.batch_size, td["locs"].shape[-2])),
            )
            _require_complete_plan_artifact(
                out["current_plan"],
                getattr(env, "num_types", 8),
            )
            if hasattr(env, "check_solution_validity"):
                plan_td = td.clone()
                plan_td.set("current_plan", out["current_plan"])
                env.check_solution_validity(
                    plan_td,
                    normalized_actions
                    if normalized_actions is not None
                    else out.get(
                        "actions",
                        torch.empty(
                            *td.batch_size,
                            0,
                            dtype=torch.long,
                            device=out["current_plan"].device,
                        ),
                    ),
                )
            if return_actions and normalized_actions is not None:
                _require_actions_replay_to_current_plan(
                    normalized_actions,
                    td["current_plan"],
                    out["current_plan"],
                    td["locs"].shape[-2],
                    env=env,
                    initial_td=td,
                )
            plans.append(out["current_plan"])
        elif return_actions and normalized_actions is not None:
            if hasattr(env, "step") and "action_mask" in td.keys():
                _replay_actions_through_env(normalized_actions, env, td)

    candidate_dim = len(batch.batch_size)
    components = torch.stack(components, dim=candidate_dim)
    rewards = torch.stack(rewards, dim=candidate_dim)
    pareto_mask = is_non_dominated(components)
    if reference_tensor is not None:
        reference_tensor = reference_tensor.to(
            device=components.device,
            dtype=components.dtype,
        )

    front_components, front_mask = _masked_pareto_values(components, pareto_mask)
    front_rewards, _ = _masked_pareto_values(rewards, pareto_mask)
    expanded_weights = weight_grid.expand(*components.shape[:-2], -1, -1)
    front_weights, _ = _masked_pareto_values(expanded_weights, pareto_mask)

    result = {
        "weights": weight_grid,
        "components": components,
        "front_components": front_components,
        "front_mask": front_mask,
        "front_weights": front_weights,
        "rewards": rewards,
        "front_rewards": front_rewards,
        "is_pareto": pareto_mask,
        "front_size": pareto_mask.sum(dim=-1).float(),
        "hypervolume": hypervolume_2d(components, reference=reference_tensor),
    }
    if return_actions:
        stacked_actions = _stack_padded_sequence_values(
            actions, dim=candidate_dim, fill_value=-1
        )
        result["actions"] = stacked_actions
        result["front_actions"], _ = _masked_pareto_values(
            stacked_actions, pareto_mask, fill_value=-1
        )
        if type_actions:
            stacked_type_actions = _stack_padded_sequence_values(
                type_actions, dim=candidate_dim, fill_value=-1
            )
            result["type_actions"] = stacked_type_actions
            result["front_type_actions"], _ = _masked_pareto_values(
                stacked_type_actions, pareto_mask, fill_value=-1
            )
        if parcel_actions:
            stacked_parcel_actions = _stack_padded_sequence_values(
                parcel_actions, dim=candidate_dim, fill_value=-1
            )
            result["parcel_actions"] = stacked_parcel_actions
            result["front_parcel_actions"], _ = _masked_pareto_values(
                stacked_parcel_actions, pareto_mask, fill_value=-1
            )
    if return_plan:
        stacked_plans = torch.stack(plans, dim=candidate_dim)
        result["current_plan"] = stacked_plans
        result["front_current_plan"], _ = _masked_pareto_values(
            stacked_plans, pareto_mask, fill_value=-1
        )
    return result
