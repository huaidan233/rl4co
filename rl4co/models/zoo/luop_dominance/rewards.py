from typing import Optional

import torch


def _validate_points(points: torch.Tensor) -> None:
    if points.dim() < 2:
        raise ValueError("points must have shape (..., candidates, objectives)")
    if points.size(-2) == 0 or points.size(-1) == 0:
        raise ValueError("points must include candidates and objectives")
    if not torch.isfinite(points).all():
        raise ValueError("points must contain finite objective values")


def _is_non_dominated_row(points: torch.Tensor) -> torch.Tensor:
    candidates = points.unsqueeze(0)
    challengers = points.unsqueeze(1)
    dominates = (challengers >= candidates).all(dim=-1) & (
        challengers > candidates
    ).any(dim=-1)
    return ~dominates.any(dim=0)


def nondominated_rank(points: torch.Tensor) -> torch.Tensor:
    """Return Pareto front ranks for maximization objective points."""
    _validate_points(points)
    flat_points = points.reshape(-1, points.size(-2), points.size(-1))
    flat_ranks = torch.zeros(
        flat_points.shape[:2],
        dtype=torch.long,
        device=points.device,
    )

    for row_idx, row in enumerate(flat_points):
        remaining = torch.ones(row.size(0), dtype=torch.bool, device=points.device)
        rank = 0
        while remaining.any():
            remaining_indices = remaining.nonzero(as_tuple=False).flatten()
            front = _is_non_dominated_row(row[remaining])
            flat_ranks[row_idx, remaining_indices[front]] = rank
            remaining[remaining_indices[front]] = False
            rank += 1

    return flat_ranks.reshape(points.shape[:-1])


def crowding_distance(
    points: torch.Tensor,
    ranks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return finite NSGA-II style crowding distances for each rank front."""
    _validate_points(points)
    if ranks is None:
        ranks = nondominated_rank(points)
    if ranks.shape != points.shape[:-1]:
        raise ValueError("ranks must have shape matching points without objectives")

    flat_points = points.reshape(-1, points.size(-2), points.size(-1))
    flat_ranks = ranks.reshape(-1, ranks.size(-1))
    flat_distance = points.new_zeros(flat_ranks.shape)

    for row_idx, (row, row_ranks) in enumerate(zip(flat_points, flat_ranks)):
        for rank in row_ranks.unique(sorted=True):
            front_mask = row_ranks.eq(rank)
            front_indices = front_mask.nonzero(as_tuple=False).flatten()
            front = row[front_indices]
            if front.size(0) == 1:
                flat_distance[row_idx, front_indices] = 1.0
                continue
            if front.size(0) == 2:
                flat_distance[row_idx, front_indices] = 1.0
                continue

            front_distance = front.new_zeros(front.size(0))
            for objective in range(front.size(-1)):
                values = front[:, objective]
                order = values.argsort()
                sorted_values = values[order]
                scale = (sorted_values[-1] - sorted_values[0]).clamp_min(1e-8)
                front_distance[order[0]] = 1.0
                front_distance[order[-1]] = 1.0
                if front.size(0) > 2:
                    increments = (sorted_values[2:] - sorted_values[:-2]) / (
                        scale * max(2, front.size(-1))
                    )
                    middle = order[1:-1]
                    front_distance[middle] = torch.maximum(
                        front_distance[middle],
                        increments.clamp_min(0),
                    )
            flat_distance[row_idx, front_indices] = front_distance.clamp(0, 1)

    return flat_distance.reshape(ranks.shape)


def _hypervolume_2d_row(
    points: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if points.numel() == 0:
        return reference.new_zeros(())
    front = points[_is_non_dominated_row(points)]
    front = torch.maximum(front, reference)
    front = front[front[:, 0].argsort()]

    hypervolume = points.new_zeros(())
    previous_x = reference[0]
    for point in front:
        width = (point[0] - previous_x).clamp_min(0)
        height = (point[1] - reference[1]).clamp_min(0)
        hypervolume = hypervolume + width * height
        previous_x = torch.maximum(previous_x, point[0])
    return hypervolume


def hypervolume_contribution_2d(
    points: torch.Tensor,
    reference=None,
) -> torch.Tensor:
    """Return per-candidate dominated hypervolume contribution for 2D maximization."""
    _validate_points(points)
    if points.size(-1) != 2:
        raise ValueError("hypervolume_contribution_2d expects two objectives")
    if reference is None:
        reference_tensor = points.new_zeros(2)
    else:
        reference_tensor = torch.as_tensor(
            reference,
            device=points.device,
            dtype=points.dtype,
        )
    if reference_tensor.shape != torch.Size([2]):
        raise ValueError("reference must have shape (2,)")

    flat_points = points.reshape(-1, points.size(-2), 2)
    flat_contribution = points.new_zeros(flat_points.shape[:-1])
    for row_idx, row in enumerate(flat_points):
        full_hv = _hypervolume_2d_row(row, reference_tensor)
        front_mask = _is_non_dominated_row(row)
        for point_idx in front_mask.nonzero(as_tuple=False).flatten():
            keep = torch.ones(row.size(0), dtype=torch.bool, device=points.device)
            keep[point_idx] = False
            without_hv = _hypervolume_2d_row(row[keep], reference_tensor)
            flat_contribution[row_idx, point_idx] = (full_hv - without_hv).clamp_min(0)
    return flat_contribution.reshape(points.shape[:-1])


def _normalize_bonus(values: torch.Tensor) -> torch.Tensor:
    max_values = values.max(dim=-1, keepdim=True).values
    return torch.where(
        max_values > 1e-8,
        values / max_values.clamp_min(1e-8),
        torch.zeros_like(values),
    )


def dominance_reward(
    points: torch.Tensor,
    reference=None,
    rank_reward_scale: float = 1.0,
    hv_reward_scale: float = 0.25,
    crowding_reward_scale: float = 0.05,
    dominated_penalty: float = 0.25,
    return_info: bool = False,
):
    """Build a finite scalar reward from Pareto rank, HV contribution, and crowding."""
    ranks = nondominated_rank(points)
    hv_contribution = hypervolume_contribution_2d(points, reference=reference)
    crowding = crowding_distance(points, ranks)

    max_rank = ranks.max(dim=-1, keepdim=True).values.to(points.dtype)
    rank_score = 1.0 - ranks.to(points.dtype) / (max_rank + 1.0)
    hv_bonus = _normalize_bonus(hv_contribution)
    crowding_bonus = _normalize_bonus(crowding)
    dominated = ranks.gt(0).to(points.dtype)

    reward = (
        rank_reward_scale * rank_score
        + hv_reward_scale * hv_bonus
        + crowding_reward_scale * crowding_bonus
        - dominated_penalty * dominated
    )
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    info = {
        "rank": ranks,
        "rank_score": rank_score,
        "hypervolume_contribution": hv_contribution,
        "crowding_distance": crowding,
        "is_pareto": ranks.eq(0),
    }
    if return_info:
        return reward, info
    return reward
