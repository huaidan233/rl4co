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

    assert ranks.tolist() == [[0, 0, 0, 1], [1, 0, 0, 0]]


def test_luop_hypervolume_contribution_rewards_unique_contributors():
    points = torch.tensor([[[0.8, 0.2], [0.6, 0.6], [0.2, 0.8], [0.3, 0.3]]])

    contribution = hypervolume_contribution_2d(
        points,
        reference=torch.tensor([0.0, 0.0]),
    )

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
