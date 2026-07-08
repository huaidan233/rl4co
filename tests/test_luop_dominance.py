import torch

from rl4co.envs.urbanplan.cityplan.env import landuseOptEnv
from rl4co.models import LUOPAttentionModel, LUOPDominanceAttentionModel
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
        metrics={
            "train": [
                "loss",
                "reward",
                "dominance_reward",
                "pareto_rank_mean",
            ],
        },
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
    encoded = env.encode_action(
        out["type_actions"],
        out["parcel_actions"],
        td["locs"].size(-2),
    )
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
