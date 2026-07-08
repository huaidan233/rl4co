import inspect

import torch
import pytest
from tensordict import TensorDict

from rl4co.envs import (
    MAlanduseOptEnv,
    landuseOptCompatibilityEnv,
    landuseOptEnv,
    landuseOptNearestEnv,
)
from rl4co.envs.urbanplan.cityplan import init as cityplan_init
from rl4co.models.nn.env_embeddings.context import LOPContext
from rl4co.models.nn.env_embeddings.dynamic import LOPDynamicEmbedding
from rl4co.models.nn.env_embeddings.init import lopInitEmbedding
from rl4co.models.common.constructive.base import (
    ConstructiveDecoder,
    ConstructivePolicy as GenericConstructivePolicy,
)
from rl4co.models.rl.reinforce.baselines import RolloutBaseline, get_reinforce_baseline
from rl4co.models.zoo.am.decoder import (
    AttentionModelDecoder as GenericAttentionModelDecoder,
)
from rl4co.models.zoo.am.policy import AttentionModelPolicy as GenericAttentionModelPolicy
from rl4co.models.zoo.luop_am.model import LUOPAttentionModel as AttentionModel
from rl4co.models.zoo.luop_am.policy import (
    LUOPConstructivePolicy,
    LUOPAttentionModelPolicy as AttentionModelPolicy,
)
from rl4co.utils.decoding import get_decoding_strategy
from rl4co.utils.multi_objective import (
    evaluate_pareto_front,
    hypervolume_2d,
    is_non_dominated,
    normalize_weights,
)


def _instance(areas, fixed_mask=None, init_plan=None):
    num_loc = len(areas)
    locs = torch.linspace(0, 1, num_loc).view(1, num_loc, 1).repeat(1, 1, 2)
    if fixed_mask is None:
        fixed_mask = torch.ones(1, num_loc, dtype=torch.bool)
    if init_plan is None:
        init_plan = torch.ones(1, num_loc, dtype=torch.long)
    return TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor([areas], dtype=torch.float32),
            "init_plan": init_plan.long(),
            "fixed_mask": fixed_mask.bool(),
        },
        batch_size=[1],
    )


def test_luop_reset_exposes_joint_type_parcel_mask_and_blocks_infeasible_pairs():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0.6, 0.4, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.6, 0.3, 0.1]))

    assert td["action_mask"].shape == (1, env.num_types * 3)
    assert td["parcel_action_mask"].shape == (1, 3)
    assert td["type_action_mask"].shape == (1, env.num_types)
    assert td["type_parcel_action_mask"].shape == (1, env.num_types, 3)
    assert torch.equal(td["current_plan"], torch.full((1, 3), -1))

    valid_type0_node2 = env.encode_action(torch.tensor([0]), torch.tensor([2]), 3)
    invalid_type1_node0 = env.encode_action(torch.tensor([1]), torch.tensor([0]), 3)

    assert td["action_mask"][0, valid_type0_node2.item()]
    assert not td["action_mask"][0, invalid_type1_node0.item()]
    assert torch.equal(
        td["type_parcel_action_mask"].reshape(1, -1),
        td["action_mask"],
    )


def test_luop_generator_supports_shaped_batch_fixed_masks_and_areas():
    env = landuseOptEnv(
        generator_params={"num_loc": 6, "num_fixed": 2},
        check_solution=False,
    )

    batch = env.generator(batch_size=[2, 3])

    assert batch.batch_size == torch.Size([2, 3])
    assert batch["locs"].shape == (2, 3, 6, 2)
    assert batch["areas"].shape == (2, 3, 6)
    assert batch["init_plan"].shape == (2, 3, 6)
    assert batch["fixed_mask"].shape == (2, 3, 6)
    assert torch.allclose(batch["areas"].sum(dim=-1), torch.ones(2, 3))
    assert torch.equal(
        (~batch["fixed_mask"]).sum(dim=-1),
        torch.full((2, 3), 2),
    )
    assert (batch["init_plan"][~batch["fixed_mask"]] == 4).sum() == 6
    assert (batch["init_plan"][~batch["fixed_mask"]] == 6).sum() == 6


def test_luop_generator_all_fixed_instances_have_finite_normalized_areas():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 3},
        check_solution=False,
    )

    batch = env.generator(batch_size=[2, 3])
    td = env.reset(batch)

    assert torch.isfinite(batch["areas"]).all()
    assert torch.allclose(batch["areas"].sum(dim=-1), torch.ones(2, 3))
    assert (~batch["fixed_mask"]).all()
    assert td["done"].all()


def test_luop_exposes_parcel_mask_conditioned_on_selected_type():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.6, 0.3, 0.1]))

    td.set("selected_type", torch.tensor([0]))
    type0_mask = env.get_parcel_action_mask(td)
    td.set("selected_type", torch.tensor([1]))
    type1_mask = env.get_parcel_action_mask(td)

    assert torch.equal(type0_mask, td["type_parcel_action_mask"][0:1, 0])
    assert torch.equal(type1_mask, td["type_parcel_action_mask"][0:1, 1])


def test_luop_parcel_mask_uses_pending_type_action_for_type_first_decoding():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0.6, 0.4, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.6, 0.3, 0.1]))

    td.set("selected_type", torch.tensor([1]))
    td.set("type_action", torch.tensor([0]))

    assert torch.equal(
        env.get_parcel_action_mask(td),
        td["type_parcel_action_mask"][0:1, 0],
    )


def test_luop_parcel_mask_prefers_pending_type_over_stale_type_action():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0.6, 0.4, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.6, 0.3, 0.1]))

    td.set("type_action", torch.tensor([1]))
    td.set("pending_type_action", torch.tensor([0]))

    assert torch.equal(
        env.get_parcel_action_mask(td),
        td["type_parcel_action_mask"][0:1, 0],
    )


def test_luop_parcel_mask_rejects_invalid_conditioning_type():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.5, 0.3, 0.2]))
    td.set("pending_type_action", torch.tensor([env.num_types]))

    with pytest.raises(ValueError, match="type-conditioned parcel mask"):
        env.get_parcel_action_mask(td)


def test_luop_reset_drops_pending_type_action():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = _instance([0.5, 0.3, 0.2])
    td.set("pending_type_action", torch.tensor([2]))

    td = env.reset(td)

    assert "pending_type_action" not in td.keys()


def test_luop_step_decodes_joint_action_updates_plan_and_masks_selected_parcel():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.5, 0.3, 0.2]))

    td.set("action", env.encode_action(torch.tensor([2]), torch.tensor([1]), 3))
    td = env.step(td)["next"]

    assert td["selected_type"].item() == 2
    assert td["current_node"].item() == 1
    assert td["current_plan"][0, 1].item() == 2
    assert td["current_types_onehot"][0, 2]
    assert not td["action_mask"].view(1, env.num_types, 3)[0, :, 1].any()


def test_luop_step_accepts_explicit_type_and_parcel_actions():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.5, 0.3, 0.2]))

    td.set("type_action", torch.tensor([3]))
    td.set("parcel_action", torch.tensor([2]))
    td = env.step(td)["next"]

    assert (
        td["action"].item()
        == env.encode_action(torch.tensor([3]), torch.tensor([2]), 3).item()
    )
    assert td["selected_type"].item() == 3
    assert td["current_node"].item() == 2
    assert td["current_plan"][0, 2].item() == 3


def test_luop_reset_and_step_support_shaped_batch_joint_actions():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch_shape = (2, 3)
    num_loc = 4
    batch = TensorDict(
        {
            "locs": torch.rand(*batch_shape, num_loc, 2),
            "areas": torch.full((*batch_shape, num_loc), 0.25),
            "init_plan": torch.ones(*batch_shape, num_loc, dtype=torch.long),
            "fixed_mask": torch.ones(*batch_shape, num_loc, dtype=torch.bool),
        },
        batch_size=batch_shape,
    )

    td = env.reset(batch)
    type_action = torch.tensor([[0, 1, 2], [3, 4, 5]])
    parcel_action = torch.tensor([[0, 1, 2], [3, 0, 1]])
    td.set("action", env.encode_action(type_action, parcel_action, num_loc))
    td = env.step(td)["next"]

    expected_plan = torch.full((*batch_shape, num_loc), -1, dtype=torch.long)
    expected_plan.scatter_(-1, parcel_action.unsqueeze(-1), type_action.unsqueeze(-1))
    assert td.batch_size == torch.Size(batch_shape)
    assert td["action_mask"].shape == (*batch_shape, env.num_types * num_loc)
    assert td["type_parcel_action_mask"].shape == (*batch_shape, env.num_types, num_loc)
    assert torch.equal(td["current_plan"], expected_plan)
    assert torch.equal(td["selected_type"], type_action)
    assert torch.equal(td["current_node"], parcel_action)


def test_luop_step_rejects_out_of_range_flat_action():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.5, 0.3, 0.2]))
    td.set("action", torch.tensor([env.num_types * 3]))

    with pytest.raises(ValueError, match="LUOP flat actions"):
        env.step(td)


def test_luop_step_rejects_out_of_range_explicit_component_actions():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )

    td = env.reset(_instance([0.5, 0.3, 0.2]))
    td.set("type_action", torch.tensor([env.num_types]))
    td.set("parcel_action", torch.tensor([0]))
    with pytest.raises(ValueError, match="LUOP component actions"):
        env.step(td)

    td = env.reset(_instance([0.5, 0.3, 0.2]))
    td.set("type_action", torch.tensor([0]))
    td.set("parcel_action", torch.tensor([3]))
    with pytest.raises(ValueError, match="LUOP component actions"):
        env.step(td)


def test_luop_step_rejects_aliasing_explicit_components_when_flat_action_is_present():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.5, 0.3, 0.2]))
    td.set("action", env.encode_action(torch.tensor([0]), torch.tensor([2]), 3))
    td.set("type_action", torch.tensor([1]))
    td.set("parcel_action", torch.tensor([-1]))

    with pytest.raises(ValueError, match="LUOP component actions"):
        env.step(td)


def test_luop_step_rejects_mismatched_flat_and_explicit_actions():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.5, 0.3, 0.2]))
    td.set("action", env.encode_action(torch.tensor([1]), torch.tensor([0]), 3))
    td.set("type_action", torch.tensor([2]))
    td.set("parcel_action", torch.tensor([0]))

    with pytest.raises(
        ValueError, match="mismatched flat and explicit LUOP joint actions"
    ):
        env.step(td)


def test_luop_step_accepts_mixed_rows_with_matching_and_stale_explicit_actions():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor(
                [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]], dtype=torch.float32
            ),
            "init_plan": torch.ones(2, 3, dtype=torch.long),
            "fixed_mask": torch.ones(2, 3, dtype=torch.bool),
        },
        batch_size=[2],
    )
    td = env.reset(batch)

    first_action = env.encode_action(torch.tensor([2, 4]), torch.tensor([0, 1]), 3)
    td.set("action", first_action)
    td = env.step(td)["next"]

    second_action = env.encode_action(torch.tensor([3, 5]), torch.tensor([1, 2]), 3)
    td.set("action", second_action)
    td.set("type_action", torch.tensor([3, 4]))
    td.set("parcel_action", torch.tensor([1, 1]))

    td = env.step(td)["next"]

    assert td["current_plan"][0, 1].item() == 3
    assert td["current_plan"][1, 2].item() == 5


def test_luop_reset_drops_transient_action_keys_from_reused_state():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.5, 0.3, 0.2]))
    td.set("action", env.encode_action(torch.tensor([1]), torch.tensor([0]), 3))
    td.set("type_action", torch.tensor([1]))
    td.set("parcel_action", torch.tensor([0]))
    td.set("reward_components", torch.ones(1, 2))

    reset_td = env.reset(td.clone())

    assert "action" not in reset_td.keys()
    assert "type_action" not in reset_td.keys()
    assert "parcel_action" not in reset_td.keys()
    assert "reward_components" not in reset_td.keys()


def test_luop_fixed_parcels_remain_assigned_and_unselectable():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 1},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(
        _instance(
            [0.4, 0.3, 0.3],
            fixed_mask=torch.tensor([[False, True, True]]),
            init_plan=torch.tensor([[4, 1, 1]]),
        )
    )

    assert td["current_plan"].tolist() == [[4, -1, -1]]
    assert not td["parcel_action_mask"][0, 0]
    assert not td["action_mask"].view(1, env.num_types, 3)[0, :, 0].any()


def test_luop_step_rejects_action_for_fixed_parcel():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 1},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(
        _instance(
            [0.4, 0.3, 0.3],
            fixed_mask=torch.tensor([[False, True, True]]),
            init_plan=torch.tensor([[4, 1, 1]]),
        )
    )

    td.set("action", env.encode_action(torch.tensor([2]), torch.tensor([0]), 3))

    with pytest.raises(ValueError, match="infeasible LUOP joint action selected"):
        env.step(td)


def test_luop_step_rejects_constraint_infeasible_type_parcel_pair():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0.5, 0.4, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.5, 0.3, 0.2]))
    td.set("action", env.encode_action(torch.tensor([0]), torch.tensor([0]), 3))
    td = env.step(td)["next"]

    td.set("action", env.encode_action(torch.tensor([0]), torch.tensor([1]), 3))

    with pytest.raises(ValueError, match="infeasible LUOP joint action selected"):
        env.step(td)


def test_luop_reset_rejects_active_state_with_no_valid_joint_action():
    env = landuseOptEnv(
        generator_params={"num_loc": 2, "num_fixed": 0},
        min_type_ratios=[0.6, 0.6, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )

    with pytest.raises(ValueError, match="no valid LUOP joint action"):
        env.reset(_instance([0.5, 0.5]))


def test_luop_step_rejects_active_state_that_loses_all_valid_joint_actions():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0.4, 0.4, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.4, 0.4, 0.2]))
    td.set("type_action", torch.tensor([2]))
    td.set("parcel_action", torch.tensor([0]))
    td.set("action_mask", torch.ones_like(td["action_mask"]))

    with pytest.raises(ValueError, match="no valid LUOP joint action"):
        env.step(td)


def test_luop_step_rejects_masked_joint_action_with_value_error():
    env = landuseOptEnv(
        generator_params={"num_loc": 2, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.5, 0.5]))
    td.set("action", torch.tensor([0]))
    td["action_mask"][0, 0] = False

    with pytest.raises(ValueError, match="infeasible LUOP joint action selected"):
        env.step(td)


def test_luop_objective_variants_share_joint_action_surface():
    for env_cls in (landuseOptNearestEnv, landuseOptCompatibilityEnv, MAlanduseOptEnv):
        env = env_cls(
            generator_params={"num_loc": 4, "num_fixed": 0},
            min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
            check_solution=False,
        )
        td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))

        assert td["action_mask"].shape == (1, env.num_types * 4)
        assert td["parcel_action_mask"].shape == (1, 4)
        assert td["type_action_mask"].shape == (1, env.num_types)

        td.set("action", env.encode_action(torch.tensor([3]), torch.tensor([2]), 4))
        td = env.step(td)["next"]

        assert td["selected_type"].item() == 3
        assert td["current_node"].item() == 2
        assert td["current_plan"][0, 2].item() == 3


def test_luop_objective_variants_run_attention_policy_with_luop_embeddings():
    for env_cls in (landuseOptNearestEnv, landuseOptCompatibilityEnv, MAlanduseOptEnv):
        env = env_cls(
            generator_params={"num_loc": 4, "num_fixed": 0},
            min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
            check_solution=False,
        )
        td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
        policy = AttentionModelPolicy(
            env_name=env.name,
            embed_dim=32,
            num_encoder_layers=1,
            num_heads=4,
            feedforward_hidden=64,
        )

        out = policy(
            td,
            env,
            phase="test",
            decode_type="greedy",
            return_actions=True,
            return_plan=True,
        )

        assert out["actions"].max().item() < env.num_types * 4
        assert out["type_actions"].shape == out["actions"].shape
        assert out["parcel_actions"].shape == out["actions"].shape
        assert torch.isfinite(out["reward"]).all()


def test_luop_reward_components_are_scalarized_by_objective_weights():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        objective_weights=[0.25, 0.75],
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td["current_plan"] = torch.tensor([[0, 1, 4, 7]])
    td["objective_weights"] = torch.tensor([[0.25, 0.75]], dtype=torch.float32)

    reward = env.get_reward(td, torch.empty(1, 0, dtype=torch.long))

    assert td["reward_components"].shape == (1, 2)
    assert torch.allclose(
        reward,
        (td["reward_components"] * td["objective_weights"]).sum(dim=-1),
        atol=1e-6,
    )


def test_luop_reward_components_support_shaped_batch_for_objective_variants():
    batch_shape = (2, 3)
    num_loc = 4
    batch = TensorDict(
        {
            "locs": torch.rand(*batch_shape, num_loc, 2),
            "areas": torch.full((*batch_shape, num_loc), 0.25),
            "init_plan": torch.ones(*batch_shape, num_loc, dtype=torch.long),
            "fixed_mask": torch.ones(*batch_shape, num_loc, dtype=torch.bool),
        },
        batch_size=batch_shape,
    )
    current_plan = torch.tensor(
        [
            [[0, 1, 4, 7], [1, 3, 5, 6], [2, 4, 6, 7]],
            [[7, 4, 1, 0], [6, 5, 3, 1], [4, 2, 7, 6]],
        ],
        dtype=torch.long,
    )
    objective_weights = torch.tensor(
        [
            [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]],
            [[0.75, 0.25], [0.5, 0.5], [0.25, 0.75]],
        ],
        dtype=torch.float32,
    )

    for env_cls in (landuseOptEnv, landuseOptNearestEnv, landuseOptCompatibilityEnv):
        env = env_cls(
            generator_params={"num_loc": num_loc, "num_fixed": 0},
            objective_weights=[0.25, 0.75],
            check_solution=False,
        )
        td = env.reset(batch.clone())
        td["current_plan"] = current_plan
        td["objective_weights"] = objective_weights

        reward = env.get_reward(td, torch.empty(*batch_shape, 0, dtype=torch.long))

        assert reward.shape == batch_shape
        assert td["reward_components"].shape == (*batch_shape, 2)
        assert torch.isfinite(reward).all()
        assert torch.allclose(
            reward,
            (td["reward_components"] * td["objective_weights"]).sum(dim=-1),
            atol=1e-6,
        )


def test_luop_reward_components_support_shaped_batch_adjacency_matrix_variants():
    batch_shape = (2, 3)
    num_loc = 4
    locs = torch.rand(*batch_shape, num_loc, 2)
    adjacency_matrix = torch.ones(
        *batch_shape,
        num_loc,
        num_loc,
        dtype=torch.bool,
    )
    adjacency_matrix.diagonal(dim1=-2, dim2=-1).zero_()
    batch = TensorDict(
        {
            "locs": locs,
            "areas": torch.full((*batch_shape, num_loc), 0.25),
            "init_plan": torch.ones(*batch_shape, num_loc, dtype=torch.long),
            "fixed_mask": torch.ones(*batch_shape, num_loc, dtype=torch.bool),
            "adjacency_list": adjacency_matrix,
            "distances": torch.cdist(locs, locs),
        },
        batch_size=batch_shape,
    )
    current_plan = torch.tensor(
        [
            [[0, 1, 4, 7], [1, 3, 5, 6], [2, 4, 6, 7]],
            [[7, 4, 1, 0], [6, 5, 3, 1], [4, 2, 7, 6]],
        ],
        dtype=torch.long,
    )

    for env_cls in (landuseOptEnv, landuseOptNearestEnv, landuseOptCompatibilityEnv):
        env = env_cls(
            generator_params={"num_loc": num_loc, "num_fixed": 0},
            objective_weights=[0.25, 0.75],
            check_solution=False,
        )
        td = env.reset(batch.clone())
        td["current_plan"] = current_plan

        reward = env.get_reward(td, torch.empty(*batch_shape, 0, dtype=torch.long))

        assert reward.shape == batch_shape
        assert td["reward_components"].shape == (*batch_shape, 2)
        assert torch.isfinite(reward).all()


def test_luop_average_accessibility_uses_exact_pair_count():
    plan = torch.tensor([[1, 0]], dtype=torch.long)
    distances = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]], dtype=torch.float32)

    accessibility = cityplan_init.calInterAccessibility_Average_tensor(
        plan,
        ["Residential"],
        ["Commercial"],
        distances,
    )

    assert torch.allclose(accessibility, torch.tensor([-2.0]))


def test_luop_reward_supports_weighted_chebyshev_scalarization():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        objective_weights=[0.25, 0.75],
        objective_scalarization="chebyshev",
        objective_ideal=[1.0, 1.0],
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td["current_plan"] = torch.tensor([[0, 1, 4, 7]])
    td["objective_weights"] = torch.tensor([[0.25, 0.75]], dtype=torch.float32)

    reward = env.get_reward(td, torch.empty(1, 0, dtype=torch.long))

    expected = -torch.max(
        td["objective_weights"] * (torch.tensor([[1.0, 1.0]]) - td["reward_components"]),
        dim=-1,
    ).values
    assert torch.allclose(reward, expected, atol=1e-6)


def test_luop_reward_rejects_unknown_scalarization():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        objective_scalarization="mystery",
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td["current_plan"] = torch.tensor([[0, 1, 4, 7]])

    with pytest.raises(ValueError, match="objective_scalarization"):
        env.get_reward(td, torch.empty(1, 0, dtype=torch.long))


def test_luop_rejects_degenerate_fixed_objective_weights():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        objective_weights=[0.0, 0.0],
        check_solution=False,
    )

    with pytest.raises(ValueError, match="objective_weights"):
        env.reset(_instance([0.25, 0.25, 0.25, 0.25]))


def test_luop_reset_rejects_degenerate_batch_objective_weights():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = _instance([0.25, 0.25, 0.25, 0.25])
    batch["objective_weights"] = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="objective_weights"):
        env.reset(batch)


def test_luop_reset_broadcasts_single_objective_weight_vector_to_batch():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 4).view(1, 4, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.full((2, 4), 0.25, dtype=torch.float32),
            "init_plan": torch.ones(2, 4, dtype=torch.long),
            "fixed_mask": torch.ones(2, 4, dtype=torch.bool),
            "objective_weights": torch.tensor([0.25, 0.75], dtype=torch.float32),
        },
        batch_size=[2],
    )

    td = env.reset(batch)

    assert td["objective_weights"].shape == (2, 2)
    assert torch.allclose(
        td["objective_weights"],
        torch.tensor([[0.25, 0.75], [0.25, 0.75]], dtype=torch.float32),
    )


def test_luop_specs_include_dual_action_and_multiobjective_runtime_fields():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )

    for key in (
        "current_type",
        "selected_type",
        "parcel_action_mask",
        "type_action_mask",
        "type_parcel_action_mask",
        "action_mask",
        "objective_weights",
        "target_ratios",
        "constraint_ratios",
        "constraint_pressure",
    ):
        assert key in env.observation_spec.keys()

    for key in ("type_action", "parcel_action", "reward_components"):
        assert key in env.observation_spec.keys()


def test_luop_nearest_reward_broadcasts_single_objective_weight_vector_to_batch():
    env = landuseOptNearestEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 4).view(1, 4, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.full((2, 4), 0.25, dtype=torch.float32),
            "init_plan": torch.ones(2, 4, dtype=torch.long),
            "fixed_mask": torch.ones(2, 4, dtype=torch.bool),
        },
        batch_size=[2],
    )
    td = env.reset(batch)
    td["current_plan"] = torch.tensor(
        [[0, 1, 4, 7], [7, 4, 1, 0]],
        dtype=torch.long,
    )
    td["objective_weights"] = torch.tensor([0.25, 0.75], dtype=torch.float32)

    reward = env.get_reward(td, torch.empty(2, 0, dtype=torch.long))

    assert td["objective_weights"].shape == (2, 2)
    assert torch.allclose(
        td["objective_weights"],
        torch.tensor([[0.25, 0.75], [0.25, 0.75]], dtype=torch.float32),
    )
    assert torch.allclose(
        reward,
        (td["reward_components"] * td["objective_weights"]).sum(dim=-1),
        atol=1e-6,
    )


def test_luop_compatibility_reset_defaults_to_compatibility_objective_weight():
    env = landuseOptCompatibilityEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )

    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))

    assert torch.allclose(
        td["objective_weights"],
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )


def test_luop_compatibility_reward_preserves_runtime_objective_weights_and_scalarizes():
    env = landuseOptCompatibilityEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 4).view(1, 4, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.full((2, 4), 0.25, dtype=torch.float32),
            "init_plan": torch.ones(2, 4, dtype=torch.long),
            "fixed_mask": torch.ones(2, 4, dtype=torch.bool),
        },
        batch_size=[2],
    )
    td = env.reset(batch)
    td["current_plan"] = torch.tensor(
        [[0, 1, 4, 7], [7, 4, 1, 0]],
        dtype=torch.long,
    )
    td["objective_weights"] = torch.tensor([0.25, 0.75], dtype=torch.float32)

    reward = env.get_reward(td, torch.empty(2, 0, dtype=torch.long))

    expected_weights = torch.tensor(
        [[0.25, 0.75], [0.25, 0.75]],
        dtype=torch.float32,
    )
    assert torch.allclose(td["objective_weights"], expected_weights)
    assert torch.allclose(
        reward,
        (td["reward_components"] * expected_weights).sum(dim=-1),
        atol=1e-6,
    )


def test_luop_reward_rejects_degenerate_runtime_objective_weights():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td["current_plan"] = torch.tensor([[0, 1, 4, 7]])
    td["objective_weights"] = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="objective_weights"):
        env.get_reward(td, torch.empty(1, 0, dtype=torch.long))


def test_luop_attention_policy_decodes_flat_joint_actions_and_returns_components():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
    )

    out = policy(
        td,
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"].dim() == 2
    assert out["actions"].max().item() < env.num_types * 4
    assert out["parcel_actions"].shape == out["actions"].shape
    assert out["type_actions"].shape == out["actions"].shape
    assert out["current_plan"].min().item() >= 0
    assert out["reward_components"].shape == (1, 2)


def test_luop_attention_policy_treats_hydra_luop_name_as_luop_family():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = AttentionModelPolicy(
        env_name="luop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )
    td = env.reset(env.generator(batch_size=[2]))

    out = policy(
        td,
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"].shape == (2, 4)
    assert out["type_actions"].shape == out["actions"].shape
    assert out["parcel_actions"].shape == out["actions"].shape
    assert out["current_plan"].min().item() >= 0
    assert torch.isfinite(out["reward"]).all()


def test_luop_attention_policy_can_decode_type_first_dual_actions():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    out = policy(
        td,
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
    )

    encoded = env.encode_action(out["type_actions"], out["parcel_actions"], 4)
    assert torch.equal(out["actions"], encoded)
    assert "type_log_likelihood" in out
    assert "parcel_log_likelihood" in out
    assert torch.allclose(
        out["log_likelihood"],
        out["type_log_likelihood"] + out["parcel_log_likelihood"],
        atol=1e-6,
    )
    assert out["current_plan"].min().item() >= 0
    assert torch.isfinite(out["reward"]).all()


def test_luop_flat_multistart_select_best_masks_collapsed_likelihood():
    env = landuseOptEnv(
        generator_params={"num_loc": 6, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(env.generator(batch_size=[2]))
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=False,
    )

    out = policy(
        td,
        env,
        phase="test",
        decode_type="multistart_greedy",
        select_best=True,
        return_actions=True,
        return_entropy=True,
        return_plan=True,
    )

    assert out["actions"].shape == (2, 6)
    assert out["log_likelihood"].shape == (2,)
    assert out["entropy"].shape == (2,)
    assert torch.isfinite(out["log_likelihood"]).all()
    assert torch.isfinite(out["entropy"]).all()
    assert out["current_plan"].min().item() >= 0
    assert torch.isfinite(out["reward"]).all()


def test_luop_attention_policy_type_first_supports_shaped_batch():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch_shape = (2, 3)
    num_loc = 4
    batch = TensorDict(
        {
            "locs": torch.rand(*batch_shape, num_loc, 2),
            "areas": torch.full((*batch_shape, num_loc), 0.25),
            "init_plan": torch.ones(*batch_shape, num_loc, dtype=torch.long),
            "fixed_mask": torch.ones(*batch_shape, num_loc, dtype=torch.bool),
        },
        batch_size=batch_shape,
    )
    td = env.reset(batch)
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    out = policy(
        td,
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"].shape == (*batch_shape, num_loc)
    assert out["type_actions"].shape == out["actions"].shape
    assert out["parcel_actions"].shape == out["actions"].shape
    assert out["current_plan"].shape == (*batch_shape, num_loc)
    assert out["reward_components"].shape == (*batch_shape, 2)
    assert torch.equal(
        out["actions"],
        env.encode_action(out["type_actions"], out["parcel_actions"], num_loc),
    )
    assert out["current_plan"].min().item() >= 0
    assert torch.isfinite(out["reward"]).all()


def test_luop_attention_policy_type_first_restores_shaped_batch_artifacts():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch_shape = (2, 3)
    num_loc = 4
    batch = TensorDict(
        {
            "locs": torch.rand(*batch_shape, num_loc, 2),
            "areas": torch.full((*batch_shape, num_loc), 0.25),
            "init_plan": torch.ones(*batch_shape, num_loc, dtype=torch.long),
            "fixed_mask": torch.ones(*batch_shape, num_loc, dtype=torch.bool),
        },
        batch_size=batch_shape,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    out = policy(
        env.reset(batch),
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_hidden=True,
        return_init_embeds=True,
    )

    assert out["init_embeds"].shape[:3] == (*batch_shape, num_loc)
    assert out["hidden"].node_embeddings.shape[:3] == (*batch_shape, num_loc)
    assert out["hidden"].graph_context.shape[:2] == batch_shape


def test_luop_attention_policy_type_first_replays_shaped_batch_dual_actions():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch_shape = (2, 3)
    num_loc = 4
    batch = TensorDict(
        {
            "locs": torch.rand(*batch_shape, num_loc, 2),
            "areas": torch.full((*batch_shape, num_loc), 0.25),
            "init_plan": torch.ones(*batch_shape, num_loc, dtype=torch.long),
            "fixed_mask": torch.ones(*batch_shape, num_loc, dtype=torch.bool),
        },
        batch_size=batch_shape,
    )
    td = env.reset(batch)
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )
    type_actions = torch.tensor(
        [
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
            [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 0]],
        ],
        dtype=torch.long,
    )
    parcel_actions = torch.tensor(
        [
            [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]],
            [[3, 0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 0]],
        ],
        dtype=torch.long,
    )

    out = policy(
        td,
        env,
        phase="test",
        type_actions=type_actions,
        parcel_actions=parcel_actions,
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"].shape == (*batch_shape, num_loc)
    assert torch.equal(out["type_actions"], type_actions)
    assert torch.equal(out["parcel_actions"], parcel_actions)
    assert torch.equal(
        out["actions"],
        env.encode_action(type_actions, parcel_actions, num_loc),
    )
    assert out["current_plan"].shape == (*batch_shape, num_loc)
    assert out["current_plan"].min().item() >= 0


def test_luop_shaped_batch_replay_rejects_flattened_dual_action_shape():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch_shape = (2, 3)
    num_loc = 4
    batch = TensorDict(
        {
            "locs": torch.rand(*batch_shape, num_loc, 2),
            "areas": torch.full((*batch_shape, num_loc), 0.25),
            "init_plan": torch.ones(*batch_shape, num_loc, dtype=torch.long),
            "fixed_mask": torch.ones(*batch_shape, num_loc, dtype=torch.bool),
        },
        batch_size=batch_shape,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="type_actions shape"):
        policy(
            env.reset(batch),
            env,
            phase="test",
            type_actions=torch.zeros(6, num_loc, dtype=torch.long),
            parcel_actions=torch.arange(num_loc).expand(6, num_loc),
        )


def test_luop_shaped_batch_replay_rejects_flattened_flat_action_shape():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch_shape = (2, 3)
    num_loc = 4
    batch = TensorDict(
        {
            "locs": torch.rand(*batch_shape, num_loc, 2),
            "areas": torch.full((*batch_shape, num_loc), 0.25),
            "init_plan": torch.ones(*batch_shape, num_loc, dtype=torch.long),
            "fixed_mask": torch.ones(*batch_shape, num_loc, dtype=torch.bool),
        },
        batch_size=batch_shape,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="actions shape"):
        policy(
            env.reset(batch),
            env,
            phase="test",
            actions=torch.zeros(6, num_loc, dtype=torch.long),
        )


def test_luop_attention_policy_broadcasts_runtime_objective_vector_for_batched_rollout():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 4).view(1, 4, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.full((2, 4), 0.25, dtype=torch.float32),
            "init_plan": torch.ones(2, 4, dtype=torch.long),
            "fixed_mask": torch.ones(2, 4, dtype=torch.bool),
        },
        batch_size=[2],
    )
    td = env.reset(batch)
    td["objective_weights"] = torch.tensor([0.25, 0.75], dtype=torch.float32)
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    out = policy(
        td,
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"].shape == (2, 4)
    assert out["type_actions"].shape == out["actions"].shape
    assert out["parcel_actions"].shape == out["actions"].shape
    assert out["current_plan"].min().item() >= 0
    assert torch.isfinite(out["reward"]).all()


def test_luop_attention_decoder_is_scale_invariant_to_equivalent_objective_weights():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 4).view(1, 4, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.full((2, 4), 0.25, dtype=torch.float32),
            "init_plan": torch.ones(2, 4, dtype=torch.long),
            "fixed_mask": torch.ones(2, 4, dtype=torch.bool),
        },
        batch_size=[2],
    )
    td_a = env.reset(batch.clone())
    td_b = env.reset(batch.clone())
    td_a["objective_weights"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    td_b["objective_weights"] = torch.tensor([10.0, 20.0], dtype=torch.float32)
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )
    policy.eval()

    hidden_a, _ = policy.encoder(td_a)
    _, _, cache_a = policy.decoder.pre_decoder_hook(td_a, env, hidden_a, 0)
    logits_a, mask_a = policy.decoder(td_a, cache_a, 0)

    hidden_b, _ = policy.encoder(td_b)
    _, _, cache_b = policy.decoder.pre_decoder_hook(td_b, env, hidden_b, 0)
    logits_b, mask_b = policy.decoder(td_b, cache_b, 0)

    assert torch.equal(mask_a, mask_b)
    assert torch.allclose(logits_a, logits_b, atol=1e-6)


def test_luop_attention_policy_rejects_degenerate_runtime_objective_weights():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td["objective_weights"] = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="objective_weights"):
        policy(
            td,
            env,
            phase="test",
            decode_type="greedy",
            return_actions=True,
        )


def test_luop_attention_policy_rejects_negative_runtime_objective_weights():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td["objective_weights"] = torch.tensor([[1.0, -0.5]], dtype=torch.float32)
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="objective_weights"):
        policy(
            td,
            env,
            phase="test",
            decode_type="greedy",
            return_actions=True,
        )


def test_luop_attention_policy_rejects_runtime_objective_weights_with_wrong_count():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td["objective_weights"] = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="objective_weights"):
        policy(
            td,
            env,
            phase="test",
            decode_type="greedy",
            return_actions=True,
        )


def test_luop_attention_policy_rejects_runtime_objective_weights_batch_mismatch():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 4).view(1, 4, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.full((2, 4), 0.25, dtype=torch.float32),
            "init_plan": torch.ones(2, 4, dtype=torch.long),
            "fixed_mask": torch.ones(2, 4, dtype=torch.bool),
        },
        batch_size=[2],
    )
    td = env.reset(batch)
    td["objective_weights"] = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="objective_weights batch shape"):
        policy(
            td,
            env,
            phase="test",
            decode_type="greedy",
            return_actions=True,
        )


def test_luop_type_first_policy_handles_already_done_all_fixed_instance():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 3},
        check_solution=False,
    )
    td = env.reset(
        _instance(
            [0.4, 0.3, 0.3],
            fixed_mask=torch.tensor([[False, False, False]]),
            init_plan=torch.tensor([[0, 1, 2]]),
        )
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    out = policy(
        td,
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"].shape == (1, 0)
    assert out["type_actions"].shape == (1, 0)
    assert out["parcel_actions"].shape == (1, 0)
    assert out["current_plan"].tolist() == [[0, 1, 2]]
    assert torch.isfinite(out["reward"]).all()


def test_luop_flat_policy_handles_already_done_all_fixed_instance():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 3},
        check_solution=False,
    )
    td = env.reset(
        _instance(
            [0.4, 0.3, 0.3],
            fixed_mask=torch.tensor([[False, False, False]]),
            init_plan=torch.tensor([[0, 1, 2]]),
        )
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
    )

    out = policy(
        td,
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"].shape == (1, 0)
    assert out["type_actions"].shape == (1, 0)
    assert out["parcel_actions"].shape == (1, 0)
    assert out["current_plan"].tolist() == [[0, 1, 2]]
    assert torch.isfinite(out["reward"]).all()


def test_luop_mixed_done_batch_has_safe_dummy_masks_and_finite_type_first_rollout():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor(
                [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]], dtype=torch.float32
            ),
            "init_plan": torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.long),
            "fixed_mask": torch.tensor(
                [[False, False, False], [True, True, True]], dtype=torch.bool
            ),
        },
        batch_size=[2],
    )
    td = env.reset(batch)

    assert td["done"].squeeze(-1).tolist() == [True, False]
    assert td["action_mask"][0].tolist() == [True] + [False] * 23
    assert td["parcel_action_mask"][0].tolist() == [True, False, False]
    assert td["type_action_mask"][0].tolist() == [True] + [False] * 7
    assert td["type_parcel_action_mask"][0, 0, 0]

    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    out = policy(
        td,
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
    )

    assert out["current_plan"][0].tolist() == [0, 1, 2]
    assert out["current_plan"][1].min().item() >= 0
    assert torch.isfinite(out["reward"]).all()

    flat_policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
    )
    flat_out = flat_policy(
        td.clone(),
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
    )

    assert flat_out["current_plan"][0].tolist() == [0, 1, 2]
    assert flat_out["current_plan"][1].min().item() >= 0
    assert torch.isfinite(flat_out["reward"]).all()


def test_luop_attention_policy_evaluates_explicit_dual_actions():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )
    type_actions = torch.tensor([[3, 2, 1, 0]])
    parcel_actions = torch.tensor([[2, 1, 0, 3]])
    flat_actions = env.encode_action(type_actions, parcel_actions, 4)

    replayed = policy(
        td,
        env,
        phase="test",
        type_actions=type_actions,
        parcel_actions=parcel_actions,
        return_actions=True,
        return_plan=True,
    )

    assert torch.equal(replayed["type_actions"], type_actions)
    assert torch.equal(replayed["parcel_actions"], parcel_actions)
    assert torch.equal(replayed["actions"], flat_actions)
    assert torch.isfinite(replayed["log_likelihood"]).all()
    assert replayed["current_plan"].min().item() >= 0


def test_luop_attention_policy_rejects_incomplete_or_mixed_dual_action_replay():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 2, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )
    td = env.reset(_instance([0.5, 0.5]))

    with pytest.raises(ValueError, match="Both type_actions and parcel_actions"):
        policy(td, env, type_actions=torch.tensor([[0, 1]]))

    td = env.reset(_instance([0.5, 0.5]))
    with pytest.raises(ValueError, match="Pass either flat actions"):
        policy(
            td,
            env,
            actions=torch.tensor([[0, 3]]),
            type_actions=torch.tensor([[0, 1]]),
            parcel_actions=torch.tensor([[0, 1]]),
        )


def test_luop_policy_rejects_dual_replay_action_length_mismatch():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="LUOP replay action length"):
        policy(
            env.reset(_instance([0.4, 0.3, 0.3])),
            env,
            type_actions=torch.tensor([[0, 1]]),
            parcel_actions=torch.tensor([[0, 1]]),
        )

    with pytest.raises(ValueError, match="LUOP replay action length"):
        policy(
            env.reset(_instance([0.4, 0.3, 0.3])),
            env,
            type_actions=torch.tensor([[0, 1, 2, 3]]),
            parcel_actions=torch.tensor([[0, 1, 2, 0]]),
        )


def test_luop_policy_rejects_flat_replay_action_length_mismatch():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
    )

    with pytest.raises(ValueError, match="LUOP replay action length"):
        policy(
            env.reset(_instance([0.4, 0.3, 0.3])),
            env,
            actions=torch.tensor([[0, 4]]),
        )

    with pytest.raises(ValueError, match="LUOP replay action length"):
        policy(
            env.reset(_instance([0.4, 0.3, 0.3])),
            env,
            actions=torch.tensor([[0, 4, 8, 3]]),
        )


def test_luop_type_first_backward_is_finite_when_some_types_have_no_parcels():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class LearnableDecoder(ConstructiveDecoder):
        def __init__(self, num_actions):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.linspace(-0.2, 0.2, num_actions))

        def forward(self, td, hidden=None, num_starts=0):
            return (
                self.logits.to(td.device).expand_as(td["action_mask"]),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        tios=[0.6, 0.4, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=LearnableDecoder(env.num_types * 3),
        env_name="lop",
        decode_type_first=True,
    )
    td = env.reset(_instance([0.4, 0.3, 0.3]))

    assert not td["type_action_mask"].all()

    out = policy(
        td,
        env,
        phase="train",
        decode_type="greedy",
    )
    loss = -out["log_likelihood"].mean()
    loss.backward()

    assert torch.isfinite(policy.decoder.logits.grad).all()


def test_luop_policy_rejects_out_of_range_dual_replay_actions():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="LUOP replay type_actions"):
        policy(
            env.reset(_instance([0.4, 0.3, 0.3])),
            env,
            type_actions=torch.tensor([[0, 8, 1]]),
            parcel_actions=torch.tensor([[0, 1, 2]]),
        )

    with pytest.raises(ValueError, match="LUOP replay parcel_actions"):
        policy(
            env.reset(_instance([0.4, 0.3, 0.3])),
            env,
            type_actions=torch.tensor([[0, 1, 2]]),
            parcel_actions=torch.tensor([[0, 3, 1]]),
        )


def test_luop_policy_rejects_non_integer_dual_replay_actions():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="integer"):
        policy(
            env.reset(_instance([0.4, 0.3, 0.3])),
            env,
            type_actions=torch.tensor([[0.0, 1.5, 2.0]]),
            parcel_actions=torch.tensor([[0.0, 1.0, 2.0]]),
        )


def test_luop_policy_rejects_negative_one_active_dual_replay_actions():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="LUOP replay type_actions"):
        policy(
            env.reset(_instance([0.4, 0.3, 0.3])),
            env,
            type_actions=torch.tensor([[0, -1, 2]]),
            parcel_actions=torch.tensor([[0, -1, 1]]),
        )


def test_luop_policy_rejects_out_of_range_flat_replay_actions():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
    )

    with pytest.raises(ValueError, match="LUOP replay actions"):
        policy(
            env.reset(_instance([0.4, 0.3, 0.3])),
            env,
            actions=torch.tensor([[0, env.num_types * 3, 1]]),
        )


def test_luop_policy_rejects_non_integer_flat_replay_actions():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
    )

    with pytest.raises(ValueError, match="integer"):
        policy(
            env.reset(_instance([0.4, 0.3, 0.3])),
            env,
            actions=torch.tensor([[0.0, 4.5, 8.0]]),
        )


def test_luop_policy_rejects_negative_one_active_flat_replay_actions():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
    )

    with pytest.raises(ValueError, match="LUOP replay actions"):
        policy(
            env.reset(_instance([0.4, 0.3, 0.3])),
            env,
            actions=torch.tensor([[0, -1, 1]]),
        )


def test_luop_policy_rejects_infeasible_flat_replay_actions_before_env_assert():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
    )

    repeated_parcel = torch.tensor([[0, 3, 7]])

    with pytest.raises(ValueError, match="LUOP replay actions infeasible at step 1"):
        policy(env.reset(_instance([0.4, 0.3, 0.3])), env, actions=repeated_parcel)


def test_luop_type_first_policy_rejects_infeasible_dual_replay_parcel_action():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    with pytest.raises(
        ValueError, match="LUOP replay parcel_actions infeasible at step 1"
    ):
        policy(
            env.reset(_instance([0.4, 0.3, 0.3])),
            env,
            type_actions=torch.tensor([[0, 1, 2]]),
            parcel_actions=torch.tensor([[0, 0, 1]]),
        )


def test_luop_type_first_dual_replay_ignores_done_row_padding_actions():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor(
                [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
                dtype=torch.float32,
            ),
            "init_plan": torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.long),
            "fixed_mask": torch.tensor(
                [[False, False, True], [False, True, True]],
                dtype=torch.bool,
            ),
        },
        batch_size=[2],
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    out = policy(
        env.reset(batch),
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
        type_actions=torch.tensor([[1, 7], [2, 3]]),
        parcel_actions=torch.tensor([[2, 0], [1, 2]]),
    )

    assert out["current_plan"][0].tolist() == [2, 3, 1]
    assert out["current_plan"][1].tolist() == [2, 2, 3]


def test_luop_type_first_dual_replay_accepts_negative_one_done_row_padding():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor(
                [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
                dtype=torch.float32,
            ),
            "init_plan": torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.long),
            "fixed_mask": torch.tensor(
                [[False, False, True], [False, True, True]],
                dtype=torch.bool,
            ),
        },
        batch_size=[2],
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    out = policy(
        env.reset(batch),
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
        type_actions=torch.tensor([[1, -1], [2, 3]]),
        parcel_actions=torch.tensor([[2, -1], [1, 2]]),
    )

    assert out["current_plan"][0].tolist() == [2, 3, 1]
    assert out["current_plan"][1].tolist() == [2, 2, 3]
    assert out["actions"][0, 1].item() == -1
    assert out["type_actions"][0, 1].item() == -1
    assert out["parcel_actions"][0, 1].item() == -1


def test_luop_type_first_dual_replay_masks_done_row_padding_log_likelihood():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class BiasedDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            logits = torch.arange(
                td["action_mask"].size(-1),
                dtype=torch.float32,
                device=td.device,
            ).expand_as(td["action_mask"])
            return logits, td["action_mask"]

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor(
                [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
                dtype=torch.float32,
            ),
            "init_plan": torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.long),
            "fixed_mask": torch.tensor(
                [[False, False, True], [False, True, True]],
                dtype=torch.bool,
            ),
        },
        batch_size=[2],
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=BiasedDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    out_short = policy(
        env.reset(batch.clone()),
        env,
        phase="test",
        decode_type="greedy",
        return_sum_log_likelihood=False,
        type_actions=torch.tensor([[1, 0], [2, 3]]),
        parcel_actions=torch.tensor([[2, 0], [1, 2]]),
    )
    out_padded = policy(
        env.reset(batch.clone()),
        env,
        phase="test",
        decode_type="greedy",
        return_sum_log_likelihood=False,
        type_actions=torch.tensor([[1, 7], [2, 3]]),
        parcel_actions=torch.tensor([[2, 0], [1, 2]]),
    )

    assert out_padded["type_log_likelihood"][0, 1].item() == 0
    assert out_padded["parcel_log_likelihood"][0, 1].item() == 0
    assert torch.allclose(
        out_short["log_likelihood"],
        out_padded["log_likelihood"],
        atol=1e-6,
    )


def test_luop_type_first_dual_replay_masks_done_row_padding_entropy():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    mixed_batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor(
                [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
                dtype=torch.float32,
            ),
            "init_plan": torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.long),
            "fixed_mask": torch.tensor(
                [[False, False, True], [False, True, True]],
                dtype=torch.bool,
            ),
        },
        batch_size=[2],
    )
    single_batch = mixed_batch[0].unsqueeze(0)
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    single = policy(
        env.reset(single_batch.clone()),
        env,
        phase="test",
        return_entropy=True,
        type_actions=torch.tensor([[1]]),
        parcel_actions=torch.tensor([[2]]),
        mask_logits=False,
    )
    mixed = policy(
        env.reset(mixed_batch.clone()),
        env,
        phase="test",
        return_entropy=True,
        type_actions=torch.tensor([[1, 7], [2, 3]]),
        parcel_actions=torch.tensor([[2, 0], [1, 2]]),
        mask_logits=False,
    )

    assert torch.allclose(mixed["entropy"][0], single["entropy"][0], atol=1e-6)


def test_luop_flat_replay_ignores_done_row_padding_actions():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor(
                [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
                dtype=torch.float32,
            ),
            "init_plan": torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.long),
            "fixed_mask": torch.tensor(
                [[False, False, True], [False, True, True]],
                dtype=torch.bool,
            ),
        },
        batch_size=[2],
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
    )
    actions = env.encode_action(
        torch.tensor([[1, 7], [2, 3]]),
        torch.tensor([[2, 0], [1, 2]]),
        num_loc=3,
    )

    out = policy(
        env.reset(batch),
        env,
        phase="test",
        return_actions=True,
        return_plan=True,
        actions=actions,
    )

    assert out["current_plan"][0].tolist() == [2, 3, 1]
    assert out["current_plan"][1].tolist() == [2, 2, 3]


def test_luop_flat_replay_accepts_negative_one_done_row_padding():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor(
                [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
                dtype=torch.float32,
            ),
            "init_plan": torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.long),
            "fixed_mask": torch.tensor(
                [[False, False, True], [False, True, True]],
                dtype=torch.bool,
            ),
        },
        batch_size=[2],
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
    )
    actions = env.encode_action(
        torch.tensor([[1, 0], [2, 3]]),
        torch.tensor([[2, 0], [1, 2]]),
        num_loc=3,
    )
    actions[0, 1] = -1

    out = policy(
        env.reset(batch),
        env,
        phase="test",
        return_actions=True,
        return_plan=True,
        actions=actions,
    )

    assert out["current_plan"][0].tolist() == [2, 3, 1]
    assert out["current_plan"][1].tolist() == [2, 2, 3]
    assert out["actions"][0, 1].item() == -1
    assert out["type_actions"][0, 1].item() == -1
    assert out["parcel_actions"][0, 1].item() == -1


def test_luop_type_first_output_uses_negative_one_for_done_row_padding():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor(
                [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
                dtype=torch.float32,
            ),
            "init_plan": torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.long),
            "fixed_mask": torch.tensor(
                [[False, False, True], [False, True, True]],
                dtype=torch.bool,
            ),
        },
        batch_size=[2],
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    out = policy(
        env.reset(batch),
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
    )

    assert out["current_plan"][0].ge(0).all()
    assert out["current_plan"][1].ge(0).all()
    assert out["actions"][0, 1].item() == -1
    assert out["type_actions"][0, 1].item() == -1
    assert out["parcel_actions"][0, 1].item() == -1


def test_luop_flat_output_uses_negative_one_for_done_row_padding():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor(
                [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
                dtype=torch.float32,
            ),
            "init_plan": torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.long),
            "fixed_mask": torch.tensor(
                [[False, False, True], [False, True, True]],
                dtype=torch.bool,
            ),
        },
        batch_size=[2],
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
    )

    out = policy(
        env.reset(batch),
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
    )

    assert out["current_plan"][0].ge(0).all()
    assert out["current_plan"][1].ge(0).all()
    assert out["actions"][0, 1].item() == -1
    assert out["type_actions"][0, 1].item() == -1
    assert out["parcel_actions"][0, 1].item() == -1


def test_luop_type_first_component_guard_rejects_masked_forced_action():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="infeasible LUOP component action selected"):
        policy._decode_luop_component(
            logits=torch.zeros(1, 3),
            mask=torch.tensor([[False, True, True]]),
            decode_strategy=get_decoding_strategy("evaluate"),
            action=torch.tensor([0]),
        )


def test_luop_type_first_component_guard_enforces_mask_when_logits_unmasked():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="infeasible LUOP component action selected"):
        policy._decode_luop_component(
            logits=torch.tensor([[10.0, 0.0, 0.0]]),
            mask=torch.tensor([[False, True, True]]),
            decode_strategy=get_decoding_strategy("greedy", mask_logits=False),
        )


def test_luop_policy_rejects_flat_replay_batch_shape_mismatch():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor(
                [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]], dtype=torch.float32
            ),
            "init_plan": torch.ones(2, 3, dtype=torch.long),
            "fixed_mask": torch.ones(2, 3, dtype=torch.bool),
        },
        batch_size=[2],
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
    )

    with pytest.raises(ValueError, match="LUOP replay actions shape"):
        policy(
            env.reset(batch),
            env,
            actions=torch.tensor([[0, 4, 8]]),
        )


def test_luop_policy_rejects_dual_replay_batch_shape_mismatch():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor(
                [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]], dtype=torch.float32
            ),
            "init_plan": torch.ones(2, 3, dtype=torch.long),
            "fixed_mask": torch.ones(2, 3, dtype=torch.bool),
        },
        batch_size=[2],
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    with pytest.raises(ValueError, match="LUOP replay type_actions shape"):
        policy(
            env.reset(batch),
            env,
            type_actions=torch.tensor([[0, 1, 2]]),
            parcel_actions=torch.tensor([[0, 1, 2]]),
        )


def test_luop_type_first_parcel_decode_sees_pending_type_action():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class RecordingDecoder(ConstructiveDecoder):
        def __init__(self):
            super().__init__()
            self.pending_type_visible = []

        def forward(self, td, hidden=None, num_starts=0):
            self.pending_type_visible.append("pending_type_action" in td.keys())
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 2, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.5, 0.5]))
    decoder = RecordingDecoder()
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=decoder,
        env_name="lop",
        decode_type_first=True,
    )

    policy(td, env, phase="test", decode_type="greedy", calc_reward=False)

    assert decoder.pending_type_visible == [False, True, False, True]


def test_luop_decoder_scores_parcels_conditioned_on_candidate_type():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
    )
    hidden, _ = policy.encoder(td)
    _, _, cache = policy.decoder.pre_decoder_hook(td, env, hidden, 0)

    logits, _ = policy.decoder(td, cache, 0)
    joint_logits = logits.view(1, env.num_types, 4)
    type0_parcel_gap = joint_logits[0, 0, 0] - joint_logits[0, 0, 1]
    type1_parcel_gap = joint_logits[0, 1, 0] - joint_logits[0, 1, 1]

    assert not torch.allclose(type0_parcel_gap, type1_parcel_gap)


def test_luop_init_embedding_is_conditioned_on_objective_weights():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    td_a = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td_b = td_a.clone()
    td_a["objective_weights"] = torch.tensor([[1.0, 0.0]])
    td_b["objective_weights"] = torch.tensor([[0.0, 1.0]])
    embedding = lopInitEmbedding(embed_dim=16)

    out_a = embedding(td_a)
    out_b = embedding(td_b)

    assert out_a.shape == (1, 4, 16)
    assert not torch.allclose(out_a, out_b)


def test_luop_context_is_conditioned_on_objective_weights_and_remaining_deficit():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        tios=[0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0],
        check_solution=False,
    )
    td_a = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td_b = td_a.clone()
    td_a["objective_weights"] = torch.tensor([[1.0, 0.0]])
    td_b["objective_weights"] = torch.tensor([[0.0, 1.0]])
    embeddings = torch.randn(1, 4, 16)
    context = LOPContext(embed_dim=16)

    out_a = context(embeddings, td_a)
    out_b = context(embeddings, td_b)

    assert out_a.shape == (1, 16)
    assert not torch.allclose(out_a, out_b)


def test_luop_dynamic_embedding_is_conditioned_on_objective_weights():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    td_a = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td_b = td_a.clone()
    td_a["objective_weights"] = torch.tensor([[1.0, 0.0]])
    td_b["objective_weights"] = torch.tensor([[0.0, 1.0]])
    embedding = LOPDynamicEmbedding(embed_dim=16)

    out_a = embedding(td_a)
    out_b = embedding(td_b)

    assert out_a[0].shape == (1, 4, 16)
    assert not torch.allclose(out_a[0], out_b[0])


def test_luop_embeddings_support_shaped_batch_fallback_state_features():
    batch_shape = (2, 3)
    num_loc = 4
    td = TensorDict(
        {
            "locs": torch.rand(*batch_shape, num_loc, 2),
            "areas": torch.full((*batch_shape, num_loc), 0.25),
            "init_plan": torch.ones(*batch_shape, num_loc, dtype=torch.long),
            "current_plan": torch.full(
                (*batch_shape, num_loc),
                -1,
                dtype=torch.long,
            ),
            "current_node": torch.zeros(*batch_shape, dtype=torch.long),
            "current_types_onehot": torch.zeros(*batch_shape, 8, dtype=torch.bool),
            "i": torch.zeros(*batch_shape, 1, dtype=torch.long),
            "objective_weights": torch.tensor(
                [
                    [[2.0, 0.0], [1.0, 1.0], [0.0, 4.0]],
                    [[0.0, 4.0], [1.0, 1.0], [2.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
        },
        batch_size=batch_shape,
    )
    init_embedding = lopInitEmbedding(embed_dim=16)
    dynamic_embedding = LOPDynamicEmbedding(embed_dim=16)
    context_embedding = LOPContext(embed_dim=16)
    node_embeddings = torch.rand(*batch_shape, num_loc, 16)

    init_out = init_embedding(td)
    dynamic_out = dynamic_embedding(td)
    context_out = context_embedding(node_embeddings, td)

    assert init_out.shape == (*batch_shape, num_loc, 16)
    assert dynamic_out[0].shape == (*batch_shape, num_loc, 16)
    assert dynamic_out[1].shape == (*batch_shape, num_loc, 16)
    assert dynamic_out[2].shape == (*batch_shape, num_loc, 16)
    assert context_out.shape == (*batch_shape, 16)
    assert not torch.allclose(init_out[0, 0], init_out[0, 2])
    assert not torch.allclose(dynamic_out[0][0, 0], dynamic_out[0][0, 2])
    assert not torch.allclose(context_out[0, 0], context_out[0, 2])


def test_luop_context_remaining_deficit_uses_area_ratios():
    env = landuseOptEnv(
        generator_params={"num_loc": 2, "num_fixed": 0},
        tios=[0.5, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([1.0, 2.0]))
    td["current_plan"] = torch.tensor([[0, -1]])
    scaled_td = env.reset(_instance([10.0, 20.0]))
    scaled_td["current_plan"] = torch.tensor([[0, -1]])
    embeddings = torch.zeros(1, 2, 16)
    context = LOPContext(embed_dim=16)
    context.project_context.weight.data.zero_()
    context.project_context.weight.data[0, 16 + 8 + 2] = 1.0

    deficit_signal = context(embeddings, td)[0, 0]
    scaled_deficit_signal = context(embeddings, scaled_td)[0, 0]

    assert torch.allclose(deficit_signal, torch.tensor(1.0 / 6.0))
    assert torch.allclose(scaled_deficit_signal, deficit_signal)


def test_luop_dynamic_embedding_remaining_deficit_uses_area_ratios():
    env = landuseOptEnv(
        generator_params={"num_loc": 2, "num_fixed": 0},
        tios=[0.5, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([1.0, 2.0]))
    td["current_plan"] = torch.tensor([[0, -1]])
    scaled_td = env.reset(_instance([10.0, 20.0]))
    scaled_td["current_plan"] = torch.tensor([[0, -1]])
    embedding = LOPDynamicEmbedding(embed_dim=36)
    embedding.projection.weight.data.zero_()
    embedding.projection.weight.data[0, 20] = 1.0

    deficit_signal = embedding(td)[0][0, :, 0]
    scaled_deficit_signal = embedding(scaled_td)[0][0, :, 0]

    assert torch.allclose(deficit_signal, torch.tensor([1.0 / 6.0, 1.0 / 6.0]))
    assert torch.allclose(scaled_deficit_signal, deficit_signal)


def test_luop_init_embedding_uses_area_and_pressure_ratios():
    env = landuseOptEnv(
        generator_params={"num_loc": 2, "num_fixed": 0},
        tios=[0.5, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([1.0, 2.0]))
    scaled_td = env.reset(_instance([10.0, 20.0]))
    embedding = lopInitEmbedding(embed_dim=16, linear_bias=False)
    embedding.init_embed.weight.data.zero_()
    embedding.init_embed.weight.data[0, 2] = 1.0
    embedding.init_embed.weight.data[1, 14] = 1.0

    embedded = embedding(td)[0, :, :2]
    scaled_embedded = embedding(scaled_td)[0, :, :2]

    assert torch.allclose(embedded[:, 0], torch.tensor([1.0 / 3.0, 2.0 / 3.0]))
    assert torch.allclose(embedded[:, 1], torch.tensor([0.5, 0.5]))
    assert torch.allclose(scaled_embedded, embedded)


def test_luop_reset_exposes_grouped_constraint_pressure_by_type():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))

    assert td["constraint_pressure"].shape == (1, env.num_types)
    assert td["constraint_pressure"][0, 0] > td["constraint_pressure"][0, 3]
    assert td["constraint_pressure"][0, 2] > td["constraint_pressure"][0, 7]
    assert td["constraint_pressure"][0, 4].item() == 0
    assert td["constraint_pressure"][0, 6].item() == 0


def test_luop_init_embedding_is_conditioned_on_constraint_pressure():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    td_a = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td_b = td_a.clone()
    td_a["constraint_pressure"] = torch.zeros(1, env.num_types)
    td_b["constraint_pressure"] = torch.zeros(1, env.num_types)
    td_b["constraint_pressure"][0, 2] = 1.0
    embedding = lopInitEmbedding(embed_dim=16)

    out_a = embedding(td_a)
    out_b = embedding(td_b)

    assert not torch.allclose(out_a, out_b)


def test_luop_context_is_conditioned_on_constraint_pressure():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    td_a = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td_b = td_a.clone()
    td_a["constraint_pressure"] = torch.zeros(1, env.num_types)
    td_b["constraint_pressure"] = torch.zeros(1, env.num_types)
    td_b["constraint_pressure"][0, 2] = 1.0
    embeddings = torch.randn(1, 4, 16)
    context = LOPContext(embed_dim=16)

    out_a = context(embeddings, td_a)
    out_b = context(embeddings, td_b)

    assert not torch.allclose(out_a, out_b)


def test_luop_context_is_conditioned_on_pending_type_action():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    td_a = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td_b = td_a.clone()
    td_a["pending_type_action"] = torch.tensor([0])
    td_b["pending_type_action"] = torch.tensor([3])
    embeddings = torch.randn(1, 4, 16)
    context = LOPContext(embed_dim=16)

    out_a = context(embeddings, td_a)
    out_b = context(embeddings, td_b)

    assert not torch.allclose(out_a, out_b)


def test_luop_dynamic_embedding_is_conditioned_on_constraint_pressure():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    td_a = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td_b = td_a.clone()
    td_a["constraint_pressure"] = torch.zeros(1, env.num_types)
    td_b["constraint_pressure"] = torch.zeros(1, env.num_types)
    td_b["constraint_pressure"][0, 2] = 1.0
    embedding = LOPDynamicEmbedding(embed_dim=16)

    out_a = embedding(td_a)
    out_b = embedding(td_b)

    assert not torch.allclose(out_a[0], out_b[0])


def test_luop_dynamic_embedding_marks_previous_selected_parcel():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    td_a = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td_b = td_a.clone()
    shared_plan = torch.tensor([[0, -1, 2, -1]])
    td_a["current_plan"] = shared_plan
    td_b["current_plan"] = shared_plan.clone()
    td_a["current_node"] = torch.tensor([0])
    td_b["current_node"] = torch.tensor([2])
    td_a["i"] = torch.ones(1, 1, dtype=torch.long)
    td_b["i"] = torch.ones(1, 1, dtype=torch.long)
    embedding = LOPDynamicEmbedding(embed_dim=16)

    out_a = embedding(td_a)[0]
    out_b = embedding(td_b)[0]

    assert not torch.allclose(out_a, out_b)


def test_luop_dynamic_embedding_does_not_mark_selected_parcel_before_first_step():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    td = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    embedding = LOPDynamicEmbedding(embed_dim=28)
    embedding.projection.weight.data.zero_()
    embedding.projection.weight.data[11, 11] = 1.0

    selected_flag = embedding(td)[0][..., 11]

    assert torch.equal(selected_flag, torch.zeros_like(selected_flag))


def test_luop_dynamic_embedding_is_conditioned_on_pending_type_action():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    td_a = env.reset(_instance([0.25, 0.25, 0.25, 0.25]))
    td_b = td_a.clone()
    td_a["pending_type_action"] = torch.tensor([0])
    td_b["pending_type_action"] = torch.tensor([3])
    embedding = LOPDynamicEmbedding(embed_dim=16)
    embedding.projection.weight.data.zero_()
    embedding.projection.weight.data[0, 12] = 1.0

    type0_signal = embedding(td_a)[0][..., 0]
    type3_signal = embedding(td_b)[0][..., 0]

    assert not torch.equal(type0_signal, type3_signal)


def test_luop_attention_model_train_step_has_finite_multiobjective_loss():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    model = AttentionModel(
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
        metrics={
            "train": [
                "loss",
                "reward",
                "compatibility_reward",
                "accessibility_reward",
            ]
        },
    )
    batch = env.generator(batch_size=[2])

    out = model.shared_step(batch, 0, phase="train")

    assert torch.isfinite(out["loss"])
    assert "train/compatibility_reward" in out
    assert "train/accessibility_reward" in out


def test_luop_attention_model_train_step_has_finite_chebyshev_loss():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        sample_objective_weights=True,
        objective_scalarization="chebyshev",
        check_solution=False,
    )
    model = AttentionModel(
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
            "decode_type_first": True,
        },
        metrics={
            "train": [
                "loss",
                "reward",
                "compatibility_reward",
                "accessibility_reward",
            ]
        },
    )
    batch = env.generator(batch_size=[2])

    out = model.shared_step(batch, 0, phase="train")

    assert torch.isfinite(out["loss"])
    assert "train/compatibility_reward" in out
    assert "train/accessibility_reward" in out


def test_luop_attention_model_train_step_has_finite_type_first_loss():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    model = AttentionModel(
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
            "decode_type_first": True,
        },
        metrics={"train": ["loss", "reward"]},
    )
    batch = env.generator(batch_size=[2])

    out = model.shared_step(batch, 0, phase="train")

    assert torch.isfinite(out["loss"])
    assert "train/reward" in out


def test_luop_attention_model_has_dedicated_zoo_entrypoint():
    from rl4co.models.zoo.luop_am import (
        LUOPAttentionModel,
        LUOPAttentionModelDecoder,
        LUOPAttentionModelPolicy,
    )

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
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

    assert isinstance(model.policy, LUOPAttentionModelPolicy)
    assert isinstance(model.policy.decoder, LUOPAttentionModelDecoder)
    assert model.policy.decode_type_first is False
    assert torch.isfinite(out["loss"])
    assert "train/reward" in out


def test_generic_attention_policy_stays_luop_module_free():
    import rl4co.models.zoo.am.policy as generic_policy_module

    source = inspect.getsource(generic_policy_module)

    assert "luop_am" not in source
    assert "LUOP" not in source


def test_generic_reinforce_stays_luop_metric_free():
    import rl4co.models.rl.reinforce.reinforce as reinforce_module

    source = inspect.getsource(reinforce_module.REINFORCE)

    assert "compatibility_reward" not in source
    assert "accessibility_reward" not in source
    assert "checkpoint_score" not in source
    assert "pareto_eval_weights" not in source
    assert "evaluate_pareto_front" not in source


def test_generic_attention_decoder_stays_luop_free():
    decoder = GenericAttentionModelDecoder(
        env_name="tsp",
        embed_dim=32,
        num_heads=4,
    )

    assert not hasattr(decoder, "project_lop_type_logits")
    assert not hasattr(decoder, "lop_type_embeddings")
    assert not hasattr(decoder, "project_lop_joint_query")


def test_generic_constructive_policy_stays_luop_free():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    policy = GenericConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    assert not hasattr(policy, "is_luop_family")
    assert not hasattr(policy, "_forward_luop_type_first")
    assert not hasattr(policy, "_validate_luop_replay_actions")


def test_pareto_utilities_identify_front_and_hypervolume():
    points = torch.tensor([[[0.4, 0.4], [0.7, 0.2], [0.2, 0.7], [0.6, 0.6]]])

    mask = is_non_dominated(points)
    hypervolume = hypervolume_2d(points, reference=torch.tensor([0.0, 0.0]))

    assert mask.tolist() == [[False, True, True, True]]
    assert torch.allclose(hypervolume, torch.tensor([0.40]), atol=1e-6)


def test_normalize_weights_rejects_invalid_weight_rows():
    invalid_weights = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, -0.5],
            [float("nan"), 1.0],
        ]
    )

    with pytest.raises(ValueError, match="non-negative finite"):
        normalize_weights(invalid_weights)


def test_luop_policy_evaluates_weight_grid_as_pareto_candidates():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
    )
    batch = env.generator(batch_size=[2])

    out = evaluate_pareto_front(
        policy,
        env,
        batch,
        weights=[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        phase="test",
        decode_type="greedy",
    )

    assert out["components"].shape == (2, 3, 2)
    assert out["is_pareto"].shape == (2, 3)
    assert out["front_size"].shape == (2,)
    assert out["type_actions"].shape[:2] == (2, 3)
    assert out["parcel_actions"].shape[:2] == (2, 3)
    assert torch.isfinite(out["hypervolume"]).all()


def test_luop_pareto_evaluation_preserves_shaped_batch_before_weight_axis():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch_shape = (2, 3)
    num_weights = 3
    num_loc = 4
    batch = TensorDict(
        {
            "locs": torch.rand(*batch_shape, num_loc, 2),
            "areas": torch.full((*batch_shape, num_loc), 0.25),
            "init_plan": torch.ones(*batch_shape, num_loc, dtype=torch.long),
            "fixed_mask": torch.ones(*batch_shape, num_loc, dtype=torch.bool),
        },
        batch_size=batch_shape,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    out = evaluate_pareto_front(
        policy,
        env,
        batch,
        weights=[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        phase="test",
        decode_type="greedy",
    )

    assert out["components"].shape == (*batch_shape, num_weights, 2)
    assert out["rewards"].shape == (*batch_shape, num_weights)
    assert out["is_pareto"].shape == (*batch_shape, num_weights)
    assert out["actions"].shape == (*batch_shape, num_weights, num_loc)
    assert out["type_actions"].shape == out["actions"].shape
    assert out["parcel_actions"].shape == out["actions"].shape
    assert out["current_plan"].shape == (*batch_shape, num_weights, num_loc)
    assert out["hypervolume"].shape == batch_shape


def test_luop_pareto_evaluation_keeps_weight_axis_after_all_shaped_batch_axes():
    class EchoWeightPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.stack(
                [
                    td["objective_weights"][..., 0],
                    td["objective_weights"][..., 1],
                ],
                dim=-1,
            )
            reward = (components * td["objective_weights"]).sum(dim=-1)
            actions = torch.tensor([0, 1, 6, 7], device=td.device).expand(
                *td.batch_size, -1
            )
            return {
                "reward": reward,
                "reward_components": components,
                "actions": actions,
                "type_actions": actions // 4,
                "parcel_actions": actions % 4,
                "current_plan": torch.tensor([0, 0, 1, 1], device=td.device).expand(
                    *td.batch_size, -1
                ),
            }

    batch_shape = (2, 4)
    num_weights = 3
    num_loc = 4
    env = landuseOptEnv(
        generator_params={"num_loc": num_loc, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = TensorDict(
        {
            "locs": torch.rand(*batch_shape, num_loc, 2),
            "areas": torch.full((*batch_shape, num_loc), 0.25),
            "init_plan": torch.ones(*batch_shape, num_loc, dtype=torch.long),
            "fixed_mask": torch.ones(*batch_shape, num_loc, dtype=torch.bool),
        },
        batch_size=batch_shape,
    )

    out = evaluate_pareto_front(
        EchoWeightPolicy(),
        env,
        batch,
        weights=[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        return_actions=True,
        return_plan=True,
    )

    assert out["components"].shape == (*batch_shape, num_weights, 2)
    assert out["rewards"].shape == (*batch_shape, num_weights)
    assert out["is_pareto"].shape == (*batch_shape, num_weights)
    assert out["actions"].shape == (*batch_shape, num_weights, num_loc)
    assert out["type_actions"].shape == out["actions"].shape
    assert out["parcel_actions"].shape == out["actions"].shape
    assert out["current_plan"].shape == (*batch_shape, num_weights, num_loc)
    assert out["hypervolume"].shape == batch_shape
    assert torch.allclose(
        out["components"][0, 0, :, 0],
        torch.tensor([1.0, 0.5, 0.0]),
    )


def test_luop_type_first_pareto_eval_handles_mixed_done_batch():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 4).view(1, 4, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.full((2, 4), 0.25, dtype=torch.float32),
            "init_plan": torch.tensor([[0, 1, 2, 3], [1, 1, 1, 1]], dtype=torch.long),
            "fixed_mask": torch.tensor(
                [[False, False, False, False], [True, True, True, True]],
                dtype=torch.bool,
            ),
        },
        batch_size=[2],
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    out = evaluate_pareto_front(
        policy,
        env,
        batch,
        weights=[[1.0, 0.0], [0.0, 1.0]],
        phase="test",
        decode_type="greedy",
    )

    assert out["components"].shape == (2, 2, 2)
    assert out["actions"].shape == (2, 2, 4)
    assert out["type_actions"].shape == (2, 2, 4)
    assert out["parcel_actions"].shape == (2, 2, 4)
    assert out["current_plan"][0].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert out["current_plan"][1].min().item() >= 0
    assert torch.isfinite(out["hypervolume"]).all()


def test_pareto_evaluation_pads_done_row_dummy_actions_to_negative_one():
    class DummyDoneRowPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.stack(
                [
                    td["objective_weights"][..., 0],
                    td["objective_weights"][..., 1],
                ],
                dim=-1,
            )
            reward = (components * td["objective_weights"]).sum(dim=-1)
            actions = torch.tensor([[0, 0], [4, 7]], device=td.device)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": actions,
                "type_actions": actions // 2,
                "parcel_actions": actions % 2,
                "current_plan": torch.tensor([[0, 1], [2, 3]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 2, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 2).view(1, 2, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
            "init_plan": torch.tensor([[0, 1], [1, 1]], dtype=torch.long),
            "fixed_mask": torch.tensor(
                [[False, False], [True, True]],
                dtype=torch.bool,
            ),
        },
        batch_size=[2],
    )

    out = evaluate_pareto_front(
        DummyDoneRowPolicy(),
        env,
        batch,
        weights=[[1.0, 0.0], [0.0, 1.0]],
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"][0].eq(-1).all()
    assert out["type_actions"][0].eq(-1).all()
    assert out["parcel_actions"][0].eq(-1).all()
    assert out["actions"][1].tolist() == [[4, 7], [4, 7]]


def test_luop_type_first_pareto_eval_supports_multistart_greedy():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )
    batch = env.generator(batch_size=[2])

    out = evaluate_pareto_front(
        policy,
        env,
        batch,
        weights=[[1.0, 0.0], [0.0, 1.0]],
        phase="test",
        decode_type="multistart_greedy",
        num_starts=2,
    )

    assert out["components"].shape == (2, 2, 2)
    assert out["actions"].shape == (2, 2, 4)
    assert out["type_actions"].shape == (2, 2, 4)
    assert out["parcel_actions"].shape == (2, 2, 4)
    assert out["current_plan"].shape == (2, 2, 4)
    assert torch.isfinite(out["hypervolume"]).all()


def test_luop_type_first_policy_supports_multistart_dual_decoding():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )
    td = env.reset(env.generator(batch_size=[2]))

    out = policy(
        td,
        env,
        phase="test",
        decode_type="multistart_greedy",
        num_starts=2,
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"].shape == (4, 4)
    assert out["type_actions"].shape == (4, 4)
    assert out["parcel_actions"].shape == (4, 4)
    assert out["current_plan"].shape == (4, 4)
    assert out["current_plan"].min().item() >= 0
    assert torch.isfinite(out["reward"]).all()
    assert torch.isfinite(out["log_likelihood"]).all()


def test_luop_type_first_multistart_defaults_to_parcel_start_count():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )
    td = env.reset(env.generator(batch_size=[2]))

    out = policy(
        td,
        env,
        phase="test",
        decode_type="multistart_greedy",
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"].shape == (8, 4)
    assert out["type_actions"].shape == (8, 4)
    assert out["parcel_actions"].shape == (8, 4)
    assert out["current_plan"].shape == (8, 4)
    assert out["current_plan"].min().item() >= 0
    assert torch.isfinite(out["reward"]).all()


def test_luop_type_first_multistart_duplicates_valid_seed_for_short_rows():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 3).view(1, 3, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.full((2, 3), 1 / 3, dtype=torch.float32),
            "init_plan": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
            "fixed_mask": torch.tensor(
                [[False, False, True], [True, True, True]],
                dtype=torch.bool,
            ),
        },
        batch_size=[2],
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    out = policy(
        env.reset(batch),
        env,
        phase="test",
        decode_type="multistart_greedy",
        num_starts=3,
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"].shape == (6, 3)
    assert out["current_plan"].min().item() >= 0
    assert torch.isfinite(out["reward"]).all()


def test_luop_type_first_multistart_accepts_more_starts_than_parcels():
    class NoOpEncoder(torch.nn.Module):
        def forward(self, td):
            return None, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=NoOpEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )

    out = policy(
        env.reset(_instance([0.4, 0.3, 0.3])),
        env,
        phase="test",
        decode_type="multistart_greedy",
        num_starts=5,
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"].shape == (5, 3)
    assert out["parcel_actions"][:, 0].ge(0).all()
    assert out["parcel_actions"][:, 0].lt(3).all()
    assert out["current_plan"].min().item() >= 0


def test_luop_type_first_multistart_select_best_collapses_start_dimension():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )
    td = env.reset(env.generator(batch_size=[2]))

    out = policy(
        td,
        env,
        phase="test",
        decode_type="multistart_greedy",
        num_starts=2,
        select_best=True,
        return_actions=True,
        return_plan=True,
    )

    assert out["actions"].shape == (2, 4)
    assert out["type_actions"].shape == (2, 4)
    assert out["parcel_actions"].shape == (2, 4)
    assert out["current_plan"].shape == (2, 4)
    assert out["reward"].shape == (2,)
    assert torch.isfinite(out["reward"]).all()


def test_luop_type_first_multistart_select_best_collapses_all_done_batch():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    locs = torch.linspace(0, 1, 4).view(1, 4, 1).repeat(2, 1, 2)
    batch = TensorDict(
        {
            "locs": locs.float(),
            "areas": torch.full((2, 4), 0.25, dtype=torch.float32),
            "init_plan": torch.tensor([[0, 1, 2, 3], [1, 1, 1, 1]], dtype=torch.long),
            "fixed_mask": torch.zeros(2, 4, dtype=torch.bool),
        },
        batch_size=[2],
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )

    out = policy(
        env.reset(batch),
        env,
        phase="test",
        decode_type="multistart_greedy",
        num_starts=3,
        select_best=True,
        return_actions=True,
        return_plan=True,
        return_entropy=True,
        return_init_embeds=True,
        return_hidden=True,
    )

    assert out["reward"].shape == (2,)
    assert out["entropy"].shape == (2,)
    assert out["actions"].shape == (2, 0)
    assert out["type_actions"].shape == (2, 0)
    assert out["parcel_actions"].shape == (2, 0)
    assert out["current_plan"].shape == (2, 4)
    assert out["init_embeds"].shape[:2] == (2, 4)
    assert out["hidden"].node_embeddings.shape[:2] == (2, 4)


def test_luop_type_first_multistart_select_best_collapses_entropy():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )
    td = env.reset(env.generator(batch_size=[2]))

    out = policy(
        td,
        env,
        phase="test",
        decode_type="multistart_greedy",
        num_starts=2,
        select_best=True,
        return_actions=True,
        return_entropy=True,
    )

    assert out["actions"].shape == (2, 4)
    assert out["entropy"].shape == (2,)
    assert torch.isfinite(out["entropy"]).all()


def test_luop_type_first_multistart_returns_batchified_init_embeds():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )
    td = env.reset(env.generator(batch_size=[2]))

    out = policy(
        td,
        env,
        phase="test",
        decode_type="multistart_greedy",
        num_starts=2,
        return_actions=True,
        return_init_embeds=True,
    )

    assert out["actions"].shape == (4, 4)
    assert out["init_embeds"].shape[:2] == (4, 4)


def test_luop_type_first_multistart_select_best_returns_selected_init_embeds():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )
    td = env.reset(env.generator(batch_size=[2]))

    out = policy(
        td,
        env,
        phase="test",
        decode_type="multistart_greedy",
        num_starts=2,
        select_best=True,
        return_actions=True,
        return_init_embeds=True,
    )

    assert out["actions"].shape == (2, 4)
    assert out["init_embeds"].shape[:2] == (2, 4)


def test_luop_type_first_multistart_returns_aligned_hidden_cache():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )
    td = env.reset(env.generator(batch_size=[2]))

    out = policy(
        td,
        env,
        phase="test",
        decode_type="multistart_greedy",
        num_starts=2,
        return_actions=True,
        return_hidden=True,
    )

    assert out["actions"].shape == (4, 4)
    assert out["hidden"].node_embeddings.shape[:2] == (4, 4)


def test_luop_type_first_multistart_select_best_returns_selected_hidden_cache():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
        decode_type_first=True,
    )
    td = env.reset(env.generator(batch_size=[2]))

    out = policy(
        td,
        env,
        phase="test",
        decode_type="multistart_greedy",
        num_starts=2,
        select_best=True,
        return_actions=True,
        return_hidden=True,
    )

    assert out["actions"].shape == (2, 4)
    assert out["hidden"].node_embeddings.shape[:2] == (2, 4)


def test_luop_type_first_multistart_select_best_returns_selected_tensor_hidden():
    class TensorHiddenEncoder(torch.nn.Module):
        def forward(self, td):
            hidden = torch.arange(td.batch_size[0], device=td.device).float()[:, None]
            return hidden, None

    class ZeroDecoder(ConstructiveDecoder):
        def forward(self, td, hidden=None, num_starts=0):
            return (
                torch.zeros_like(td["action_mask"], dtype=torch.float32),
                td["action_mask"],
            )

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    policy = LUOPConstructivePolicy(
        encoder=TensorHiddenEncoder(),
        decoder=ZeroDecoder(),
        env_name="lop",
        decode_type_first=True,
    )
    td = env.reset(env.generator(batch_size=[2]))

    out = policy(
        td,
        env,
        phase="test",
        decode_type="multistart_greedy",
        num_starts=2,
        select_best=True,
        return_actions=True,
        return_hidden=True,
    )

    assert out["actions"].shape == (2, 4)
    assert out["hidden"].shape == (2, 1)


def test_pareto_evaluation_requires_policy_reward_components():
    class RewardOnlyPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            return {"reward": torch.zeros(td.batch_size, device=td.device)}

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="reward_components"):
        evaluate_pareto_front(
            RewardOnlyPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_pareto_evaluation_rejects_non_finite_reward_components():
    class NonFiniteComponentPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[float("nan"), 0.3]], device=td.device)
            return {
                "reward": torch.zeros(td.batch_size, device=td.device),
                "reward_components": components,
                "actions": torch.tensor([[0, 1]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="finite reward_components"):
        evaluate_pareto_front(
            NonFiniteComponentPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_pareto_evaluation_rejects_non_finite_rewards():
    class NonFiniteRewardPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            return {
                "reward": torch.tensor([float("inf")], device=td.device),
                "reward_components": components,
                "actions": torch.tensor([[0, 1]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="finite rewards"):
        evaluate_pareto_front(
            NonFiniteRewardPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_pareto_evaluation_rejects_reward_components_with_wrong_shape():
    class WrongComponentShapePolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.2, 0.1]], device=td.device)
            return {
                "reward": torch.zeros(td.batch_size, device=td.device),
                "reward_components": components,
                "actions": torch.tensor([[0, 1]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="reward_components shape"):
        evaluate_pareto_front(
            WrongComponentShapePolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_pareto_evaluation_rejects_reward_components_with_wrong_batch_shape():
    class WrongComponentBatchPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            return {
                "reward": torch.zeros(td.batch_size, device=td.device),
                "reward_components": components,
                "actions": torch.tensor([[0, 1], [0, 1]], device=td.device),
                "current_plan": torch.tensor(
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                    device=td.device,
                ),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[2])

    with pytest.raises(ValueError, match="reward_components shape"):
        evaluate_pareto_front(
            WrongComponentBatchPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_pareto_evaluation_rejects_rewards_with_wrong_batch_shape():
    class WrongRewardShapePolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor(
                [[0.7, 0.3], [0.6, 0.4]],
                device=td.device,
            )
            return {
                "reward": torch.zeros(1, device=td.device),
                "reward_components": components,
                "actions": torch.tensor([[0, 1], [0, 1]], device=td.device),
                "current_plan": torch.tensor(
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                    device=td.device,
                ),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[2])

    with pytest.raises(ValueError, match="reward shape"):
        evaluate_pareto_front(
            WrongRewardShapePolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_pareto_evaluation_requires_policy_action_artifacts_when_requested():
    class ComponentOnlyPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="actions"):
        evaluate_pareto_front(
            ComponentOnlyPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
        )


def test_pareto_evaluation_requires_policy_plan_artifact_when_requested():
    class NoPlanPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="current_plan"):
        evaluate_pareto_front(
            NoPlanPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_plan=True,
        )


def test_pareto_evaluation_rejects_actions_with_wrong_batch_shape():
    class WrongActionBatchPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor(
                [[0.7, 0.3], [0.6, 0.4]],
                device=td.device,
            )
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0, 1]], device=td.device),
                "current_plan": torch.tensor(
                    [[0, 0, 1, 1], [1, 1, 0, 0]],
                    device=td.device,
                ),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[2])

    with pytest.raises(ValueError, match="actions shape"):
        evaluate_pareto_front(
            WrongActionBatchPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
        )


def test_pareto_evaluation_rejects_non_integer_action_artifacts():
    class NonIntegerActionPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0.0, 1.5]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="actions.*integer"):
        evaluate_pareto_front(
            NonIntegerActionPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
        )


def test_pareto_evaluation_rejects_out_of_range_flat_action_artifacts():
    class OutOfRangeFlatActionPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[32]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="actions.*range"):
        evaluate_pareto_front(
            OutOfRangeFlatActionPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
        )


def test_pareto_evaluation_rejects_out_of_range_dual_action_artifacts():
    class OutOfRangeDualActionPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0]], device=td.device),
                "type_actions": torch.tensor([[8]], device=td.device),
                "parcel_actions": torch.tensor([[0]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="type_actions.*range"):
        evaluate_pareto_front(
            OutOfRangeDualActionPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
        )


def test_pareto_evaluation_rejects_dual_actions_with_mismatched_shape():
    class MismatchedDualActionPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0, 1]], device=td.device),
                "type_actions": torch.tensor([[0, 0]], device=td.device),
                "parcel_actions": torch.tensor([[0]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="parcel_actions shape"):
        evaluate_pareto_front(
            MismatchedDualActionPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
        )


def test_pareto_evaluation_rejects_dual_action_artifacts_with_different_padding():
    class MismatchedPaddingPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0, -1]], device=td.device),
                "type_actions": torch.tensor([[0, 1]], device=td.device),
                "parcel_actions": torch.tensor([[0, -1]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="padding"):
        evaluate_pareto_front(
            MismatchedPaddingPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
        )


def test_pareto_evaluation_rejects_flat_actions_with_non_suffix_padding():
    class NonSuffixPaddingPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[-1, 0]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="padding"):
        evaluate_pareto_front(
            NonSuffixPaddingPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
        )


def test_pareto_evaluation_rejects_dual_actions_with_non_suffix_padding():
    class NonSuffixDualPaddingPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[-1, 0]], device=td.device),
                "type_actions": torch.tensor([[-1, 0]], device=td.device),
                "parcel_actions": torch.tensor([[-1, 0]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="padding"):
        evaluate_pareto_front(
            NonSuffixDualPaddingPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
        )


def test_pareto_evaluation_rejects_dual_actions_that_do_not_encode_flat_actions():
    class InconsistentDualPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[1]], device=td.device),
                "type_actions": torch.tensor([[1]], device=td.device),
                "parcel_actions": torch.tensor([[1]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="encode"):
        evaluate_pareto_front(
            InconsistentDualPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
        )


def test_pareto_evaluation_rejects_actions_that_do_not_replay_to_current_plan():
    class InconsistentPlanPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0, 3]], device=td.device),
                "type_actions": torch.tensor([[0, 1]], device=td.device),
                "parcel_actions": torch.tensor([[0, 1]], device=td.device),
                "current_plan": torch.tensor([[1, 0]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 2, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = _instance([0.5, 0.5])

    with pytest.raises(ValueError, match="replay.*current_plan"):
        evaluate_pareto_front(
            InconsistentPlanPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
            return_plan=True,
        )


def test_pareto_evaluation_rejects_actions_with_too_few_valid_steps():
    class ShortActionTracePolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0, -1]], device=td.device),
                "type_actions": torch.tensor([[0, -1]], device=td.device),
                "parcel_actions": torch.tensor([[0, -1]], device=td.device),
                "current_plan": torch.tensor([[0, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 2, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = _instance([0.5, 0.5])

    with pytest.raises(ValueError, match="one valid action"):
        evaluate_pareto_front(
            ShortActionTracePolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
            return_plan=True,
        )


def test_pareto_evaluation_rejects_action_trace_that_violates_current_mask():
    class MaskViolatingTracePolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[8, 0, 4]], device=td.device),
                "type_actions": torch.tensor([[2, 0, 1]], device=td.device),
                "parcel_actions": torch.tensor([[2, 0, 1]], device=td.device),
                "current_plan": torch.tensor([[0, 1, 2]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0.5, 0.4, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = _instance([0.5, 0.4, 0.1])

    with pytest.raises(ValueError, match="current LUOP action_mask"):
        evaluate_pareto_front(
            MaskViolatingTracePolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
            return_plan=True,
        )


def test_pareto_evaluation_rejects_action_only_trace_that_violates_current_mask():
    class MaskViolatingActionOnlyPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[8, 0, 4]], device=td.device),
                "type_actions": torch.tensor([[2, 0, 1]], device=td.device),
                "parcel_actions": torch.tensor([[2, 0, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0.5, 0.4, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = _instance([0.5, 0.4, 0.1])

    with pytest.raises(ValueError, match="current LUOP action_mask"):
        evaluate_pareto_front(
            MaskViolatingActionOnlyPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_actions=True,
            return_plan=False,
        )


def test_pareto_evaluation_rejects_current_plan_with_wrong_batch_shape():
    class WrongPlanBatchPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor(
                [[0.7, 0.3], [0.6, 0.4]],
                device=td.device,
            )
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0, 1], [0, 1]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[2])

    with pytest.raises(ValueError, match="current_plan shape"):
        evaluate_pareto_front(
            WrongPlanBatchPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_plan=True,
        )


def test_pareto_evaluation_rejects_incomplete_current_plan_artifact():
    class IncompletePlanPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0, 1]], device=td.device),
                "current_plan": torch.tensor([[0, -1, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="current_plan.*complete"):
        evaluate_pareto_front(
            IncompletePlanPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_plan=True,
        )


def test_pareto_evaluation_rejects_current_plan_with_unknown_land_use_type():
    class UnknownTypePlanPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0, 1]], device=td.device),
                "current_plan": torch.tensor([[0, 1, 8, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="current_plan.*land-use type"):
        evaluate_pareto_front(
            UnknownTypePlanPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_plan=True,
        )


def test_pareto_evaluation_rejects_current_plan_that_violates_constraints():
    class ConstraintViolatingPlanPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0, 1, 2]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 0]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0.5, 0.4, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = _instance([0.5, 0.3, 0.2])

    with pytest.raises(ValueError, match="minimum land-use ratios"):
        evaluate_pareto_front(
            ConstraintViolatingPlanPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            return_plan=True,
        )


def test_pareto_evaluation_accepts_single_weight_vector():
    class FixedPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0, 1, 6, 7]], device=td.device),
                "type_actions": torch.tensor([[0, 0, 1, 1]], device=td.device),
                "parcel_actions": torch.tensor([[0, 1, 2, 3]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    out = evaluate_pareto_front(
        FixedPolicy(),
        env,
        batch,
        weights=[1.0, 0.0],
    )

    assert out["weights"].shape == (1, 2)
    assert out["components"].shape == (1, 1, 2)
    assert out["front_size"].tolist() == [1.0]


def test_pareto_evaluation_rejects_weight_vectors_with_wrong_objective_count():
    class UnusedPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            raise AssertionError("Policy should not be called for invalid weights")

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="two objective dimensions"):
        evaluate_pareto_front(
            UnusedPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0, 0.0]],
        )


def test_pareto_evaluation_rejects_reference_with_wrong_objective_count():
    class UnusedPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            raise AssertionError("Policy should not be called for invalid reference")

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="reference"):
        evaluate_pareto_front(
            UnusedPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
            reference=[0.0, 0.0, 0.0],
        )


def test_pareto_evaluation_requires_paired_dual_action_artifacts():
    class TypeOnlyPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": torch.tensor([[0, 1]], device=td.device),
                "type_actions": torch.tensor([[0, 0]], device=td.device),
                "current_plan": torch.tensor([[0, 0, 1, 1]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    with pytest.raises(ValueError, match="type_actions and parcel_actions"):
        evaluate_pareto_front(
            TypeOnlyPolicy(),
            env,
            batch,
            weights=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_pareto_evaluation_decodes_flat_luop_actions_into_dual_artifacts():
    class FlatOnlyLuopPolicy(torch.nn.Module):
        def forward(self, td, env, **kwargs):
            components = torch.tensor([[0.7, 0.3]], device=td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            actions = torch.tensor([[2, 7]], device=td.device)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": actions,
                "current_plan": torch.tensor([[1, 3]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 2, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = _instance([0.5, 0.5])

    out = evaluate_pareto_front(
        FlatOnlyLuopPolicy(),
        env,
        batch,
        weights=[[1.0, 0.0], [0.0, 1.0]],
        return_actions=True,
        return_plan=True,
    )

    assert out["type_actions"].tolist() == [[[1, 3], [1, 3]]]
    assert out["parcel_actions"].tolist() == [[[0, 1], [0, 1]]]
    assert out["front_type_actions"].shape[-1] == 2
    assert out["front_parcel_actions"].shape[-1] == 2


def test_pareto_evaluation_returns_front_components_with_dominated_points_removed():
    class FixedComponentPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.components = {
                (1.0, 0.0): torch.tensor([[0.6, 0.6]]),
                (0.5, 0.5): torch.tensor([[0.4, 0.4]]),
                (0.0, 1.0): torch.tensor([[0.2, 0.8]]),
            }
            self.action_values = {
                (1.0, 0.0): torch.tensor([[0, 1, 6, 7]]),
                (0.5, 0.5): torch.tensor([[8, 9, 14, 15]]),
                (0.0, 1.0): torch.tensor([[16, 17, 22, 23]]),
            }
            self.plans = {
                (1.0, 0.0): torch.tensor([[0, 0, 1, 1]]),
                (0.5, 0.5): torch.tensor([[2, 2, 3, 3]]),
                (0.0, 1.0): torch.tensor([[4, 4, 5, 5]]),
            }

        def forward(self, td, env, **kwargs):
            key = tuple(td["objective_weights"][0].tolist())
            components = self.components[key].to(td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": self.action_values[key].to(td.device),
                "type_actions": self.action_values[key].to(td.device) // 4,
                "parcel_actions": self.action_values[key].to(td.device) % 4,
                "current_plan": self.plans[key].to(td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    out = evaluate_pareto_front(
        FixedComponentPolicy(),
        env,
        batch,
        weights=[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
    )

    assert out["front_size"].tolist() == [2.0]
    assert out["front_components"].shape == (1, 2, 2)
    assert torch.allclose(
        out["front_components"],
        torch.tensor([[[0.6, 0.6], [0.2, 0.8]]]),
        atol=1e-6,
    )
    assert torch.allclose(
        out["front_weights"],
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        atol=1e-6,
    )
    assert torch.allclose(out["front_rewards"], torch.tensor([[0.6, 0.8]]), atol=1e-6)
    assert out["front_actions"].tolist() == [[[0, 1, 6, 7], [16, 17, 22, 23]]]
    assert out["front_current_plan"].tolist() == [[[0, 0, 1, 1], [4, 4, 5, 5]]]


def test_pareto_evaluation_pads_variable_length_action_artifacts():
    class VariableLengthPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.components = {
                (1.0, 0.0): torch.tensor([[0.8, 0.2]]),
                (0.5, 0.5): torch.tensor([[0.1, 0.1]]),
                (0.0, 1.0): torch.tensor([[0.2, 0.8]]),
            }
            self.actions_by_weight = {
                (1.0, 0.0): torch.tensor([[10, 11, 12]]),
                (0.5, 0.5): torch.tensor([[20]]),
                (0.0, 1.0): torch.empty(1, 0, dtype=torch.long),
            }

        def forward(self, td, env, **kwargs):
            key = tuple(td["objective_weights"][0].tolist())
            components = self.components[key].to(td.device)
            actions = self.actions_by_weight[key].to(td.device)
            reward = (components * td["objective_weights"]).sum(dim=-1)
            return {
                "reward": reward,
                "reward_components": components,
                "actions": actions,
                "type_actions": actions // 4,
                "parcel_actions": actions % 4,
                "current_plan": torch.tensor([[0, 1, 2, 3]], device=td.device),
            }

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    batch = env.generator(batch_size=[1])

    out = evaluate_pareto_front(
        VariableLengthPolicy(),
        env,
        batch,
        weights=[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        return_plan=False,
    )

    assert out["actions"].tolist() == [[[10, 11, 12], [20, -1, -1], [-1, -1, -1]]]
    assert out["front_actions"].tolist() == [[[10, 11, 12], [-1, -1, -1]]]
    assert out["front_type_actions"].tolist() == [[[2, 2, 3], [-1, -1, -1]]]
    assert out["front_parcel_actions"].tolist() == [[[2, 3, 0], [-1, -1, -1]]]


def test_luop_solution_validity_rejects_changed_fixed_parcel():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 1},
        tios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(
        _instance(
            [0.4, 0.3, 0.3],
            fixed_mask=torch.tensor([[False, True, True]]),
            init_plan=torch.tensor([[4, 1, 1]]),
        )
    )
    td["current_plan"] = torch.tensor([[1, 1, 1]])

    with pytest.raises(ValueError, match="fixed parcels"):
        env.check_solution_validity(td, torch.empty(1, 0, dtype=torch.long))


def test_luop_solution_validity_rejects_missing_required_ratio():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0.5, 0.4, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.5, 0.3, 0.2]))
    td["current_plan"] = torch.tensor([[0, 0, 0]])

    with pytest.raises(ValueError, match="minimum land-use ratios"):
        env.check_solution_validity(td, torch.empty(1, 0, dtype=torch.long))


def test_luop_solution_validity_rejects_unassigned_plan():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.4, 0.3, 0.3]))
    td["current_plan"] = torch.tensor([[0, -1, 1]])

    with pytest.raises(ValueError, match="All parcels must be assigned"):
        env.check_solution_validity(td, torch.empty(1, 0, dtype=torch.long))


def test_luop_solution_validity_rejects_unknown_land_use_type():
    env = landuseOptEnv(
        generator_params={"num_loc": 3, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.4, 0.3, 0.3]))
    td["current_plan"] = torch.tensor([[0, env.num_types, 1]])

    with pytest.raises(ValueError, match="known land-use type"):
        env.check_solution_validity(td, torch.empty(1, 0, dtype=torch.long))


def test_luop_default_constraints_accept_original_group_valid_plan_without_rc():
    env = landuseOptEnv(
        generator_params={"num_loc": 8, "num_fixed": 0},
        check_solution=False,
    )
    td = env.reset(_instance([0.10, 0.20, 0.05, 0.15, 0.02, 0.20, 0.18, 0.10]))
    td["current_plan"] = torch.tensor([[0, 1, 2, 7, 5, 4, 6, 7]])

    env.check_solution_validity(td, torch.empty(1, 0, dtype=torch.long))


def test_luop_default_constraints_reject_office_soho_group_deficit():
    env = landuseOptEnv(
        generator_params={"num_loc": 8, "num_fixed": 0},
        check_solution=False,
    )
    td = env.reset(_instance([0.10, 0.20, 0.05, 0.09, 0.02, 0.20, 0.24, 0.10]))
    td["current_plan"] = torch.tensor([[0, 1, 2, 7, 5, 4, 6, 3]])

    with pytest.raises(ValueError, match="grouped minimum land-use ratios"):
        env.check_solution_validity(td, torch.empty(1, 0, dtype=torch.long))


def test_luop_default_masks_stop_forcing_group_flex_types_after_groups_are_met():
    env = landuseOptEnv(
        generator_params={"num_loc": 6, "num_fixed": 0},
        check_solution=False,
    )
    td = env.reset(_instance([0.10, 0.20, 0.05, 0.15, 0.02, 0.48]))
    td["current_plan"] = torch.tensor([[0, 1, 2, 7, 5, -1]])
    parcel_mask = td["current_plan"] < 0

    _, type_action_mask, _ = env._build_action_masks(
        td["current_plan"], td["areas"], parcel_mask
    )

    assert type_action_mask[0].all()


def test_luop_default_masks_reject_action_that_strands_large_group_deficit():
    env = landuseOptEnv(
        generator_params={"num_loc": 8, "num_fixed": 3},
        check_solution=False,
    )
    td = env.reset(
        _instance(
            [0.0504, 0.1818, 0.0504, 0.1599, 0.1353, 0.1358, 0.1334, 0.1530],
            fixed_mask=torch.tensor(
                [[False, False, False, True, True, True, True, True]]
            ),
            init_plan=torch.tensor([[4, 6, 4, 1, 1, 1, 1, 1]]),
        )
    )

    stranded_soho_action = env.encode_action(torch.tensor([7]), torch.tensor([7]), 8)

    assert not td["action_mask"][0, stranded_soho_action.item()]


def test_luop_default_masks_reject_action_that_leaves_too_few_large_parcels_for_groups():
    env = landuseOptEnv(
        generator_params={"num_loc": 12, "num_fixed": 3},
        check_solution=False,
    )
    td = env.reset(
        _instance(
            [
                0.1817999929189682,
                0.07964859902858734,
                0.07748012989759445,
                0.07798828184604645,
                0.08872216939926147,
                0.05040000006556511,
                0.08302974700927734,
                0.06469552218914032,
                0.07111480087041855,
                0.10499231517314911,
                0.05040000006556511,
                0.06972844898700714,
            ],
            fixed_mask=torch.tensor(
                [
                    [
                        False,
                        True,
                        True,
                        True,
                        True,
                        False,
                        True,
                        True,
                        True,
                        True,
                        False,
                        True,
                    ]
                ]
            ),
            init_plan=torch.tensor([[6, 1, 1, 1, 1, 4, 1, 1, 1, 1, 4, 1]]),
        )
    )
    td["current_plan"] = torch.tensor([[6, 3, -1, 5, -1, 4, 3, 2, -1, -1, 4, 7]])

    action_mask, _, _ = env._build_action_masks(
        td["current_plan"],
        td["areas"],
        td["current_plan"] < 0,
    )
    stranded_office_action = env.encode_action(torch.tensor([2]), torch.tensor([2]), 12)

    assert not action_mask[0, stranded_office_action.item()]


def test_luop_masks_reject_action_when_deficit_needs_more_than_topk_parcels():
    env = landuseOptEnv(
        generator_params={"num_loc": 12, "num_fixed": 0},
        min_type_ratios=[0.95, 0.01, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(_instance([0.01] + [0.09] * 11))
    td["current_plan"] = torch.tensor([[-1] * 12])

    action_mask, _, _ = env._build_action_masks(
        td["current_plan"],
        td["areas"],
        td["current_plan"] < 0,
    )
    tiny_type0_action = env.encode_action(torch.tensor([0]), torch.tensor([0]), 12)

    assert not action_mask[0, tiny_type0_action.item()]


def test_luop50_generated_instance_keeps_feasible_masks_and_valid_rollout():
    env = landuseOptEnv(
        generator_params={"num_loc": 50, "num_fixed": 3},
        check_solution=True,
    )
    batch = env.generator(batch_size=[2])
    td = env.reset(batch)

    assert td["action_mask"].any(dim=-1).all()
    assert td["parcel_action_mask"].sum(dim=-1).eq(47).all()

    policy = AttentionModelPolicy(
        env_name="lop",
        embed_dim=32,
        num_encoder_layers=1,
        num_heads=4,
        feedforward_hidden=64,
    )
    out = policy(
        td,
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True,
        return_plan=True,
    )

    assert out["current_plan"].shape == (2, 50)
    assert out["current_plan"].min().item() >= 0
    assert torch.isfinite(out["reward"]).all()


def test_luop_attention_model_val_step_logs_pareto_metrics():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    model = AttentionModel(
        env,
        baseline="no",
        batch_size=2,
        train_data_size=2,
        val_data_size=2,
        test_data_size=2,
        pareto_eval_weights=[[1.0, 0.0], [0.0, 1.0]],
        policy_kwargs={
            "embed_dim": 32,
            "num_encoder_layers": 1,
            "num_heads": 4,
            "feedforward_hidden": 64,
        },
        metrics={
            "val": [
                "reward",
                "pareto_hypervolume",
                "pareto_front_size",
            ]
        },
    )
    batch = env.generator(batch_size=[2])

    out = model.shared_step(batch, 0, phase="val")

    assert "val/pareto_hypervolume" in out
    assert "val/pareto_front_size" in out
    assert torch.isfinite(out["val/pareto_hypervolume"])


def test_luop_attention_model_val_step_logs_checkpoint_score():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    model = AttentionModel(
        env,
        baseline="no",
        batch_size=2,
        train_data_size=2,
        val_data_size=2,
        test_data_size=2,
        pareto_eval_weights=[[1.0, 0.0], [0.0, 1.0]],
        policy_kwargs={
            "embed_dim": 32,
            "num_encoder_layers": 1,
            "num_heads": 4,
            "feedforward_hidden": 64,
        },
        metrics={
            "val": [
                "reward",
                "compatibility_reward",
                "accessibility_reward",
                "pareto_hypervolume",
                "checkpoint_score",
            ]
        },
    )
    batch = env.generator(batch_size=[2])

    out = model.shared_step(batch, 0, phase="val")

    assert "val/checkpoint_score" in out
    expected = torch.stack(
        [
            out["val/reward"],
            out["val/accessibility_reward"],
            out["val/pareto_hypervolume"],
        ]
    ).mean()
    assert torch.allclose(out["val/checkpoint_score"], expected)


def test_luop_attention_model_val_step_logs_pareto_metrics_with_multistart_decode():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    model = AttentionModel(
        env,
        baseline="no",
        batch_size=2,
        train_data_size=2,
        val_data_size=2,
        test_data_size=2,
        pareto_eval_weights=[[1.0, 0.0], [0.0, 1.0]],
        policy_kwargs={
            "embed_dim": 32,
            "num_encoder_layers": 1,
            "num_heads": 4,
            "feedforward_hidden": 64,
            "decode_type_first": True,
            "val_decode_type": "multistart_greedy",
        },
        metrics={
            "val": [
                "reward",
                "pareto_hypervolume",
                "pareto_front_size",
            ]
        },
    )
    batch = env.generator(batch_size=[2])

    out = model.shared_step(batch, 0, phase="val")

    assert "val/pareto_hypervolume" in out
    assert "val/pareto_front_size" in out
    assert torch.isfinite(out["val/pareto_hypervolume"])


def test_luop_rollout_baseline_materializes_sampled_objective_weights():
    class ObjectiveWeightPolicy(torch.nn.Module):
        def forward(self, td, env, decode_type="greedy"):
            assert "objective_weights" in td.keys()
            return {"reward": td["objective_weights"][..., 0]}

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        sample_objective_weights=True,
        check_solution=False,
    )
    dataset = env.dataset(4, phase="train")
    baseline = RolloutBaseline()
    baseline.policy = ObjectiveWeightPolicy()

    wrapped = baseline.wrap_dataset(dataset, env, batch_size=2, device="cpu")
    sample = wrapped[0]

    assert "objective_weights" in sample
    assert "extra" in sample
    assert torch.allclose(sample["extra"], sample["objective_weights"][0])


def test_luop_rollout_baseline_preserves_env_materialized_objective_weights():
    class ObjectiveWeightPolicy(torch.nn.Module):
        def forward(self, td, env, decode_type="greedy"):
            assert "objective_weights" in td.keys()
            return {"reward": td["objective_weights"][..., 0]}

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        sample_objective_weights=True,
        check_solution=False,
    )
    dataset = env.dataset(4, phase="train")
    baseline = RolloutBaseline()
    baseline.policy = ObjectiveWeightPolicy()

    wrapped = baseline.wrap_dataset(dataset, env, batch_size=2, device="cpu")

    for idx in range(len(wrapped)):
        sample = wrapped[idx]
        assert "objective_weights" in sample
        assert "extra" in sample


def test_luop_rollout_baseline_eval_uses_greedy_decode():
    class RecordingPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decode_types = []

        def forward(self, td, env, decode_type=None, **kwargs):
            self.decode_types.append(decode_type)
            return {"reward": torch.full(td.batch_size, 0.25)}

    baseline = RolloutBaseline()
    baseline.policy = RecordingPolicy()
    td = TensorDict({}, batch_size=[3])

    bl_val, bl_loss = baseline.eval(td, torch.zeros(3), env=None)

    assert baseline.policy.decode_types == ["greedy"]
    assert torch.equal(bl_val, torch.full((3,), 0.25))
    assert bl_loss == 0


def test_luop_rollout_baseline_rollout_restores_policy_training_mode():
    class RecordingPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_training_modes = []

        def forward(self, td, env, decode_type="greedy"):
            self.forward_training_modes.append(self.training)
            return {"reward": torch.zeros(td.batch_size)}

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        check_solution=False,
    )
    dataset = env.dataset(2, phase="val")
    policy = RecordingPolicy()
    policy.train()
    baseline = RolloutBaseline()

    baseline.rollout(policy, env, batch_size=2, device="cpu", dataset=dataset)

    assert policy.forward_training_modes == [False]
    assert policy.training is True


def test_luop_dataset_samples_objective_weights_only_for_training_by_default():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        objective_weights=[0.25, 0.75],
        sample_objective_weights=True,
        check_solution=False,
    )

    train_dataset = env.dataset(3, phase="train")
    expected_eval_weights = torch.tensor([0.25, 0.75])

    train_sample = train_dataset[0]
    assert "objective_weights" in train_sample
    assert torch.allclose(train_sample["objective_weights"].sum(), torch.tensor(1.0))
    assert not torch.allclose(train_sample["objective_weights"], expected_eval_weights)

    for phase in ("val", "test"):
        sample = env.dataset(3, phase=phase)[0]
        assert "objective_weights" in sample
        assert torch.allclose(sample["objective_weights"], expected_eval_weights)


def test_luop_sampled_objective_weights_respect_minimum_weight_floor():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        sample_objective_weights=True,
        objective_weight_min=0.2,
        check_solution=False,
    )

    dataset = env.dataset(64, phase="train")
    weights = torch.stack([dataset[index]["objective_weights"] for index in range(64)])

    assert torch.all(weights >= 0.2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(64))


def test_luop_dataset_uses_eval_objective_weights_for_validation_and_test():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        objective_weights=[0.25, 0.75],
        eval_objective_weights=[0.1, 0.9],
        sample_objective_weights=True,
        check_solution=False,
    )

    expected_eval_weights = torch.tensor([0.1, 0.9])

    for phase in ("val", "test"):
        sample = env.dataset(3, phase=phase)[0]
        assert "objective_weights" in sample
        assert torch.allclose(sample["objective_weights"], expected_eval_weights)


def test_luop_dataset_can_opt_in_to_sampled_eval_objective_weights():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        objective_weights=[0.25, 0.75],
        sample_objective_weights=True,
        sample_eval_objective_weights=True,
        check_solution=False,
    )

    expected_eval_weights = torch.tensor([0.25, 0.75])
    val_sample = env.dataset(3, phase="val")[0]

    assert "objective_weights" in val_sample
    assert torch.allclose(val_sample["objective_weights"].sum(), torch.tensor(1.0))
    assert not torch.allclose(val_sample["objective_weights"], expected_eval_weights)


def test_luop_sampled_objective_weights_follow_env_seed():
    params = {
        "generator_params": {"num_loc": 4, "num_fixed": 0},
        "sample_objective_weights": True,
        "check_solution": False,
        "seed": 1234,
    }
    env_a = landuseOptEnv(**params)
    env_b = landuseOptEnv(**params)

    weights_a = env_a.dataset(4, phase="train")[0]["objective_weights"]
    weights_b = env_b.dataset(4, phase="train")[0]["objective_weights"]

    assert torch.allclose(weights_a, weights_b)


def test_luop_attention_model_setup_materializes_sampled_objective_weights_without_rollout():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        sample_objective_weights=True,
        check_solution=False,
    )
    model = AttentionModel(
        env,
        baseline="no",
        batch_size=2,
        train_data_size=4,
        val_data_size=2,
        test_data_size=2,
        policy_kwargs={
            "embed_dim": 32,
            "num_encoder_layers": 1,
            "num_heads": 4,
            "feedforward_hidden": 64,
            "decode_type_first": True,
        },
    )

    model.setup()

    sample = model.train_dataset[0]
    assert "objective_weights" in sample
    assert torch.allclose(sample["objective_weights"].sum(), torch.tensor(1.0))


def test_luop_warmup_rollout_epoch_callback_reuses_conditioned_dataset():
    class ObjectiveWeightPolicy(torch.nn.Module):
        def forward(self, td, env, decode_type="greedy"):
            assert "objective_weights" in td.keys()
            return {"reward": td["objective_weights"][..., 0]}

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        sample_objective_weights=True,
        check_solution=False,
    )
    baseline = get_reinforce_baseline("rollout", n_epochs=1)
    policy = ObjectiveWeightPolicy()

    baseline.setup(policy, env, batch_size=2, device="cpu", dataset_size=4)
    baseline.epoch_callback(
        policy,
        env=env,
        batch_size=2,
        device="cpu",
        epoch=0,
        dataset_size=4,
    )

    assert baseline.alpha == 1.0


def test_luop_warmup_rollout_partial_alpha_keeps_mixed_baseline_path():
    class ConstantPolicy(torch.nn.Module):
        def forward(self, td, env, decode_type="greedy"):
            return {"reward": torch.ones(td.batch_size)}

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        sample_objective_weights=True,
        check_solution=False,
    )
    baseline = get_reinforce_baseline("rollout", n_epochs=3)
    baseline.baseline.policy = ConstantPolicy()
    baseline.alpha = 1 / 3
    dataset = env.dataset(4, phase="train")

    wrapped = baseline.wrap_dataset(dataset, env, batch_size=2, device="cpu")

    assert "extra" not in wrapped[0]
    assert "objective_weights" in wrapped[0]


def test_luop_rollout_baseline_update_reuses_fixed_challenge_dataset(monkeypatch):
    class AreaPolicy(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, td, env, decode_type="greedy"):
            return {"reward": td["areas"][..., 0] * self.scale}

    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        sample_objective_weights=True,
        check_solution=False,
        seed=1234,
    )
    original_dataset = env.dataset
    created_datasets = []

    def tracking_dataset(*args, **kwargs):
        dataset = original_dataset(*args, **kwargs)
        created_datasets.append(dataset)
        return dataset

    monkeypatch.setattr(env, "dataset", tracking_dataset)

    baseline = RolloutBaseline(bl_alpha=1.0)
    baseline.setup(AreaPolicy(scale=1.0), env, batch_size=4, device="cpu", dataset_size=8)
    challenge_dataset = baseline.dataset

    baseline.epoch_callback(
        AreaPolicy(scale=2.0),
        env=env,
        batch_size=4,
        device="cpu",
        epoch=1,
        dataset_size=8,
    )

    assert baseline.dataset is challenge_dataset
    assert len(created_datasets) == 1
