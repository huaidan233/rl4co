import torch

from rl4co.envs import landuseOptEnv
from rl4co.utils.ops import calculate_entropy


def test_calculate_entropy_treats_masked_negative_infinity_as_zero_probability():
    logprobs = torch.tensor(
        [
            [
                [0.0, float("-inf")],
                [torch.log(torch.tensor(0.25)), torch.log(torch.tensor(0.75))],
            ]
        ]
    )

    entropy = calculate_entropy(logprobs)

    expected = -(torch.tensor(0.25) * torch.log(torch.tensor(0.25)))
    expected = expected - torch.tensor(0.75) * torch.log(torch.tensor(0.75))
    assert torch.allclose(entropy, expected.unsqueeze(0), atol=1e-6)


def test_luop_generic_multistart_helpers_seed_parcels_not_flat_joint_actions():
    env = landuseOptEnv(
        generator_params={"num_loc": 4, "num_fixed": 0},
        min_type_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
        check_solution=False,
    )
    td = env.reset(env.generator(batch_size=[2]))
    td["current_plan"][0, 1] = 3
    (
        td["action_mask"],
        td["type_action_mask"],
        td["type_parcel_action_mask"],
    ) = env._build_action_masks(td["current_plan"], td["areas"])
    td["parcel_action_mask"] = td["current_plan"] < 0

    assert env.get_num_starts(td) == 4

    starts = env.select_start_nodes(td, num_starts=4).view(4, 2).transpose(0, 1)

    assert starts[0].tolist() == [0, 2, 3, 0]
    assert starts[1].tolist() == [0, 1, 2, 3]
    assert starts.max().item() < td["locs"].shape[-2]
