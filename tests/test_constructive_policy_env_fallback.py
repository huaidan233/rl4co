import torch
import pytest

from tensordict import TensorDict

import rl4co.models.common.constructive.base as constructive_base
from rl4co.models.common.constructive.base import ConstructiveDecoder, ConstructivePolicy


class DoneEncoder(torch.nn.Module):
    def forward(self, td):
        return None, None


class UnusedDecoder(ConstructiveDecoder):
    def forward(self, td, hidden=None, num_starts=0):
        raise AssertionError("Decoder should not be called for a done state")


def test_constructive_policy_uses_its_own_env_name_when_env_is_missing():
    requested_env_names = []

    def fake_get_env(env_name):
        requested_env_names.append(env_name)
        return object()

    policy = ConstructivePolicy(
        encoder=DoneEncoder(),
        decoder=UnusedDecoder(),
        env_name="tsp",
    )
    td = TensorDict(
        {
            "done": torch.ones(1, 1, dtype=torch.bool),
            "reward": torch.zeros(1, 1),
        },
        batch_size=[1],
    )

    with pytest.raises(AssertionError, match="No logprobs"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(constructive_base, "get_env", fake_get_env)
            policy(td, env=None, calc_reward=False)

    assert requested_env_names == ["tsp"]
