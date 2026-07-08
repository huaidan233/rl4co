from typing import Union

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.mlp.policy import MLPPolicy


class MLPModel(REINFORCE):
    """MLP Model based on REINFORCE for combinatorial optimization.
    Uses MLP instead of Attention mechanism.

    Check :class:`REINFORCE` and :class:`rl4co.models.RL4COLitModule` for more details.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to rollout (1 epoch of exponential, then greedy rollout baseline)
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: MLPPolicy = None,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = MLPPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
