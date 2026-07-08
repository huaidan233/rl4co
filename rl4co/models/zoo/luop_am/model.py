from typing import Any, Optional, Union

import torch

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.luop_am.policy import LUOPAttentionModelPolicy
from rl4co.utils.multi_objective import evaluate_pareto_front


class LUOPAttentionModel(REINFORCE):
    """REINFORCE attention model for LUOP multi-objective land-use planning."""

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: LUOPAttentionModelPolicy = None,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        pareto_eval_weights: Optional[list] = None,
        pareto_reference: Optional[list] = None,
        **kwargs,
    ):
        if policy is None:
            policy = LUOPAttentionModelPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
        self.pareto_eval_weights = pareto_eval_weights
        self.pareto_reference = pareto_reference

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        out = self.policy(td, self.env, phase=phase, select_best=phase != "train")
        self._add_luop_metrics(out, batch, phase)

        if phase == "train":
            out = self.calculate_loss(td, batch, out)

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def _add_luop_metrics(self, out: dict, batch: Any, phase: str) -> None:
        if "reward_components" in out:
            out["compatibility_reward"] = out["reward_components"][..., 0]
            out["accessibility_reward"] = out["reward_components"][..., 1]

        if phase == "train" or self.pareto_eval_weights is None:
            return

        pareto_out = evaluate_pareto_front(
            self.policy,
            self.env,
            batch,
            weights=self.pareto_eval_weights,
            reference=self.pareto_reference,
            phase=phase,
            decode_type=getattr(self.policy, f"{phase}_decode_type"),
            return_actions=False,
            return_plan=False,
        )
        out["pareto_hypervolume"] = pareto_out["hypervolume"]
        out["pareto_front_size"] = pareto_out["front_size"]
        if "accessibility_reward" in out:
            out["checkpoint_score"] = torch.stack(
                [
                    out["reward"].mean(),
                    out["accessibility_reward"].mean(),
                    out["pareto_hypervolume"].mean(),
                ]
            ).mean()
