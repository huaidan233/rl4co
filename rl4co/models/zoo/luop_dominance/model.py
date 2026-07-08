from typing import Any, Optional, Union

import torch

from tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.luop_am.model import LUOPAttentionModel
from rl4co.models.zoo.luop_am.policy import LUOPAttentionModelPolicy
from rl4co.models.zoo.luop_dominance.rewards import dominance_reward


class LUOPDominanceAttentionModel(LUOPAttentionModel):
    """LUOP attention model trained with group-level Pareto dominance rewards."""

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: LUOPAttentionModelPolicy = None,
        baseline: Union[REINFORCEBaseline, str] = "shared",
        policy_kwargs={},
        baseline_kwargs={},
        pareto_eval_weights: Optional[list] = None,
        pareto_reference: Optional[list] = None,
        num_dominance_candidates: int = 4,
        rank_reward_scale: float = 1.0,
        hv_reward_scale: float = 0.25,
        crowding_reward_scale: float = 0.05,
        dominated_penalty: float = 0.25,
        dominance_reference: Optional[list] = None,
        **kwargs,
    ):
        if num_dominance_candidates < 2:
            raise ValueError("num_dominance_candidates must be at least 2")
        super().__init__(
            env=env,
            policy=policy,
            baseline=baseline,
            policy_kwargs=policy_kwargs,
            baseline_kwargs=baseline_kwargs,
            pareto_eval_weights=pareto_eval_weights,
            pareto_reference=pareto_reference,
            **kwargs,
        )
        self.num_dominance_candidates = num_dominance_candidates
        self.rank_reward_scale = rank_reward_scale
        self.hv_reward_scale = hv_reward_scale
        self.crowding_reward_scale = crowding_reward_scale
        self.dominated_penalty = dominated_penalty
        self.dominance_reference = dominance_reference

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        if phase != "train":
            return super().shared_step(batch, batch_idx, phase, dataloader_idx)

        candidate_batch = self._expand_candidate_batch(batch)
        td = self.env.reset(candidate_batch)
        out = self.policy(td, self.env, phase=phase, select_best=False)
        self._add_luop_metrics(out, candidate_batch, phase)
        out = self._add_dominance_reward(out)
        out = self.calculate_loss(
            td,
            candidate_batch,
            out,
            reward=out["dominance_reward"],
        )

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def _expand_candidate_batch(self, batch: TensorDict) -> TensorDict:
        candidate_dim = len(batch.batch_size)
        expanded = batch.unsqueeze(candidate_dim).expand(
            *batch.batch_size,
            self.num_dominance_candidates,
        )
        return expanded.clone()

    def _add_dominance_reward(self, out: dict) -> dict:
        if "reward_components" not in out:
            raise ValueError(
                "LUOPDominanceAttentionModel requires policy outputs to include "
                "reward_components"
            )
        reward, reward_info = dominance_reward(
            out["reward_components"],
            reference=self.dominance_reference,
            rank_reward_scale=self.rank_reward_scale,
            hv_reward_scale=self.hv_reward_scale,
            crowding_reward_scale=self.crowding_reward_scale,
            dominated_penalty=self.dominated_penalty,
            return_info=True,
        )
        out["scalarized_reward"] = out["reward"]
        out["dominance_reward"] = reward
        out["reward"] = reward
        out["pareto_rank"] = reward_info["rank"].to(dtype=reward.dtype)
        out["pareto_rank_mean"] = out["pareto_rank"]
        out["pareto_front_size"] = reward_info["is_pareto"].sum(dim=-1).float()
        out["hv_contribution"] = reward_info["hypervolume_contribution"]
        out["crowding_distance"] = reward_info["crowding_distance"]
        return out
