from math import prod
from typing import Any, Callable, Optional, Tuple, Union

import torch

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.constructive.base import (
    ConstructiveDecoder,
    ConstructiveEncoder,
    ConstructivePolicy,
)
from rl4co.models.zoo.luop_am.decoder import LUOP_ENV_NAMES
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
    process_logits,
)
from rl4co.utils.ops import batchify, calculate_entropy, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class LUOPConstructivePolicy(ConstructivePolicy):
    """Constructive policy with LUOP-specific replay and type-first decoding."""

    def __init__(
        self,
        encoder: Union[ConstructiveEncoder, Callable],
        decoder: Union[ConstructiveDecoder, Callable],
        env_name: str = "luop",
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        decode_type_first: bool = False,
        **unused_kw,
    ):
        if env_name not in LUOP_ENV_NAMES:
            raise ValueError(
                "LUOPConstructivePolicy only supports LUOP-family env names; "
                f"got {env_name!r}"
            )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kw,
        )
        self.is_luop_family = True
        self.decode_type_first = decode_type_first

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_plan: bool = False,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        type_actions=None,
        parcel_actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        luop_original_batch_shape = None
        luop_flat_batch_size = None
        if len(td.batch_size) > 1:
            luop_original_batch_shape = torch.Size(td.batch_size)
            luop_flat_batch_size = prod(luop_original_batch_shape)
            self._validate_luop_shaped_action_prefix(
                "actions", actions, luop_original_batch_shape
            )
            self._validate_luop_shaped_action_prefix(
                "type_actions", type_actions, luop_original_batch_shape
            )
            self._validate_luop_shaped_action_prefix(
                "parcel_actions", parcel_actions, luop_original_batch_shape
            )
            td = td.reshape(luop_flat_batch_size)
            actions = self._flatten_luop_shaped_actions(
                actions, luop_original_batch_shape, luop_flat_batch_size
            )
            type_actions = self._flatten_luop_shaped_actions(
                type_actions, luop_original_batch_shape, luop_flat_batch_size
            )
            parcel_actions = self._flatten_luop_shaped_actions(
                parcel_actions, luop_original_batch_shape, luop_flat_batch_size
            )

        hidden, init_embeds = self.encoder(td)

        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        replay_action_name = "actions"
        if type_actions is not None or parcel_actions is not None:
            if type_actions is None or parcel_actions is None:
                raise ValueError(
                    "Both type_actions and parcel_actions are required for dual-action replay"
                )
            if actions is not None:
                raise ValueError(
                    "Pass either flat actions or type_actions+parcel_actions, not both"
                )
            num_loc = td["locs"].shape[-2]
            type_actions = type_actions.to(device=td.device)
            parcel_actions = parcel_actions.to(device=td.device)
            self._validate_luop_dual_replay_actions(
                type_actions, parcel_actions, td, num_types=getattr(env, "num_types", 8)
            )
            type_actions = type_actions.long()
            parcel_actions = parcel_actions.long()
            type_actions = self._normalize_luop_replay_padding(type_actions, td)
            parcel_actions = self._normalize_luop_replay_padding(parcel_actions, td)
            actions = env.encode_action(
                type_actions,
                parcel_actions,
                num_loc,
            )
            replay_action_name = "type_actions+parcel_actions"

        if actions is not None:
            self._validate_luop_replay_actions(actions, td)
            actions = self._normalize_luop_replay_padding(actions, td)

        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        luop_initial_plan_for_likelihood = (
            td["current_plan"].clone() if "current_plan" in td.keys() else None
        )

        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        if self.decode_type_first:
            td, env, num_starts = self._pre_decoder_hook_luop_type_first(
                td, env, decode_strategy
            )
        else:
            td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        if self.decode_type_first:
            outdict = self._forward_luop_type_first(
                td=td,
                env=env,
                hidden=hidden,
                num_starts=num_starts,
                decode_strategy=decode_strategy,
                calc_reward=calc_reward,
                return_actions=return_actions,
                return_plan=return_plan,
                return_entropy=return_entropy,
                return_hidden=return_hidden,
                return_init_embeds=return_init_embeds,
                init_embeds=init_embeds,
                return_sum_log_likelihood=return_sum_log_likelihood,
                actions=actions,
                max_steps=max_steps,
            )
            return self._restore_luop_shaped_output(
                outdict, luop_original_batch_shape, luop_flat_batch_size
            )

        step = 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            if actions is not None:
                self._validate_luop_replay_mask(
                    mask,
                    actions[..., step],
                    td,
                    step=step,
                    action_name=replay_action_name,
                )
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        if step == 0:
            empty_actions = torch.empty(
                *td.batch_size, 0, dtype=torch.long, device=td.device
            )
            empty_logprobs = torch.empty(
                *td.batch_size, 0, dtype=td["reward"].dtype, device=td.device
            )
            if calc_reward:
                td.set("reward", env.get_reward(td, empty_actions))
            log_likelihood = (
                empty_logprobs.sum(1) if return_sum_log_likelihood else empty_logprobs
            )
            outdict = {
                "reward": td["reward"],
                "log_likelihood": log_likelihood,
            }
            if "reward_components" in td.keys():
                outdict["reward_components"] = td["reward_components"]
            if return_actions:
                outdict["actions"] = empty_actions
                if "current_plan" in td.keys():
                    outdict["type_actions"] = empty_actions
                    outdict["parcel_actions"] = empty_actions
            if return_entropy:
                outdict["entropy"] = empty_logprobs.sum(1)
            if return_hidden:
                outdict["hidden"] = hidden
            if return_init_embeds:
                outdict["init_embeds"] = init_embeds
            if return_plan:
                outdict["current_plan"] = td["current_plan"]
            return self._restore_luop_shaped_output(
                outdict, luop_original_batch_shape, luop_flat_batch_size
            )

        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)
        likelihood_mask = (
            self._luop_likelihood_mask_from_initial_plan(
                luop_initial_plan_for_likelihood,
                actions,
                getattr(decode_strategy, "num_starts", 0),
            )
            if luop_initial_plan_for_likelihood is not None
            else td.get("mask", None)
        )

        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, likelihood_mask, return_sum_log_likelihood
            ),
        }
        if "reward_components" in td.keys():
            outdict["reward_components"] = td["reward_components"]

        if return_actions:
            output_actions = self._pad_luop_action_outputs_from_initial_plan(
                actions,
                luop_initial_plan_for_likelihood,
                getattr(decode_strategy, "num_starts", 0),
            )
            outdict["actions"] = output_actions
            if "current_plan" in td.keys():
                num_loc = td["current_plan"].shape[-1]
                output_type_actions = actions // num_loc
                output_parcel_actions = actions % num_loc
                if output_actions is not actions:
                    output_type_actions = torch.where(
                        output_actions.eq(-1),
                        output_actions,
                        output_type_actions,
                    )
                    output_parcel_actions = torch.where(
                        output_actions.eq(-1),
                        output_actions,
                        output_parcel_actions,
                    )
                outdict["type_actions"] = output_type_actions
                outdict["parcel_actions"] = output_parcel_actions
        if return_entropy:
            outdict["entropy"] = self._calculate_entropy(logprobs, likelihood_mask)
        if return_hidden:
            outdict["hidden"] = hidden
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds
        if return_plan:
            outdict["current_plan"] = td["current_plan"]

        return self._restore_luop_shaped_output(
            outdict, luop_original_batch_shape, luop_flat_batch_size
        )

    @staticmethod
    def _validate_luop_shaped_action_prefix(
        name: str,
        actions: Optional[Tensor],
        batch_shape: torch.Size,
    ) -> None:
        if actions is None:
            return
        if actions.shape[: len(batch_shape)] != batch_shape:
            raise ValueError(
                f"LUOP replay {name} shape must match shaped batch "
                f"{tuple(batch_shape)} before policy flattening; got "
                f"{tuple(actions.shape)}"
            )

    @staticmethod
    def _flatten_luop_shaped_actions(
        actions: Optional[Tensor],
        batch_shape: Optional[torch.Size],
        flat_batch_size: Optional[int],
    ) -> Optional[Tensor]:
        if actions is None or batch_shape is None:
            return actions
        if actions.shape[: len(batch_shape)] == batch_shape:
            return actions.reshape(flat_batch_size, *actions.shape[len(batch_shape) :])
        return actions

    @classmethod
    def _restore_luop_shaped_output(
        cls,
        outdict: dict,
        batch_shape: Optional[torch.Size],
        flat_batch_size: Optional[int],
    ) -> dict:
        if batch_shape is None:
            return outdict
        return {
            key: cls._restore_luop_shaped_value(value, batch_shape, flat_batch_size)
            for key, value in outdict.items()
        }

    @staticmethod
    def _restore_luop_shaped_value(value, batch_shape, flat_batch_size):
        if isinstance(value, Tensor) and value.shape[:1] == torch.Size([flat_batch_size]):
            return value.reshape(*batch_shape, *value.shape[1:])
        if isinstance(value, TensorDict) and value.batch_size == torch.Size(
            [flat_batch_size]
        ):
            return value.reshape(*batch_shape)
        if hasattr(value, "fields"):
            return type(value)(
                *[
                    (
                        LUOPConstructivePolicy._restore_luop_shaped_value(
                            field, batch_shape, flat_batch_size
                        )
                        if isinstance(field, (Tensor, TensorDict))
                        else field
                    )
                    for field in value.fields
                ]
            )
        return value

    @staticmethod
    def _require_luop_integer_replay_actions(name: str, actions: Tensor) -> None:
        if not torch.isfinite(actions).all():
            raise ValueError(f"LUOP replay {name} must contain finite integer ids")
        if actions.is_floating_point() and not torch.equal(actions, actions.round()):
            raise ValueError(f"LUOP replay {name} must contain integer ids")

    @staticmethod
    def _luop_replay_padding_mask(td: TensorDict, num_steps: int) -> Tensor:
        row_steps = (td["current_plan"] < 0).sum(dim=-1)
        view_shape = (1,) * row_steps.dim() + (num_steps,)
        step_idx = torch.arange(
            num_steps, device=row_steps.device, dtype=row_steps.dtype
        ).view(view_shape)
        return step_idx >= row_steps.unsqueeze(-1)

    @staticmethod
    def _normalize_luop_replay_padding(actions: Tensor, td: TensorDict) -> Tensor:
        padding = LUOPConstructivePolicy._luop_replay_padding_mask(
            td, actions.size(-1)
        ).to(device=actions.device)
        return torch.where(
            padding & actions.eq(-1),
            torch.zeros_like(actions),
            actions,
        )

    @staticmethod
    def _luop_likelihood_mask_from_initial_plan(
        initial_plan: Tensor,
        actions: Tensor,
        num_starts: int = 0,
    ) -> Tensor:
        row_steps = initial_plan.lt(0).sum(dim=-1).to(device=actions.device)
        step_idx = torch.arange(
            actions.size(-1),
            device=actions.device,
            dtype=row_steps.dtype,
        ).view((1,) * row_steps.dim() + (actions.size(-1),))
        mask = step_idx < row_steps.unsqueeze(-1)
        if mask.shape[:-1] == actions.shape[:-1]:
            return mask
        if num_starts and num_starts > 1:
            expanded_mask = batchify(mask, num_starts)
            if expanded_mask.shape[:-1] == actions.shape[:-1]:
                return expanded_mask
        return mask.reshape(actions.shape)

    @staticmethod
    def _pad_luop_action_outputs_from_initial_plan(
        actions: Tensor,
        initial_plan: Optional[Tensor],
        num_starts: int = 0,
    ) -> Tensor:
        if initial_plan is None:
            return actions
        valid_action_mask = (
            LUOPConstructivePolicy._luop_likelihood_mask_from_initial_plan(
                initial_plan,
                actions,
                num_starts=num_starts,
            )
        )
        return torch.where(valid_action_mask, actions, actions.new_full((), -1))

    def _pre_decoder_hook_luop_type_first(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        decode_strategy: DecodingStrategy,
    ) -> Tuple[TensorDict, RL4COEnvBase, int]:
        if decode_strategy.multisample:
            raise ValueError("decode_type_first does not support multisample decoding")
        if not decode_strategy.multistart:
            if decode_strategy.num_starts is not None and decode_strategy.num_starts >= 1:
                log.warn(
                    f"num_starts={decode_strategy.num_starts} is ignored for "
                    f"decode_type={decode_strategy.name}"
                )
            decode_strategy.num_starts = 0
            return td, env, 0

        if decode_strategy.num_starts is None:
            decode_strategy.num_starts = int(
                td["parcel_action_mask"].sum(dim=-1).max().item()
            )
        num_starts = decode_strategy.num_starts
        if num_starts < 1:
            return td, env, 0

        if decode_strategy.select_start_nodes_fn is not None:
            raise ValueError(
                "decode_type_first multistart does not support custom "
                "select_start_nodes_fn; seed starts come from LUOP parcel masks"
            )

        return batchify(td, num_starts), env, num_starts

    @staticmethod
    def _luop_multistart_seed_parcels(td: TensorDict, num_starts: int) -> Tensor:
        parcel_mask = td["parcel_action_mask"]
        td_unbatched = unbatchify(td, num_starts)
        base_parcel_mask = td_unbatched["parcel_action_mask"][:, 0]
        base_done = td_unbatched["done"][:, 0].squeeze(-1)
        batch_size, num_loc = base_parcel_mask.shape
        parcel_idx = torch.arange(num_loc, device=parcel_mask.device).view(1, -1)
        masked_idx = torch.where(
            base_parcel_mask,
            parcel_idx.expand(batch_size, -1),
            torch.full((batch_size, num_loc), num_loc, device=parcel_mask.device),
        )
        selected = masked_idx.sort(dim=-1).values[:, : min(num_starts, num_loc)]
        if selected.size(-1) < num_starts:
            pad = selected.new_full(
                (batch_size, num_starts - selected.size(-1)),
                num_loc,
            )
            selected = torch.cat([selected, pad], dim=-1)
        row_has_valid_start = base_parcel_mask.any(dim=-1, keepdim=True)
        fallback_start = torch.where(
            row_has_valid_start,
            selected[:, :1].clamp(max=num_loc - 1),
            torch.zeros_like(selected[:, :1]),
        )
        selected = torch.where(
            selected.eq(num_loc),
            fallback_start.expand_as(selected),
            selected,
        )
        selected = torch.where(
            base_done.unsqueeze(-1),
            torch.zeros_like(selected),
            selected,
        )
        return selected.transpose(0, 1).reshape(-1)

    @staticmethod
    def _luop_multistart_best_indices(rewards: Tensor, num_starts: int) -> Tensor:
        _, best_start = unbatchify(rewards, num_starts).max(dim=-1)
        batch_size = best_start.size(0)
        return torch.arange(batch_size, device=rewards.device) + best_start * batch_size

    @staticmethod
    def _validate_luop_replay_actions(actions: Tensor, td: TensorDict) -> None:
        LUOPConstructivePolicy._require_luop_integer_replay_actions("actions", actions)
        actions = actions.long()
        num_actions = td["action_mask"].size(-1)
        expected_steps = int((td["current_plan"] < 0).sum(dim=-1).max().item())
        if actions.shape[:-1] != td.batch_size:
            raise ValueError(
                "LUOP replay actions shape must match "
                f"batch shape {tuple(td.batch_size)}; got {tuple(actions.shape)}"
            )
        provided_steps = actions.size(-1)
        if provided_steps != expected_steps:
            raise ValueError(
                "LUOP replay action length must match the number of decoding "
                f"steps ({expected_steps}); got {provided_steps}"
            )
        padding = LUOPConstructivePolicy._luop_replay_padding_mask(td, provided_steps).to(
            device=actions.device
        )
        valid_padding = padding & actions.eq(-1)
        invalid = ((actions < 0) & ~valid_padding) | (actions >= num_actions)
        if invalid.any():
            raise ValueError(
                "LUOP replay actions must be in [0, "
                f"{num_actions - 1}] for active rows, with -1 allowed only "
                "as done-row padding"
            )

    @staticmethod
    def _validate_luop_dual_replay_actions(
        type_actions: Tensor,
        parcel_actions: Tensor,
        td: TensorDict,
        num_types: int,
    ) -> None:
        LUOPConstructivePolicy._require_luop_integer_replay_actions(
            "type_actions", type_actions
        )
        LUOPConstructivePolicy._require_luop_integer_replay_actions(
            "parcel_actions", parcel_actions
        )
        type_actions = type_actions.long()
        parcel_actions = parcel_actions.long()
        expected_steps = int((td["current_plan"] < 0).sum(dim=-1).max().item())
        if type_actions.shape[:-1] != td.batch_size:
            raise ValueError(
                "LUOP replay type_actions shape must match "
                f"batch shape {tuple(td.batch_size)}; got {tuple(type_actions.shape)}"
            )
        if parcel_actions.shape[:-1] != td.batch_size:
            raise ValueError(
                "LUOP replay parcel_actions shape must match "
                f"batch shape {tuple(td.batch_size)}; got {tuple(parcel_actions.shape)}"
            )
        if type_actions.shape != parcel_actions.shape:
            raise ValueError(
                "LUOP replay type_actions and parcel_actions must have the same shape"
            )
        provided_steps = type_actions.size(-1)
        if provided_steps != expected_steps:
            raise ValueError(
                "LUOP replay action length must match the number of decoding "
                f"steps ({expected_steps}); got {provided_steps}"
            )
        num_loc = td["locs"].shape[-2]
        padding = LUOPConstructivePolicy._luop_replay_padding_mask(td, provided_steps).to(
            device=type_actions.device
        )
        negative_type = type_actions.eq(-1)
        negative_parcel = parcel_actions.eq(-1)
        if (negative_type ^ negative_parcel).any():
            raise ValueError(
                "LUOP replay type_actions and parcel_actions must use matching "
                "-1 padding"
            )
        valid_padding = padding & negative_type & negative_parcel
        invalid_type = ((type_actions < 0) & ~valid_padding) | (type_actions >= num_types)
        if invalid_type.any():
            raise ValueError(
                "LUOP replay type_actions must be in [0, "
                f"{num_types - 1}] for active rows, with -1 allowed only "
                "as done-row padding"
            )
        invalid_parcel = ((parcel_actions < 0) & ~valid_padding) | (
            parcel_actions >= num_loc
        )
        if invalid_parcel.any():
            raise ValueError(
                "LUOP replay parcel_actions must be in [0, "
                f"{num_loc - 1}] for active rows, with -1 allowed only "
                "as done-row padding"
            )

    @staticmethod
    def _validate_luop_replay_mask(
        mask: Tensor,
        action: Tensor,
        td: TensorDict,
        step: int,
        action_name: str,
    ) -> None:
        action = action.long()
        active = ~td["done"].squeeze(-1)
        if active.shape != action.shape:
            active = active.reshape(action.shape)
        feasible = mask.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        invalid = active & ~feasible
        if invalid.any():
            rows = invalid.nonzero(as_tuple=False).detach().cpu().tolist()
            if len(rows) > 8:
                rows = rows[:8] + [["..."]]
            raise ValueError(
                f"LUOP replay {action_name} infeasible at step {step}; "
                "forced actions must satisfy the current action mask. "
                f"Invalid batch rows: {rows}"
            )

    def _decode_luop_component(
        self,
        logits: Tensor,
        mask: Tensor,
        decode_strategy: DecodingStrategy,
        action: Optional[Tensor] = None,
        active: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        feasibility_mask = mask.to(dtype=torch.bool) if mask is not None else None
        effective_mask = feasibility_mask if decode_strategy.mask_logits else None
        logprobs = process_logits(
            logits,
            effective_mask,
            temperature=decode_strategy.temperature,
            top_p=decode_strategy.top_p,
            top_k=decode_strategy.top_k,
            tanh_clipping=decode_strategy.tanh_clipping,
            mask_logits=decode_strategy.mask_logits,
        )

        if action is not None:
            selected = action.long()
        elif decode_strategy.name == "greedy":
            selected = decode_strategy.greedy(logprobs, effective_mask)
        elif decode_strategy.name == "sampling":
            selected = decode_strategy.sampling(logprobs, effective_mask)
        else:
            raise ValueError(
                f"decode_type_first does not support decode type {decode_strategy.name!r}"
            )
        if active is not None:
            active = active.to(device=selected.device, dtype=torch.bool)
            selected = torch.where(active, selected, torch.zeros_like(selected))

        if feasibility_mask is not None:
            invalid = (~feasibility_mask).gather(1, selected.unsqueeze(-1)).squeeze(-1)
            if active is not None:
                invalid = invalid & active
            if invalid.any():
                rows = invalid.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
                raise ValueError(
                    "infeasible LUOP component action selected for batch rows " f"{rows}"
                )

        selected_logprob = logprobs.gather(1, selected.unsqueeze(-1)).squeeze(-1)
        if active is not None:
            selected_logprob = torch.where(
                active,
                selected_logprob,
                torch.zeros_like(selected_logprob),
            )
        return logprobs, selected, selected_logprob

    @staticmethod
    def _calculate_entropy(
        logprobs: Tensor, step_mask: Optional[Tensor] = None
    ) -> Tensor:
        if step_mask is not None:
            step_mask = step_mask.to(device=logprobs.device, dtype=torch.bool)
            while step_mask.dim() < logprobs.dim():
                step_mask = step_mask.unsqueeze(-1)
            logprobs = logprobs.masked_fill(~step_mask, 0)
        return calculate_entropy(logprobs)

    def _forward_luop_type_first(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        hidden: Any,
        num_starts: int,
        decode_strategy: DecodingStrategy,
        calc_reward: bool,
        return_actions: bool,
        return_plan: bool,
        return_entropy: bool,
        return_hidden: bool,
        return_init_embeds: bool,
        init_embeds: Optional[Tensor],
        return_sum_log_likelihood: bool,
        actions,
        max_steps: int,
    ) -> dict:
        step = 0
        flat_actions = []
        type_actions = []
        parcel_actions = []
        type_selected_logprobs = []
        parcel_selected_logprobs = []
        type_logprob_dists = []
        parcel_logprob_dists = []
        active_masks = []

        while not td["done"].all():
            active = ~td["done"].squeeze(-1)
            active_masks.append(active)
            logits, mask = self.decoder(td, hidden, num_starts)
            num_loc = td["current_plan"].shape[-1]
            num_types = getattr(env, "num_types", 8)
            batch_size = logits.shape[0]
            device = logits.device

            joint_logits = logits.view(batch_size, num_types, num_loc)
            joint_mask = mask.view(batch_size, num_types, num_loc)
            masked_joint_logits = joint_logits.masked_fill(~joint_mask, float("-inf"))
            seed_parcel = None
            seeded_multistart_step = num_starts > 1 and actions is None and step == 0
            if seeded_multistart_step:
                seed_parcel = self._luop_multistart_seed_parcels(td, num_starts)
                batch_idx = torch.arange(batch_size, device=device)
                type_logits = masked_joint_logits[batch_idx, :, seed_parcel]
                type_mask = joint_mask[batch_idx, :, seed_parcel]
            else:
                type_mask = joint_mask.any(dim=-1)
                finite_floor = torch.finfo(joint_logits.dtype).min
                safe_joint_logits = torch.where(
                    joint_mask,
                    joint_logits,
                    joint_logits.new_full((), finite_floor),
                )
                type_logits = safe_joint_logits.logsumexp(dim=-1)
                type_logits = type_logits.masked_fill(~type_mask, finite_floor)

            forced_action = actions[..., step] if actions is not None else None
            if forced_action is None:
                forced_type = None
                forced_parcel = seed_parcel
            else:
                forced_type, forced_parcel = env.decode_action(
                    forced_action.long(), num_loc
                )
                self._validate_luop_replay_mask(
                    type_mask,
                    forced_type,
                    td,
                    step=step,
                    action_name="type_actions",
                )

            type_logprobs, selected_type, type_logprob = self._decode_luop_component(
                type_logits,
                type_mask,
                decode_strategy,
                action=forced_type,
                active=active,
            )

            td.set("pending_type_action", selected_type)
            parcel_logits, parcel_mask = self.decoder(td, hidden, num_starts)
            parcel_logits = parcel_logits.view(batch_size, num_types, num_loc)
            parcel_mask = parcel_mask.view(batch_size, num_types, num_loc)

            batch_idx = torch.arange(batch_size, device=device)
            parcel_logits = parcel_logits[batch_idx, selected_type]
            parcel_mask = parcel_mask[batch_idx, selected_type]
            if forced_parcel is not None:
                self._validate_luop_replay_mask(
                    parcel_mask,
                    forced_parcel,
                    td,
                    step=step,
                    action_name="parcel_actions",
                )

            (
                parcel_logprobs,
                selected_parcel,
                parcel_logprob,
            ) = self._decode_luop_component(
                parcel_logits,
                parcel_mask,
                decode_strategy,
                action=forced_parcel,
                active=active,
            )
            if seeded_multistart_step:
                parcel_logprob = torch.zeros_like(parcel_logprob)

            action = env.encode_action(selected_type, selected_parcel, num_loc)
            td = td.exclude("pending_type_action", inplace=False)
            td.set("type_action", selected_type)
            td.set("parcel_action", selected_parcel)
            td.set("action", action)
            td = env.step(td)["next"]

            flat_actions.append(action)
            type_actions.append(selected_type)
            parcel_actions.append(selected_parcel)
            type_selected_logprobs.append(type_logprob)
            parcel_selected_logprobs.append(parcel_logprob)
            if return_entropy:
                type_logprob_dists.append(type_logprobs)
                parcel_logprob_dists.append(parcel_logprobs)

            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        if not flat_actions:
            hidden_out = hidden
            init_embeds_out = init_embeds
            if num_starts > 1 and init_embeds_out is not None:
                init_embeds_out = batchify(init_embeds_out, num_starts)
            if num_starts > 1 and hidden_out is not None:
                if hasattr(hidden_out, "batchify"):
                    hidden_out = hidden_out.batchify(num_starts)
                elif isinstance(hidden_out, Tensor):
                    hidden_out = batchify(hidden_out, num_starts)
            empty_actions = torch.empty(
                *td.batch_size, 0, dtype=torch.long, device=td.device
            )
            empty_logprobs = torch.empty(
                *td.batch_size, 0, dtype=td["reward"].dtype, device=td.device
            )
            if calc_reward:
                td.set("reward", env.get_reward(td, empty_actions))
            if num_starts > 1 and decode_strategy.select_best:
                best_idx = self._luop_multistart_best_indices(td["reward"], num_starts)
                td = td[best_idx]
                empty_actions = empty_actions[best_idx]
                empty_logprobs = empty_logprobs[best_idx]
                if init_embeds_out is not None:
                    init_embeds_out = init_embeds_out[best_idx]
                if hidden_out is not None and hasattr(hidden_out, "fields"):
                    hidden_out = type(hidden_out)(
                        *[
                            field[best_idx] if isinstance(field, Tensor) else field
                            for field in hidden_out.fields
                        ]
                    )
                elif isinstance(hidden_out, Tensor):
                    hidden_out = hidden_out[best_idx]
            outdict = {
                "reward": td["reward"],
                "log_likelihood": (
                    empty_logprobs.sum(1) if return_sum_log_likelihood else empty_logprobs
                ),
                "type_log_likelihood": (
                    empty_logprobs.sum(1) if return_sum_log_likelihood else empty_logprobs
                ),
                "parcel_log_likelihood": (
                    empty_logprobs.sum(1) if return_sum_log_likelihood else empty_logprobs
                ),
            }
            if "reward_components" in td.keys():
                outdict["reward_components"] = td["reward_components"]
            if return_actions:
                outdict["actions"] = empty_actions
                outdict["type_actions"] = empty_actions
                outdict["parcel_actions"] = empty_actions
            if return_entropy:
                outdict["entropy"] = empty_logprobs.sum(1)
            if return_hidden:
                outdict["hidden"] = hidden_out
            if return_init_embeds:
                outdict["init_embeds"] = init_embeds_out
            if return_plan:
                outdict["current_plan"] = td["current_plan"]
            return outdict

        actions = torch.stack(flat_actions, dim=1)
        type_actions = torch.stack(type_actions, dim=1)
        parcel_actions = torch.stack(parcel_actions, dim=1)
        type_logprobs = torch.stack(type_selected_logprobs, dim=1)
        parcel_logprobs = torch.stack(parcel_selected_logprobs, dim=1)
        logprobs = type_logprobs + parcel_logprobs
        active_mask = torch.stack(active_masks, dim=1)

        type_log_likelihood = (
            type_logprobs.sum(1) if return_sum_log_likelihood else type_logprobs
        )
        parcel_log_likelihood = (
            parcel_logprobs.sum(1) if return_sum_log_likelihood else parcel_logprobs
        )
        log_likelihood = logprobs.sum(1) if return_sum_log_likelihood else logprobs

        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        init_embeds_out = init_embeds
        if num_starts > 1 and init_embeds_out is not None:
            init_embeds_out = batchify(init_embeds_out, num_starts)
        hidden_out = hidden
        if num_starts > 1 and hidden_out is not None:
            if hasattr(hidden_out, "batchify"):
                hidden_out = hidden_out.batchify(num_starts)
            elif isinstance(hidden_out, Tensor):
                hidden_out = batchify(hidden_out, num_starts)

        entropy = None
        if return_entropy:
            type_entropy = self._calculate_entropy(
                torch.stack(type_logprob_dists, dim=1),
                active_mask,
            )
            parcel_entropy = self._calculate_entropy(
                torch.stack(parcel_logprob_dists, dim=1),
                active_mask,
            )
            entropy = type_entropy + parcel_entropy

        if num_starts > 1 and decode_strategy.select_best:
            best_idx = self._luop_multistart_best_indices(td["reward"], num_starts)
            td = td[best_idx]
            actions = actions[best_idx]
            type_actions = type_actions[best_idx]
            parcel_actions = parcel_actions[best_idx]
            active_mask = active_mask[best_idx]
            log_likelihood = log_likelihood[best_idx]
            type_log_likelihood = type_log_likelihood[best_idx]
            parcel_log_likelihood = parcel_log_likelihood[best_idx]
            if entropy is not None:
                entropy = entropy[best_idx]
            if init_embeds_out is not None:
                init_embeds_out = init_embeds_out[best_idx]
            if hidden_out is not None and hasattr(hidden_out, "fields"):
                hidden_out = type(hidden_out)(
                    *[
                        field[best_idx] if isinstance(field, Tensor) else field
                        for field in hidden_out.fields
                    ]
                )
            elif isinstance(hidden_out, Tensor):
                hidden_out = hidden_out[best_idx]

        outdict = {
            "reward": td["reward"],
            "log_likelihood": log_likelihood,
            "type_log_likelihood": type_log_likelihood,
            "parcel_log_likelihood": parcel_log_likelihood,
        }
        if "reward_components" in td.keys():
            outdict["reward_components"] = td["reward_components"]

        if return_actions:
            outdict["actions"] = torch.where(
                active_mask,
                actions,
                actions.new_full((), -1),
            )
            outdict["type_actions"] = torch.where(
                active_mask,
                type_actions,
                type_actions.new_full((), -1),
            )
            outdict["parcel_actions"] = torch.where(
                active_mask,
                parcel_actions,
                parcel_actions.new_full((), -1),
            )
        if return_entropy:
            outdict["entropy"] = entropy
        if return_hidden:
            outdict["hidden"] = hidden_out
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds_out
        if return_plan:
            outdict["current_plan"] = td["current_plan"]

        return outdict
