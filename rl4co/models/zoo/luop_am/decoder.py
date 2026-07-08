import math
from typing import Tuple

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.models.zoo.am.decoder import AttentionModelDecoder, PrecomputedCache


LUOP_ENV_NAMES = {
    "luop",
    "lop",
    "lop_nearest",
    "lop_compatibility",
    "MAlop",
    "MAOpt",
}


class LUOPAttentionModelDecoder(AttentionModelDecoder):
    """Attention decoder for LUOP joint land-use type and parcel actions."""

    def __init__(
        self,
        *args,
        env_name: str = "luop",
        linear_bias: bool = False,
        **kwargs,
    ):
        if env_name not in LUOP_ENV_NAMES:
            raise ValueError(
                "LUOPAttentionModelDecoder only supports LUOP-family env names; "
                f"got {env_name!r}"
            )
        super().__init__(
            *args,
            env_name=env_name,
            linear_bias=linear_bias,
            **kwargs,
        )
        self.project_lop_type_logits = nn.Linear(self.embed_dim, 8, bias=linear_bias)
        self.lop_type_embeddings = nn.Embedding(8, self.embed_dim)
        self.project_lop_joint_query = nn.Linear(
            2 * self.embed_dim,
            self.embed_dim,
            bias=linear_bias,
        )

    def forward(
        self,
        td: TensorDict,
        cached: PrecomputedCache,
        num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute flattened type-parcel logits for LUOP decoding."""

        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1

        if has_dyn_emb_multi_start:
            cached = cached.batchify(num_starts=num_starts)
        elif num_starts > 1:
            from rl4co.utils.ops import unbatchify

            td = unbatchify(td, num_starts)

        glimpse_q = self._compute_q(cached, td)
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)

        parcel_mask = td["parcel_action_mask"]
        parcel_logits = self.pointer(
            glimpse_q, glimpse_k, glimpse_v, logit_k, parcel_mask
        )
        type_context = glimpse_q.squeeze(-2)
        type_logits = self.project_lop_type_logits(type_context)
        type_embeddings = self.lop_type_embeddings.weight.unsqueeze(0).expand(
            type_context.size(0), -1, -1
        )
        type_context = type_context.unsqueeze(1).expand_as(type_embeddings)
        joint_query = torch.tanh(
            self.project_lop_joint_query(
                torch.cat([type_context, type_embeddings], dim=-1)
            )
        )
        type_parcel_logits = torch.einsum(
            "bte,bne->btn", joint_query, logit_k
        ) / math.sqrt(self.embed_dim)
        logits = (
            type_logits.unsqueeze(-1) + parcel_logits.unsqueeze(-2) + type_parcel_logits
        ).reshape(td["action_mask"].shape)
        mask = td["action_mask"]

        if num_starts > 1 and not has_dyn_emb_multi_start:
            from einops import rearrange

            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)
        return logits, mask
