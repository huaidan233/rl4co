from dataclasses import dataclass, fields
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.autoregressive.decoder import AutoregressiveDecoder
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@dataclass
class MLPPrecomputedCache:
    """Cache for precomputed embeddings in MLP decoder"""
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    node_keys: Tensor  # Precomputed keys for efficient logit computation

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts):
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return MLPPrecomputedCache(*new_embs)


class MLPDecoder(AutoregressiveDecoder):
    """
    MLP-based Auto-regressive decoder for combinatorial optimization.
    Uses efficient bilinear/dot-product style computation instead of concatenation.

    Memory-efficient design:
    - Precomputes node projections once
    - Uses dot-product style scoring (no large intermediate tensors per step)

    Args:
        embed_dim: Embedding dimension
        env_name: Name of the environment used to initialize embeddings
        context_embedding: Context embedding module
        dynamic_embedding: Dynamic embedding module
        hidden_dim: Hidden dimension for query projection
        num_layers: Number of layers for query MLP
        use_graph_context: Whether to use the graph context
        check_nan: Whether to check for nan values during decoding
        tanh_clipping: Tanh clipping value for logits
    """

    def __init__(
        self,
        embed_dim: int = 128,
        env_name: str = "tsp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_graph_context: bool = True,
        check_nan: bool = True,
        tanh_clipping: float = 10.0,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.check_nan = check_nan
        self.tanh_clipping = tanh_clipping

        self.context_embedding = (
            env_context_embedding(self.env_name, {"embed_dim": embed_dim})
            if context_embedding is None
            else context_embedding
        )
        self.dynamic_embedding = (
            env_dynamic_embedding(self.env_name, {"embed_dim": embed_dim})
            if dynamic_embedding is None
            else dynamic_embedding
        )
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )

        # Project nodes to keys (precomputed once)
        self.project_node_keys = nn.Linear(embed_dim, embed_dim, bias=False)

        # Project query through MLP: query -> hidden -> embed_dim
        # This processes the context to create a query vector
        query_layers = []
        in_dim = embed_dim
        for i in range(num_layers - 1):
            query_layers.append(nn.Linear(in_dim, hidden_dim))
            query_layers.append(nn.ReLU())
            in_dim = hidden_dim
        query_layers.append(nn.Linear(in_dim, embed_dim))
        self.query_mlp = nn.Sequential(*query_layers)

        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)
        self.use_graph_context = use_graph_context

        # Scaling factor for dot product
        self.scale = 1.0 / math.sqrt(embed_dim)

    def _compute_query(self, cached: MLPPrecomputedCache, td: TensorDict):
        """Compute the query vector from context"""
        node_embeds_cache = cached.node_embeddings
        graph_context_cache = cached.graph_context

        if td.dim() == 2 and isinstance(graph_context_cache, Tensor):
            graph_context_cache = graph_context_cache.unsqueeze(1)

        step_context = self.context_embedding(node_embeds_cache, td)
        query = step_context + graph_context_cache

        return query

    def forward(
        self,
        td: TensorDict,
        cached: MLPPrecomputedCache,
        num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state

        Args:
            cached: Precomputed embeddings
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """

        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1

        # Handle efficient multi-start decoding
        if has_dyn_emb_multi_start:
            cached = cached.batchify(num_starts=num_starts)
        elif num_starts > 1:
            td = unbatchify(td, num_starts)

        # Get query (context) vector: [batch, embed_dim]
        query = self._compute_query(cached, td)

        # Transform query through MLP: [batch, embed_dim]
        query = self.query_mlp(query)

        # Get precomputed node keys: [batch, num_nodes, embed_dim]
        node_keys = cached.node_keys

        # Handle dynamic embeddings if present
        if self.is_dynamic_embedding:
            glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(td)
            node_keys = node_keys + logit_k_dyn

        # Efficient dot-product scoring (no large intermediate tensors!)
        # query: [batch, embed_dim] -> [batch, 1, embed_dim]
        # node_keys: [batch, num_nodes, embed_dim]
        # logits = query @ node_keys.T -> [batch, num_nodes]
        logits = torch.einsum('bd,bnd->bn', query, node_keys) * self.scale

        # Apply tanh clipping
        if self.tanh_clipping > 0:
            logits = self.tanh_clipping * torch.tanh(logits)

        # Get action mask
        mask = td["action_mask"]

        # Apply mask
        if num_starts > 1 and not has_dyn_emb_multi_start:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)

        # Mask invalid actions
        logits[~mask] = float("-inf")

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits, mask

    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, MLPPrecomputedCache]:
        """Precompute the embeddings cache before the decoder is called"""
        return td, env, self._precompute_cache(embeddings, num_starts=num_starts)

    def _precompute_cache(
        self, embeddings: torch.Tensor, num_starts: int = 0
    ) -> MLPPrecomputedCache:
        """Compute the cached embeddings.

        Args:
            embeddings: Precomputed embeddings for the nodes
            num_starts: Number of starts for the multi-start decoding
        """
        # Precompute node keys (done once, reused every step!)
        node_keys = self.project_node_keys(embeddings)

        # Optionally compute graph context from the initial embedding
        if self.use_graph_context:
            graph_context = self.project_fixed_context(embeddings.mean(1))
        else:
            graph_context = 0

        return MLPPrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            node_keys=node_keys,
        )
