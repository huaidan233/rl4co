from typing import Tuple, Union

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding


class MLPEncoder(AutoregressiveEncoder):
    """MLP Encoder for combinatorial optimization problems.
    Uses a Multi-Layer Perceptron instead of Graph Attention Network.

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_layers: Number of MLP layers
        hidden_dim: Hidden dimension in the MLP layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "tsp",
        num_layers: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super(MLPEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )

        # Build MLP layers
        layers = []
        in_dim = embed_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, embed_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply (unused in MLP, kept for API compatibility)

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """
        # Transfer to embedding space
        init_h = self.init_embedding(td)

        # Process embedding through MLP
        # init_h shape: [batch_size, num_nodes, embed_dim]
        h = self.net(init_h)

        # Return latent representation and initial embedding
        return h, init_h
