from typing import Callable

import torch.nn as nn

from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from rl4co.models.zoo.mlp.decoder import MLPDecoder
from rl4co.models.zoo.mlp.encoder import MLPEncoder


class MLPPolicy(AutoregressivePolicy):
    """
    MLP-based Policy for combinatorial optimization problems.
    This model uses MLP layers instead of attention mechanism for both encoding and decoding.

    Memory-efficient design:
    - Encoder: MLP processes each node independently
    - Decoder: Uses dot-product scoring (precomputed keys, no large intermediate tensors)

    Args:
        encoder: Encoder module, defaults to :class:`MLPEncoder`
        decoder: Decoder module, defaults to :class:`MLPDecoder`
        embed_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        encoder_hidden_dim: Hidden dimension in the encoder MLP
        num_decoder_layers: Number of layers in the decoder query MLP
        decoder_hidden_dim: Hidden dimension in the decoder query MLP
        env_name: Name of the environment used to initialize embeddings
        init_embedding: Module to use for the initialization of the embeddings
        context_embedding: Module to use for the context embedding
        dynamic_embedding: Module to use for the dynamic embedding
        use_graph_context: Whether to use the graph context
        dropout: Dropout rate for encoder
        check_nan: Whether to check for nan values during decoding
        temperature: Temperature for the softmax (used in base policy)
        tanh_clipping: Tanh clipping value (see Bello et al., 2016)
        mask_logits: Whether to mask the logits during decoding
        train_decode_type: Type of decoding to use during training
        val_decode_type: Type of decoding to use during validation
        test_decode_type: Type of decoding to use during testing
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        encoder_hidden_dim: int = 512,
        num_decoder_layers: int = 2,
        decoder_hidden_dim: int = 128,
        env_name: str = "tsp",
        init_embedding: nn.Module = None,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        use_graph_context: bool = True,
        dropout: float = 0.0,
        check_nan: bool = True,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kwargs,
    ):
        if encoder is None:
            encoder = MLPEncoder(
                embed_dim=embed_dim,
                env_name=env_name,
                num_layers=num_encoder_layers,
                hidden_dim=encoder_hidden_dim,
                init_embedding=init_embedding,
                dropout=dropout,
            )

        if decoder is None:
            decoder = MLPDecoder(
                embed_dim=embed_dim,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                hidden_dim=decoder_hidden_dim,
                num_layers=num_decoder_layers,
                use_graph_context=use_graph_context,
                check_nan=check_nan,
                tanh_clipping=tanh_clipping,
            )

        super(MLPPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kwargs,
        )
