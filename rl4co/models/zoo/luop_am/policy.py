from typing import Callable

import torch.nn as nn

from rl4co.models.common.constructive.autoregressive.encoder import AutoregressiveEncoder
from rl4co.models.common.constructive.autoregressive.decoder import AutoregressiveDecoder
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from rl4co.models.zoo.luop_am.constructive import LUOPConstructivePolicy
from rl4co.models.zoo.luop_am.decoder import LUOPAttentionModelDecoder


class LUOPAutoregressivePolicy(LUOPConstructivePolicy):
    """Autoregressive wrapper for LUOP constructive policies."""

    def __init__(
        self,
        encoder: AutoregressiveEncoder,
        decoder: AutoregressiveDecoder,
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
        if decoder is None:
            raise ValueError("LUOPAutoregressivePolicy requires a decoder.")

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
            decode_type_first=decode_type_first,
            **unused_kw,
        )


class LUOPAttentionModelPolicy(LUOPAutoregressivePolicy):
    """Attention policy specialized for LUOP joint type-parcel decoding."""

    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        env_name: str = "luop",
        encoder_network: nn.Module = None,
        init_embedding: nn.Module = None,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        use_graph_context: bool = True,
        linear_bias_decoder: bool = False,
        sdpa_fn: Callable = None,
        sdpa_fn_encoder: Callable = None,
        sdpa_fn_decoder: Callable = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        check_nan: bool = True,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        decode_type_first: bool = False,
        moe_kwargs: dict = None,
        **unused_kwargs,
    ):
        if moe_kwargs is None:
            moe_kwargs = {"encoder": None, "decoder": None}

        if encoder is None:
            encoder = AttentionModelEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                env_name=env_name,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                net=encoder_network,
                init_embedding=init_embedding,
                sdpa_fn=sdpa_fn if sdpa_fn_encoder is None else sdpa_fn_encoder,
                moe_kwargs=moe_kwargs["encoder"],
            )

        if decoder is None:
            decoder = LUOPAttentionModelDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                sdpa_fn=sdpa_fn if sdpa_fn_decoder is None else sdpa_fn_decoder,
                mask_inner=mask_inner,
                out_bias_pointer_attn=out_bias_pointer_attn,
                linear_bias=linear_bias_decoder,
                use_graph_context=use_graph_context,
                check_nan=check_nan,
                moe_kwargs=moe_kwargs["decoder"],
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
            decode_type_first=decode_type_first,
            **unused_kwargs,
        )
