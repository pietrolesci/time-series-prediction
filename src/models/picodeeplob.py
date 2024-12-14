"""
The Pico Model: A Lightweight Transformer Language Model

Pico uses a simple LLAMA-style transformer architecture, written for clarity and educational purposes.

Everything is written with a modular design for easy modification and experimentation.

Key features:
- RMSNorm for layer normalization
- Rotary Positional Embeddings (RoPE)
- Multi-head attention with KV-cache support
- SwiGLU activation function
- Residual connections throughout

- KV-cache for faster autoregressive generation

References:
    - RoPE: https://arxiv.org/abs/2104.09864
    - SwiGLU: https://arxiv.org/abs/2002.05202
    - LLAMA: https://arxiv.org/abs/2302.13971

Adapted from:
    - OLMO: https://github.com/allenai/OLMo
    - LLAMA: https://github.com/meta/llama
"""

import math
from dataclasses import dataclass

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import seed_everything
from torch import Tensor

from src.models.deeplob import DeepLOBConfig
from src.utilities import TOTAL_NUMBER_OF_CLASSES


@dataclass
class PicoDeepLOBConfig(DeepLOBConfig):
    n_layers: int = 2
    max_seq_len: int = 128
    attention_n_heads: int = 8
    attention_n_kv_heads: int = 4
    activation_hidden_dim: int = 256
    norm_eps: float = 1e-6
    position_emb_theta: float = 10000.0

    @property
    def name(self) -> str:
        return f"picodeeplob-{'clf' if self.is_classification else 'reg'}-{self.num_levels}"

    def get_model(self) -> nn.Module:
        seed_everything(42)
        return PicoDeepLOB(self)


class PicoDeepLOB(nn.Module):
    def __init__(self, config: PicoDeepLOBConfig) -> None:
        super().__init__()

        # Convolution blocks
        self.conv1 = self._get_conv_block(1, config.conv_out_channels, (1, 2), (1, 2))
        self.conv2 = self._get_conv_block(config.conv_out_channels, config.conv_out_channels, (1, 2), (1, 2))
        self.conv3 = self._get_conv_block(
            config.conv_out_channels, config.conv_out_channels, (1, config.num_levels), (1, 1)
        )

        # Inception blocks
        self.inp1 = self._get_inception_block(config.conv_out_channels, config.incep_out_channels, (3, 1))
        self.inp2 = self._get_inception_block(config.conv_out_channels, config.incep_out_channels, (5, 1))
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(config.conv_out_channels, config.incep_out_channels, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(config.incep_out_channels),
        )

        model_dim = config.incep_out_channels * 3

        # Pico blocks
        self.pico = Pico(
            n_layers=config.n_layers,
            model_dim=model_dim,
            attention_n_heads=config.attention_n_heads,
            attention_n_kv_heads=config.attention_n_kv_heads,
            activation_hidden_dim=config.activation_hidden_dim,
            max_seq_len=config.max_seq_len,
            theta=config.position_emb_theta,
            eps=config.norm_eps,
        )

        # Regression or classification output
        if config.is_classification:
            self.fc = nn.Linear(model_dim, TOTAL_NUMBER_OF_CLASSES)
        else:
            self.fc = nn.Sequential(nn.Linear(model_dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        # Add channel dimention
        # x: batch_size, n_channels, seq_len, num_levels * 4
        x = x.unsqueeze(1)

        # Convolution blocks
        # x: batch_size, n_channels, seq_len, num_levels * 2
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Inception blocks
        # batch_size, incep_out_channels, seq_len, 1
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        # Concatenate inception blocks
        # batch_size, incep_out_channels * 3, seq_len, 1
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        # Reshape to prepare for LSTM
        # batch_size, seq_len, incep_out_channels * 3
        # Changed with a simpler to understand version
        # >>> x = x.permute(0, 2, 1, 3)
        # >>> x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x = x.squeeze(-1).permute(0, 2, 1)

        # batch_size, seq_len, incep_out_channels * 3
        x, _ = self.pico(x)

        # batch_size, seq_len
        return self.fc(x).squeeze(-1)

    def _get_conv_block(
        self, in_channels: int, out_channels: int, first_kernel_size: tuple[int, int], first_stride: tuple[int, int]
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=first_kernel_size, stride=first_stride),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels),
        )

    def _get_inception_block(
        self, in_channels: int, out_channels: int, mid_kernel_size: tuple[int, int]
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=mid_kernel_size, padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels),
        )


########################################################
# Layer Normalization
########################################################


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    A variant of Layer Normalization that uses RMS statistics instead of mean/variance,
    resulting in improved stability and performance.

    References:
        https://arxiv.org/abs/1910.07467
    """

    def __init__(self, model_dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(model_dim))

    def _norm(self, x: Tensor) -> Tensor:
        """
        Normalizes the input tensor by its RMS value.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies RMS normalization to the input tensor and scales it by the weight parameter.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


########################################################
# Positional Embedding
########################################################


class RoPE(nn.Module):
    """Rotary Positional Embeddings (RoPE).

    Implements position-dependent rotation of keys and queries in attention mechanism,
    allowing better modeling of relative positions in sequences. Uses complex number
    operations for efficient rotation.

    References:
        https://arxiv.org/abs/2104.09864
    """

    def __init__(self, model_dim: int, attention_n_heads: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        self._freqs_cis = self._setup_freqs_cis(max_seq_len, theta, model_dim // attention_n_heads)

    def _setup_freqs_cis(self, seq_len: int, theta: float, dim: int) -> Tensor:
        """
        Sets up the complex frequency tensor that is used to compute the RoPE embeddings.

        Note other implementations will use cos and sin directly, but using the complex
        number representation is (probably?) more efficient:

            e^(theta * i * t) = cos(theta * t) + i * sin(theta * t) [Euler's formula]
        """
        _freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        positions = torch.arange(seq_len)
        freqs = torch.outer(positions, _freqs)
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64

    def get_freqs_cis(self, input_shape: torch.Size, start_pos: int, end_pos: int) -> Tensor:
        """
        Reshapes the frequency tensor to be broadcastable with the input tensor.
        """
        _freqs_cis = self._freqs_cis[start_pos:end_pos]
        ndim = len(input_shape)
        assert 0 <= 1 < ndim
        assert _freqs_cis.shape == (input_shape[1], input_shape[-1])

        # TODO: Check whether this is correct (might be able to remove this)
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(input_shape)]
        return _freqs_cis.view(*shape)

    def forward(self, queries: Tensor, keys: Tensor, start_pos: int = 0) -> tuple[Tensor, Tensor]:
        """
        Applies the rotary positional embeddings to the input tensors via complex num multiplication

        NOTE: The start_pos is used if we want to use the kv_cache in the attention mechanism.
        """
        queries_ = torch.view_as_complex(queries.float().reshape(*queries.shape[:-1], -1, 2))
        keys_ = torch.view_as_complex(keys.float().reshape(*keys.shape[:-1], -1, 2))

        input_shape = queries_.shape  # same as keys: (batch_size, seq_len, n_heads, head_dim/2)
        freqs_start_pos = start_pos
        freqs_end_pos = freqs_start_pos + queries_.shape[1]

        freqs_cis = self.get_freqs_cis(input_shape, freqs_start_pos, freqs_end_pos).to(queries.device)

        queries_rotated = torch.view_as_real(queries_ * freqs_cis).flatten(3)
        keys_rotated = torch.view_as_real(keys_ * freqs_cis).flatten(3)
        return queries_rotated.type_as(queries), keys_rotated.type_as(keys)


########################################################
# Attention
########################################################


class Attention(nn.Module):
    """Multi-head Attention with Group Query Attention support.

    Implements scaled dot-product attention and supports:
    - Grouped Query Attention (GQA)
    - Key-Value caching for efficient inference
    - RoPE integration

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """

    def __init__(
        self, model_dim: int, attention_n_heads: int, attention_n_kv_heads: int, max_seq_len: int, theta: float
    ) -> None:
        super().__init__()

        self.n_heads = attention_n_heads
        self.n_kv_heads = attention_n_kv_heads
        self.head_dim = model_dim // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        # Layers
        self.q_proj = nn.Linear(model_dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, model_dim, bias=False)
        self.rope = RoPE(model_dim=model_dim, attention_n_heads=attention_n_heads, max_seq_len=max_seq_len, theta=theta)

    def forward(
        self, input: Tensor, mask: Tensor | None = None, past_key_values: tuple | None = None, use_cache: bool = False
    ) -> tuple[Tensor, tuple]:
        bsz, seq_len, _ = input.shape
        _queries, _keys, _values = (self.q_proj(input), self.k_proj(input), self.v_proj(input))

        # Reshaping for multi-head attention
        queries = _queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = _keys.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        values = _values.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # The start position is used to apply the RoPE embeddings to only the new tokens
        # when using the kv_cache in the attention mechanism.
        # We want to start from the last position in the cache.
        start_pos = past_key_values[0].shape[1] if past_key_values is not None else 0

        # apply rotary positional embeddings
        queries, keys = self.rope(queries, keys, start_pos)

        if past_key_values is not None:
            keys = torch.cat([past_key_values[0], keys], dim=1)
            values = torch.cat([past_key_values[1], values], dim=1)

        if use_cache:
            cached_keys = keys
            cached_values = values
        else:
            cached_keys = None
            cached_values = None

        if self.n_rep > 1:
            keys = torch.repeat_interleave(keys, self.n_rep, dim=2)
            values = torch.repeat_interleave(values, self.n_rep, dim=2)

        # Dimension of queries: (bs, n_heads, seq_len, head_dim)
        # Dimension of keys/values: (bs, n_kv_heads, (cache_len) + seq_len, head_dim)j
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, seq_len, (cache_len) + seq_len)
        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        output = torch.matmul(scores, values)  # (bs, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(output)

        return output, (cached_keys, cached_values)


########################################################
# SwiGLU (Combines MLP and Activation)
########################################################


class SwiGLU(nn.Module):
    """SwiGLU Activation Function with Linear Projections.

    Implements the SwiGLU activation function combined with linear transformations,
    serving as the feed-forward network in transformer blocks.

    Args:
        config (Union[PicoDeepLOBConfig, PicoHFConfig]): Configuration containing:
            - config.model_dim: Model dimension
            - config.activation_hidden_dim: Hidden dimension (typically 4 * d_model)

    References:
        https://arxiv.org/abs/2002.05202
    """

    def __init__(self, model_dim: int, act_hidden_dim: int) -> None:
        super().__init__()
        self.w_0 = nn.Linear(model_dim, act_hidden_dim, bias=False)
        self.w_1 = nn.Linear(model_dim, act_hidden_dim, bias=False)
        self.w_2 = nn.Linear(act_hidden_dim, model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(F.silu(self.w_0(x)) * self.w_1(x))


########################################################
# PicoBlock and the Pico Model
########################################################


class PicoBlock(nn.Module):
    """Single Transformer Block with Attention and Feed-forward layers.

    Implements a standard transformer block with:
    - Multi-head attention with normalization and residual connection
    - SwiGLU feed-forward network with normalization and residual connection
    """

    def __init__(
        self,
        model_dim: int,
        attention_n_heads: int,
        attention_n_kv_heads: int,
        activation_hidden_dim: int,
        max_seq_len: int,
        theta: float,
        eps: float,
    ) -> None:
        super().__init__()

        self.attention = Attention(
            model_dim=model_dim,
            attention_n_heads=attention_n_heads,
            attention_n_kv_heads=attention_n_kv_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.feed_forward = SwiGLU(model_dim=model_dim, act_hidden_dim=activation_hidden_dim)
        self.attention_norm = RMSNorm(model_dim=model_dim, eps=eps)
        self.swiglu_norm = RMSNorm(model_dim=model_dim, eps=eps)

    def forward(
        self,
        input: Tensor,
        mask: Tensor | None = None,
        past_key_values: tuple[Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        attention_output, cached_key_values = self.attention(
            self.attention_norm(input), mask=mask, past_key_values=past_key_values, use_cache=use_cache
        )
        # NOTE: cached_key_values is None if use_cache is False

        h = input + attention_output
        out = h + self.feed_forward(self.swiglu_norm(h))
        return out, cached_key_values


class Pico(nn.Module):
    """Core Pico implementation."""

    def __init__(
        self,
        n_layers: int,
        model_dim: int,
        attention_n_heads: int,
        attention_n_kv_heads: int,
        activation_hidden_dim: int,
        max_seq_len: int,
        theta: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PicoBlock(
                    model_dim=model_dim,
                    attention_n_heads=attention_n_heads,
                    attention_n_kv_heads=attention_n_kv_heads,
                    activation_hidden_dim=activation_hidden_dim,
                    max_seq_len=max_seq_len,
                    theta=theta,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )

        self.output_norm = RMSNorm(model_dim=model_dim, eps=eps)

    def forward(
        self,
        x: Tensor,  # batch_size, seq_len, model_dim
        past_key_values: tuple[tuple[Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple | None]:
        """
        This is the forward pass for the entire Pico model. It boils down to:
        - Embedding the input ids
        - Creating a causal mask
        - Processing through the pico layers
        - Projecting the output to logits

        NOTE: One feature that might be confusing is the KV cache. The KV cache is used to speed up
        generation by caching the KV pairs from previous forward passes. This is useful when doing
        tasks that require generating multiple tokens conditioned on previous tokens (e.g. language
        modeling, text generation, etc.). The way the KV cache is implemented is that each layer has
        its own KV cache, and we aggregate the KV pairs from each layer in a tuple.
        """

        seq_len = x.shape[1]

        # Calculate start position from past cached KV pairs. Remember that each layer has its
        # own KV Cache. So when we index past_key_values, we need to index into the KV pairs for the
        # correct layer and then for either the keys or values.
        start_pos = 0 if past_key_values is None else past_key_values[0][0].shape[1]

        # Create causal mask for current sequence
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)

            # If using KV cache, extend mask to cover cached sequence length
            if past_key_values is not None:
                # Add zeros for cached tokens (we can attend to all of them)
                mask = torch.hstack([torch.zeros((seq_len, start_pos)), mask]).type_as(x)

        # NOTE: If we are using the cache, we need to store the cached KV pairs for each layer
        #       in a tuple. Each layer will have its own cached KV pair which we aggregate in a tuple.
        cached_key_values = () if use_cache else None

        # Process through transformer blocks
        for idx, layer in enumerate(self.layers):
            layer_past_key_values = past_key_values[idx] if past_key_values is not None else None

            x, layer_cached_key_values = layer(x, mask=mask, past_key_values=layer_past_key_values, use_cache=True)

            if use_cache:
                cached_key_values += (layer_cached_key_values,)  # type: ignore

        # Final norm and projection
        x = self.output_norm(x)

        return x, cached_key_values
