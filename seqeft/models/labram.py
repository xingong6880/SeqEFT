# SPDX-License-Identifier: GPL-3.0-only
"""LaBraM temporal encoder and checkpoint loader."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping, Tuple

import torch
from torch import nn
from torch.nn import functional as F

# Indices in LaBraM's channel position table, starting with CLS.
LABRAM_VU_INPUT_INDICES = (
    0, 1, 3, 18, 22, 40, 44, 62, 66, 81, 83, 16, 24,
    38, 46, 60, 68, 2, 20, 42, 64, 75, 82, 26, 36, 48, 58,
)


def _drop_path(
    value: torch.Tensor,
    probability: float,
    training: bool,
) -> torch.Tensor:
    """Apply stochastic depth independently per sample during training."""
    if probability == 0.0 or not training:
        return value
    keep_probability = 1.0 - probability
    shape = (value.shape[0],) + (1,) * (value.ndim - 1)
    mask = value.new_empty(shape).bernoulli_(keep_probability)
    return value.div(keep_probability) * mask


class DropPath(nn.Module):

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _drop_path(value, self.probability, self.training)


class MLP(nn.Module):

    def __init__(self, width: int, hidden_width: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, hidden_width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.fc1(value)
        value = self.act(value)
        value = self.fc2(value)
        return self.drop(value)


class Attention(nn.Module):
    """Multi-head attention with the LaBraM-base parameter layout."""

    def __init__(
        self,
        width: int,
        heads: int,
        *,
        qkv_bias: bool,
        attention_dropout: float,
        projection_dropout: float,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("The temporal width must be divisible by the number of heads")
        self.num_heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(width, width * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(width))
            self.v_bias = nn.Parameter(torch.zeros(width))
        else:
            self.q_bias = None
            self.v_bias = None
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(width, width)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = value.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias), self.v_bias)
            )
        qkv = F.linear(value, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value_tensor = qkv.unbind(0)
        query = self.q_norm(query) * self.scale
        key = self.k_norm(key)
        weights = self.attn_drop((query @ key.transpose(-2, -1)).softmax(dim=-1))
        output = (weights @ value_tensor).transpose(1, 2).reshape(
            batch,
            tokens,
            width,
        )
        return self.proj_drop(self.proj(output))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with LaBraM LayerScale parameters."""

    def __init__(
        self,
        width: int,
        heads: int,
        *,
        mlp_ratio: float,
        qkv_bias: bool,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        init_values: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = Attention(
            width,
            heads,
            qkv_bias=qkv_bias,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = MLP(width, int(width * mlp_ratio), dropout)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(width))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.drop_path(
            self.gamma_1 * self.attn(self.norm1(value))
        )
        return value + self.drop_path(
            self.gamma_2 * self.mlp(self.norm2(value))
        )


class TemporalConv(nn.Module):
    """Encode each of the 416 EEG patches as a 200-dimensional token."""

    def __init__(self, out_channels: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            1,
            out_channels,
            kernel_size=(1, 15),
            stride=(1, 8),
            padding=(0, 7),
        )
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
        )
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
        )
        self.norm3 = nn.GroupNorm(4, out_channels)
        self.gelu3 = nn.GELU()

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        batch, channels, patches, samples = eeg.shape
        value = eeg.reshape(batch, 1, channels * patches, samples)
        value = self.gelu1(self.norm1(self.conv1(value)))
        value = self.gelu2(self.norm2(self.conv2(value)))
        value = self.gelu3(self.norm3(self.conv3(value)))
        return value.permute(0, 2, 3, 1).reshape(
            batch,
            channels * patches,
            -1,
        )


class LaBraMTemporalEncoder(nn.Module):
    """Temporal trunk compatible with LaBraM-base checkpoints."""

    def __init__(
        self,
        *,
        embed_dim: int = 200,
        depth: int = 12,
        num_heads: int = 10,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        init_values: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = TemporalConv(out_channels=8)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 129, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        drop_paths = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=drop_rate,
                attention_dropout=attn_drop_rate,
                drop_path=drop_paths[index],
                init_values=init_values,
            )
            for index in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._initialize_module)
        self._rescale_projection_weights()

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _rescale_projection_weights(self) -> None:
        for layer_id, block in enumerate(self.blocks, start=1):
            denominator = math.sqrt(2.0 * layer_id)
            block.attn.proj.weight.data.div_(denominator)
            block.mlp.fc2.weight.data.div_(denominator)

    def _forward_raw_patch_tokens(
        self,
        eeg: torch.Tensor,
        input_chans: Tuple[int, ...] = LABRAM_VU_INPUT_INDICES,
    ) -> torch.Tensor:
        if eeg.ndim != 4 or tuple(eeg.shape[1:]) != (26, 16, 200):
            raise ValueError(
                "Expected LaBraM input shape [NB, 26, 16, 200], got {}".format(
                    tuple(eeg.shape)
                )
            )
        if len(input_chans) != 27 or input_chans[0] != 0:
            raise ValueError("Expected 27 LaBraM channel indices including CLS")
        batch = eeg.shape[0]
        tokens = self.patch_embed(eeg)
        cls = self.cls_token.expand(batch, -1, -1)
        value = torch.cat((cls, tokens), dim=1)
        selected = self.pos_embed[:, list(input_chans)]
        channel_positions = selected[:, 1:, :].unsqueeze(2).expand(
            batch,
            26,
            16,
            self.embed_dim,
        ).flatten(1, 2)
        positions = torch.cat((selected[:, :1, :].expand(batch, -1, -1), channel_positions), dim=1)
        value = value + positions
        time_positions = self.time_embed.expand(batch, -1, -1).unsqueeze(1).expand(
            batch,
            26,
            16,
            self.embed_dim,
        ).flatten(1, 2)
        value[:, 1:, :] = value[:, 1:, :] + time_positions
        value = self.pos_drop(value)
        for block in self.blocks:
            value = block(value)
        return value[:, 1:, :]

    def forward_patch_tokens(
        self,
        eeg: torch.Tensor,
        input_chans: Tuple[int, ...] = LABRAM_VU_INPUT_INDICES,
    ) -> torch.Tensor:
        """Return normalized features for each EEG patch."""
        return self.norm(self._forward_raw_patch_tokens(eeg, input_chans))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.norm(self._forward_raw_patch_tokens(eeg).mean(dim=1))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

@dataclass(frozen=True)
class CheckpointLoadReport:
    """Record checkpoint coverage, ignored keys, and file hash."""

    expected_core: int
    loaded_core: int
    ignored: Tuple[str, ...]
    checkpoint_sha256: str = ""


def load_temporal_checkpoint(
    module: nn.Module,
    checkpoint_path: Path,
) -> CheckpointLoadReport:
    """Validate the temporal weights before loading them into the module."""
    source = Path(checkpoint_path)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = (
        checkpoint.get("model", checkpoint)
        if isinstance(checkpoint, Mapping)
        else checkpoint
    )
    if not isinstance(state, Mapping):
        raise RuntimeError("The LaBraM checkpoint has no usable state dictionary")
    student = {
        key[len("student."):]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("student.")
    }
    target = module.state_dict()
    mismatches = []
    for key, target_value in target.items():
        if key not in student:
            mismatches.append("{}: missing".format(key))
        elif not isinstance(student[key], torch.Tensor):
            mismatches.append("{}: not a tensor".format(key))
        elif student[key].shape != target_value.shape:
            mismatches.append(
                "{}: checkpoint {} != model {}".format(
                    key,
                    tuple(student[key].shape),
                    tuple(target_value.shape),
                )
            )
    if mismatches:
        raise RuntimeError(
            "LaBraM temporal checkpoint mismatch: {}".format(
                "; ".join(mismatches[:10])
            )
        )

    selected = {key: student[key] for key in target}
    module.load_state_dict(selected, strict=True)
    loaded_sources = {"student." + key for key in target}
    ignored = tuple(sorted(str(key) for key in state if key not in loaded_sources))
    return CheckpointLoadReport(
        expected_core=len(target),
        loaded_core=len(selected),
        ignored=ignored,
        checkpoint_sha256=_sha256_path(source),
    )
