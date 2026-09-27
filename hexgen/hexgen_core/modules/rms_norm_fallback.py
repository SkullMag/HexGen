"""Unfused RMSNorm for FlashAttention wheels without dropout_layer_norm."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)
