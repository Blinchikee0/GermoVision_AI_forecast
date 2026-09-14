from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class CausalConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, dilation: int) -> None:
        super().__init__()
        self.pad = dilation * (C.CNN_KERNEL - 1)
        self.conv = nn.Conv1d(ch_in, ch_out, C.CNN_KERNEL, dilation=dilation)
        self.norm = nn.BatchNorm1d(ch_out)
        self.drop = nn.Dropout(C.CNN_DROPOUT)
        self.project = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pad(x, (self.pad, 0))
        h = self.conv(h)
        h = self.drop(F.gelu(self.norm(h)))
        return h + self.project(x)


class CausalDilatedCNN(nn.Module):
    def __init__(self, n_features: int = C.N_FEATURES) -> None:
        super().__init__()
        chans = [n_features] + [C.CNN_CHANNELS] * C.CNN_LAYERS
        self.blocks = nn.ModuleList(
            CausalConvBlock(chans[i], chans[i + 1], d) for i, d in enumerate(C.CNN_DILATIONS)
        )
        assert C.RECEPTIVE_FIELD > C.T_IN, (
            f"receptive field {C.RECEPTIVE_FIELD} does not cover window {C.T_IN}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        return h.transpose(1, 2)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        scale = -math.log(10000.0) / d_model
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * scale)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, d_model: int = C.D_MODEL, heads: int = C.ATTN_HEADS, d_k: int = C.ATTN_DK) -> None:
        super().__init__()
        self.h, self.d_k = heads, d_k
        inner = heads * d_k
        self.q = nn.Linear(d_model, inner)
        self.k = nn.Linear(d_model, inner)
        self.v = nn.Linear(d_model, inner)
        self.out = nn.Linear(inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(C.CNN_DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        shape = (b, t, self.h, self.d_k)
        q = self.q(x).view(shape).transpose(1, 2)
        k = self.k(x).view(shape).transpose(1, 2)
        v = self.v(x).view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        ctx = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, t, self.h * self.d_k)
        return self.norm(x + self.drop(self.out(ctx)))


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int = C.D_MODEL) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.score(x).squeeze(-1).softmax(dim=1)
        return (a.unsqueeze(-1) * x).sum(dim=1), a


class GermoVisionNet(nn.Module):
    def __init__(self, n_features: int = C.N_FEATURES, n_regions: int = C.N_REGIONS,
                 horizon: int = C.HORIZON, *, use_lstm: bool = True,
                 use_attention: bool = True, use_prob_head: bool = True) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_regions = n_regions
        self.use_lstm = use_lstm
        self.use_attention = use_attention
        self.use_prob_head = use_prob_head

        self.cnn = CausalDilatedCNN(n_features)
        d_seq = C.CNN_CHANNELS

        if use_lstm:
            self.lstm = nn.LSTM(
                C.CNN_CHANNELS, C.LSTM_HIDDEN, num_layers=C.LSTM_LAYERS,
                batch_first=True, bidirectional=C.LSTM_BIDIRECTIONAL,
                dropout=C.CNN_DROPOUT if C.LSTM_LAYERS > 1 else 0.0,
            )
            d_seq = C.D_MODEL
        else:
            self.widen = nn.Linear(C.CNN_CHANNELS, C.D_MODEL)
            d_seq = C.D_MODEL

        if use_attention:
            self.posenc = SinusoidalPositionalEncoding(d_seq)
            self.attn = MultiHeadSelfAttention(d_seq)

        self.pool = AttentionPooling(d_seq)

        self.head_prob = nn.Sequential(
            nn.Linear(d_seq, 256), nn.GELU(), nn.Dropout(C.CNN_DROPOUT),
            nn.Linear(256, horizon * 2 if use_prob_head else horizon),
        )
        self.head_event = nn.Sequential(nn.Linear(d_seq, 128), nn.GELU(), nn.Linear(128, 1))
        self.head_policy = nn.Sequential(
            nn.Linear(d_seq, 128), nn.GELU(), nn.Linear(128, n_regions))
        self.head_critic = nn.Sequential(nn.Linear(d_seq, 128), nn.GELU(), nn.Linear(128, 1))
        self.head_recon = nn.Linear(d_seq, n_features)

        self.register_buffer("fixed_sigma", torch.tensor(0.5))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.cnn(x)
        h = self.lstm(h)[0] if self.use_lstm else self.widen(h)
        if self.use_attention:
            h = self.attn(self.posenc(h))
        pooled, a = self.pool(h)
        return pooled, a, h

    def encoder_parameter_groups(self) -> list[list[nn.Parameter]]:
        groups: list[list[nn.Parameter]] = [list(b.parameters()) for b in self.cnn.blocks]
        if self.use_lstm:
            groups.append(list(self.lstm.parameters()))
        else:
            groups.append(list(self.widen.parameters()))
        if self.use_attention:
            groups.append(list(self.attn.parameters()))
        groups.append(list(self.pool.parameters()))
        return groups

    def set_encoder_grad(self, requires: bool) -> None:
        for group in self.encoder_parameter_groups():
            for p in group:
                p.requires_grad_(requires)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled, a, seq = self.encode(x)

        raw = self.head_prob(pooled)
        if self.use_prob_head:
            mu, s_raw = raw[:, : self.horizon], raw[:, self.horizon :]
            sigma = F.softplus(s_raw) + 1e-3
        else:
            mu = raw
            sigma = self.fixed_sigma.expand_as(mu)

        alpha = F.softplus(self.head_policy(pooled)) + 1.0
        return {
            "mu": mu,
            "sigma": sigma,
            "event_logit": self.head_event(pooled).squeeze(-1),
            "alpha": alpha,
            "value": self.head_critic(pooled).squeeze(-1),
            "recon": self.head_recon(seq),
            "attention": a,
            "embedding": pooled,
        }

    def forward_export(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        o = self(x)
        return o["mu"], o["sigma"], o["event_logit"], o["alpha"], o["attention"]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ExportWrapper(nn.Module):
    def __init__(self, net: GermoVisionNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.net.forward_export(x)


class SingleLSTM(nn.Module):
    def __init__(self, n_features: int = C.N_FEATURES, horizon: int = C.HORIZON) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, C.LSTM_HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Linear(C.LSTM_HIDDEN, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.lstm(x)[0][:, -1])


def count_flops(model: nn.Module, seq_len: int = C.T_IN, n_features: int = C.N_FEATURES) -> int:
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            flops += m.in_channels * m.out_channels * m.kernel_size[0] * seq_len
        elif isinstance(m, nn.Linear):
            flops += m.in_features * m.out_features
        elif isinstance(m, nn.LSTM):
            d = 2 if m.bidirectional else 1
            for layer in range(m.num_layers):
                inp = m.input_size if layer == 0 else m.hidden_size * d
                flops += d * seq_len * 4 * m.hidden_size * (inp + m.hidden_size)
    for m in model.modules():
        if isinstance(m, MultiHeadSelfAttention):
            flops += 2 * m.h * seq_len * seq_len * m.d_k
    del n_features
    return int(flops)
