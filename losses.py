from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

import config as C

SQRT_PI = math.sqrt(math.pi)


def depth_weights(depth: torch.Tensor, k: float = C.DEPTH_WEIGHT_K) -> torch.Tensor:
    return depth / (depth + k)


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                 weights: torch.Tensor | None = None) -> torch.Tensor:
    per = 0.5 * torch.log(sigma**2) + (y - mu) ** 2 / (2 * sigma**2)
    per = per.mean(dim=1)
    if weights is not None:
        return (per * weights).sum() / weights.sum().clamp_min(C.EPS)
    return per.mean()


def focal_loss(logit: torch.Tensor, target: torch.Tensor,
               gamma: float = C.FOCAL_GAMMA, alpha: float = C.FOCAL_ALPHA) -> torch.Tensor:
    p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
    pt = torch.where(target > 0.5, p, 1 - p)
    at = torch.where(target > 0.5, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
    return (-at * (1 - pt) ** gamma * torch.log(pt)).mean()


def masked_reconstruction(
    recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return ((recon - target) ** 2 * m).sum() / m.sum().clamp_min(C.EPS) / target.shape[-1]


def ppo_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, advantage: torch.Tensor,
             eps: float = C.PPO_CLIP_EPS) -> torch.Tensor:
    rho = torch.exp(logp_new - logp_old)
    return -torch.min(rho * advantage, rho.clamp(1 - eps, 1 + eps) * advantage).mean()


def reinforce_loss(logp: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    return -(logp * advantage).mean()


def value_loss(value: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(value, returns)


def dirichlet_entropy(alpha: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Dirichlet(alpha).entropy().mean()


def l2_penalty(model: torch.nn.Module) -> torch.Tensor:
    return sum((p**2).sum() for p in model.parameters() if p.requires_grad)


def gae(rewards: np.ndarray, values: np.ndarray,
        gamma: float = C.GAE_GAMMA, lam: float = C.GAE_LAMBDA) -> tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in range(n - 1, -1, -1):
        next_v = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def mae(y: np.ndarray, mu: np.ndarray) -> float:
    return float(np.mean(np.abs(y - mu)))


def rmse(y: np.ndarray, mu: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - mu) ** 2)))


def mase(y: np.ndarray, mu: np.ndarray, naive: np.ndarray) -> float:
    denom = np.mean(np.abs(y - naive))
    return float(np.mean(np.abs(y - mu)) / max(denom, C.EPS))


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.mean(crps_pointwise(y, mu, sigma)))


def crps_pointwise(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.clip(sigma, 1e-6, None)
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / SQRT_PI)


def picp(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, level: float) -> float:
    half = stats.norm.ppf(0.5 + level / 2) * np.clip(sigma, 1e-6, None)
    return float(np.mean((y >= mu - half) & (y <= mu + half)))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def lead_time(t_end: np.ndarray, region: np.ndarray, lineage: np.ndarray,
              y_abs: np.ndarray, mu_abs: np.ndarray, y_hist: np.ndarray) -> np.ndarray:
    dom = math.log(C.DOMINANCE_THRESHOLD / (1 - C.DOMINANCE_THRESHOLD))
    out: list[float] = []
    for key in {(int(r), int(li)) for r, li in zip(region, lineage, strict=True)}:
        idx = np.flatnonzero((region == key[0]) & (lineage == key[1]))
        if idx.size == 0:
            continue
        idx = idx[np.argsort(t_end[idx])]

        t_dom = np.inf
        for i in idx:
            hit = np.flatnonzero(y_abs[i] >= dom)
            if hit.size:
                t_dom = min(t_dom, float(t_end[i] + 1 + hit[0]))
        if not np.isfinite(t_dom):
            continue

        t_alarm = np.inf
        for i in idx:
            if y_hist[i, -1] >= dom:
                continue
            if t_end[i] >= t_dom:
                continue
            if np.any(mu_abs[i] >= dom):
                t_alarm = float(t_end[i])
                break
        if np.isfinite(t_alarm):
            out.append(t_dom - t_alarm)
    return np.asarray(out, dtype=np.float64)


def evaluate(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, naive: np.ndarray,
             event: np.ndarray, event_score: np.ndarray,
             lt: np.ndarray | None = None) -> dict[str, float]:
    lt = np.asarray([]) if lt is None else np.asarray(lt)
    out = {
        "MAE": mae(y, mu),
        "RMSE": rmse(y, mu),
        "MASE": mase(y, mu, naive),
        "CRPS": crps_gaussian(y, mu, sigma),
        "PICP50": picp(y, mu, sigma, 0.50),
        "PICP80": picp(y, mu, sigma, 0.80),
        "LT": float(np.mean(lt)) if lt.size else float("nan"),
        "AUPRC": average_precision(event, event_score),
    }
    missing = {m.key for m in C.METRICS} - set(out)
    if missing:
        raise AssertionError(f"metrics {sorted(missing)} declared but not computed")
    return out


def mae_by_horizon(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.abs(y - mu).mean(axis=0)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = d.size
    if n < 2:
        return float("nan"), float("nan")
    var = d.var(ddof=1)
    if var <= 0:
        return float("nan"), float("nan")
    dm = float(d.mean() / math.sqrt(var / n))
    return dm, float(2 * (1 - stats.norm.cdf(abs(dm))))


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    thresh = alpha * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.flatnonzero(passed)[-1]
        keep[order[: cutoff + 1]] = True
    return keep
