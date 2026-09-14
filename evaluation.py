from __future__ import annotations

import logging
import warnings

import numpy as np
import torch
from scipy import optimize, stats

import config as C
import losses as L
from data import WindowSet
from model import SingleLSTM

log = logging.getLogger("germovision.eval")
warnings.filterwarnings("ignore", category=RuntimeWarning)


@torch.no_grad()
def predict(net: torch.nn.Module, x: np.ndarray, batch: int = 512) -> dict[str, np.ndarray]:
    net.eval()
    keys = ("mu", "sigma", "event_logit", "attention", "embedding", "alpha")
    acc: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for i in range(0, len(x), batch):
        out = net(torch.from_numpy(x[i : i + batch]))
        for k in keys:
            acc[k].append(out[k].cpu().numpy())
    res = {k: np.concatenate(v) for k, v in acc.items()}
    res["event_prob"] = 1.0 / (1.0 + np.exp(-res["event_logit"]))
    return res


def baseline_naive(ws: WindowSet) -> np.ndarray:
    return np.repeat(ws.y_hist[:, -1:], C.HORIZON, axis=1)


def baseline_arima(ws: WindowSet, order: tuple[int, int, int] = (1, 1, 1)) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    out = np.zeros((len(ws), C.HORIZON), dtype=np.float64)
    fails = 0
    for i, hist in enumerate(ws.y_hist):
        try:
            model = ARIMA(hist.astype(np.float64), order=order,
                          enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(method_kwargs={"warn_convergence": False})
            out[i] = fit.forecast(C.HORIZON)
        except Exception:
            fails += 1
            out[i] = hist[-1]
    if fails:
        log.info("ARIMA: %d of %d windows failed to converge, fell back to last value",
                 fails, len(ws))
    return out


MLR_RIDGE = 5.0


def _mlr_fit_region(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    days, n_lin = counts.shape
    t = np.arange(days, dtype=np.float64)

    def negll(theta: np.ndarray) -> float:
        a = np.concatenate([[0.0], theta[: n_lin - 1]])
        s = np.concatenate([[0.0], theta[n_lin - 1 :]])
        z = a[None, :] + s[None, :] * t[:, None]
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        ridge = MLR_RIDGE * float(np.sum(theta[n_lin - 1 :] ** 2)) * days**2
        return -float((counts * logp).sum()) + ridge

    theta0 = np.zeros(2 * (n_lin - 1))
    res = optimize.minimize(negll, theta0, method="L-BFGS-B",
                            options={"maxiter": 120, "maxfun": 240})
    a = np.concatenate([[0.0], res.x[: n_lin - 1]])
    s = np.concatenate([[0.0], res.x[n_lin - 1 :]])
    return a, s


def baseline_mlr(ws: WindowSet, panel) -> np.ndarray:
    out = np.zeros((len(ws), C.HORIZON), dtype=np.float64)
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for i in range(len(ws)):
        ri, li, te = int(ws.region[i]), int(ws.lineage[i]), int(ws.t_end[i])
        key = (ri, te)
        if key not in cache:
            block = panel.counts[ri, :, te - C.T_IN + 1 : te + 1].T
            cache[key] = _mlr_fit_region(block)
        a, s = cache[key]
        future = np.arange(C.T_IN, C.T_IN + C.HORIZON, dtype=np.float64)
        z_all = a[None, :] + s[None, :] * future[:, None]
        z_all -= z_all.max(axis=1, keepdims=True)
        p = np.exp(z_all)
        p /= p.sum(axis=1, keepdims=True)
        pi = np.clip(p[:, li], 1e-6, 1 - 1e-6)
        z_spec = C.FEATURES[C.FEATURE_INDEX["z"]]
        out[i] = np.clip(np.log(pi / (1 - pi)), z_spec.lo, z_spec.hi)
    return out


def baseline_single_lstm(
    train: WindowSet, test: WindowSet, seed: int, epochs: int = 20) -> np.ndarray:
    torch.manual_seed(seed)
    net = SingleLSTM()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=C.WEIGHT_DECAY)
    xt = torch.from_numpy(train.x)
    yt = torch.from_numpy(train.y)
    n = len(train)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, C.BATCH_SIZE):
            idx = perm[i : i + C.BATCH_SIZE]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP_NORM)
            opt.step()
    net.eval()
    with torch.no_grad():
        delta = np.concatenate([net(torch.from_numpy(test.x[i : i + 512])).numpy()
                                for i in range(0, len(test), 512)])
    return delta + test.anchor[:, None]


def sigma_from_residuals(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    resid = truth - pred
    s = resid.std(axis=0, keepdims=True)
    return np.repeat(np.clip(s, 1e-3, None), len(pred), axis=0)


def compare_models(preds: dict[str, dict], ws: WindowSet) -> dict:
    naive = baseline_naive(ws)
    metrics: dict[str, dict[str, float]] = {}
    losses: dict[str, np.ndarray] = {}
    leads: dict[str, list[float]] = {}

    for name, p in preds.items():
        mu, sigma = p["mu"], p["sigma"]
        score = p.get("event_prob")
        if score is None:
            dom = np.log(C.DOMINANCE_THRESHOLD / (1 - C.DOMINANCE_THRESHOLD))
            score = mu.max(axis=1) - dom
        lt = L.lead_time(ws.t_end, ws.region, ws.lineage, ws.y_abs, mu, ws.y_hist)
        metrics[name] = L.evaluate(ws.y_abs, mu, sigma, naive, ws.event, score, lt)
        losses[name] = np.abs(ws.y_abs - mu).mean(axis=1)
        leads[name] = lt.tolist()

    return {"metrics": metrics, "dm_losses": {k: v.tolist() for k, v in losses.items()},
            "lead_times_by_model": leads}


def integrated_gradients(net: torch.nn.Module, x: np.ndarray, steps: int = 48) -> np.ndarray:
    net.eval()
    xt = torch.from_numpy(x)
    base = torch.zeros_like(xt)
    total = torch.zeros_like(xt)
    for a in np.linspace(1.0 / steps, 1.0, steps):
        point = (base + a * (xt - base)).clone().requires_grad_(True)
        out = net(point)["mu"].mean()
        grad = torch.autograd.grad(out, point)[0]
        total += grad
    attributions = ((xt - base) * total / steps).detach().numpy()
    return attributions.sum(axis=(0, 1))


def correlation_analysis(ws: WindowSet, panel, rng: np.random.Generator,
                         max_n: int = 6000) -> dict:
    idx = rng.choice(len(ws), size=min(max_n, len(ws)), replace=False)
    x = ws.x[idx]

    names = [f.name_en for f in C.FEATURES] + [C.CONTROL_FEATURE.name_en]
    keys = list(C.FEATURE_KEYS) + [C.CONTROL_FEATURE.key]

    values = [x[:, -1, i] for i in range(C.N_FEATURES)]
    values.append(panel.control[ws.region[idx], ws.t_end[idx]])

    horizons = [h for h in (1, 7, 14, 21, C.HORIZON) if h <= C.HORIZON]
    r_mat = np.zeros((len(values), len(horizons)))
    p_mat = np.zeros_like(r_mat)
    for j, h in enumerate(horizons):
        target = ws.y[idx][:, h - 1]
        for i, v in enumerate(values):
            r, p = stats.pearsonr(v, target)
            r_mat[i, j], p_mat[i, j] = float(r), float(p)

    keep_mat = L.benjamini_hochberg(p_mat.ravel()).reshape(p_mat.shape)
    keep_any = keep_mat.any(axis=1)
    final = horizons.index(C.HORIZON)
    return {
        "keys": keys,
        "names": names,
        "horizons": horizons,
        "r_matrix": r_mat.tolist(),
        "p_matrix": p_mat.tolist(),
        "significant_matrix": keep_mat.tolist(),
        "r": r_mat[:, final].tolist(),
        "p": p_mat[:, final].tolist(),
        "significant": {k: bool(s) for k, s in zip(keys, keep_any, strict=True)},
        "n": int(len(idx)),
    }


def cusum_detector(z_series: np.ndarray, k: float | None = None,
                   h: float | None = None) -> dict:
    dz = np.gradient(z_series)
    warm = max(30, len(dz) // 5)
    ref = dz[:warm]
    k = float(np.median(ref) + 1.0 * ref.std()) if k is None else k
    h = float(4.0 * max(ref.std(), 1e-3)) if h is None else h

    s = np.zeros(len(dz))
    for t in range(1, len(dz)):
        s[t] = max(0.0, s[t - 1] + (dz[t] - k))
    fired = np.flatnonzero(s > h)
    return {
        "s": s, "k": k, "h": h,
        "t_alarm": int(fired[0]) if fired.size else None,
        "dz": dz,
    }


def geographic_forecast(panel, lineage: int, t_now: int) -> dict:
    from data import _gravity_matrix

    flow = _gravity_matrix(panel.coords, panel.population)
    present = panel.counts[:, lineage, max(0, t_now - 14) : t_now + 1].sum(axis=1) > 0
    share = panel.counts[:, lineage, t_now] / np.clip(panel.depth[:, t_now], 1.0, None)

    eta = np.zeros(len(panel.region_names))
    for r in range(len(eta)):
        if present[r]:
            eta[r] = 0.0
            continue
        best = np.inf
        for src in np.flatnonzero(present):
            best = min(best, 6.0 / (flow[src, r] + 0.02))
        eta[r] = best if np.isfinite(best) else 90.0

    edges = []
    iu = np.triu_indices(len(eta), k=1)
    strength = flow[iu]
    for j in np.argsort(-strength)[:20]:
        edges.append((int(iu[0][j]), int(iu[1][j]), float(strength[j])))

    return {
        "eta": eta, "share": share, "present": present, "edges": edges,
        "coords": panel.coords, "names": panel.region_names,
        "lineage": panel.lineage_names[lineage], "t_now": t_now,
    }
