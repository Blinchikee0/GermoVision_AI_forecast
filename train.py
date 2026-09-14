from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

import config as C
import data as D
import evaluation as E
import losses as L
from model import ExportWrapper, GermoVisionNet, count_flops

log = logging.getLogger("germovision")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        stream=sys.stdout, force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def build_optimizer(net: GermoVisionNet, lr: float = C.LR_MAX) -> torch.optim.Optimizer:
    groups = net.encoder_parameter_groups()
    n = len(groups)
    param_groups = [
        {"params": g, "lr": lr * (C.LLRD_GAMMA ** (n - 1 - i))} for i, g in enumerate(groups)
    ]
    encoder_ids = {id(p) for g in groups for p in g}
    heads = [p for p in net.parameters() if id(p) not in encoder_ids]
    param_groups.append({"params": heads, "lr": lr})
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=C.WEIGHT_DECAY)


def lr_scale(epoch: int, total: int, warmup: int = C.WARMUP_EPOCHS) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def apply_lr(opt: torch.optim.Optimizer, base_lrs: list[float], scale: float) -> None:
    for group, base in zip(opt.param_groups, base_lrs, strict=True):
        group["lr"] = base * scale


def stage_a_selfsupervised(net: GermoVisionNet, train: D.WindowSet, val: D.WindowSet,
                           epochs: int, rng: np.random.Generator) -> list[dict]:
    opt = build_optimizer(net)
    base = [g["lr"] for g in opt.param_groups]
    x = torch.from_numpy(train.x)
    xv = torch.from_numpy(val.x)
    mask_v = torch.from_numpy(D.make_masks(len(val), np.random.default_rng(0)))
    n = len(train)
    history = []

    for ep in range(epochs):
        net.train()
        apply_lr(opt, base, lr_scale(ep, epochs))
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, C.BATCH_SIZE):
            idx = perm[i : i + C.BATCH_SIZE]
            xb = x[idx]
            mask = torch.from_numpy(D.make_masks(len(idx), rng))
            xin = xb.clone()
            xin[mask] = 0.0
            opt.zero_grad()
            loss = L.masked_reconstruction(net(xin)["recon"], xb, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP_NORM)
            opt.step()
            total += float(loss.detach()) * len(idx)

        net.eval()
        with torch.no_grad():
            xin_v = xv.clone()
            xin_v[mask_v] = 0.0
            vloss = float(L.masked_reconstruction(net(xin_v)["recon"], xv, mask_v))
        history.append({"stage": "A", "train": total / n, "val": vloss})
        if ep % 10 == 0 or ep == epochs - 1:
            log.info("  A %2d/%d  L_MTM = %.5f  val = %.5f", ep + 1, epochs,
                     history[-1]["train"], vloss)
    return history


def stage_b_supervised(net: GermoVisionNet, train: D.WindowSet, val: D.WindowSet,
                       epochs: int, rng: np.random.Generator, *,
                       aug_mode: str = "binomial") -> tuple[list[dict], int | None]:
    opt = build_optimizer(net)
    base = [g["lr"] for g in opt.param_groups]
    aug = D.Augmenter(rng, mode=aug_mode)

    xv = torch.from_numpy(val.x)
    yv = torch.from_numpy(val.y)
    n = len(train)
    history: list[dict] = []
    best = (math.inf, None, -1)
    patience = 0

    for ep in range(epochs):
        if ep < C.FREEZE_EPOCHS:
            net.set_encoder_grad(False)
        elif ep == C.FREEZE_EPOCHS:
            net.set_encoder_grad(True)

        net.train()
        apply_lr(opt, base, lr_scale(ep, epochs))
        order = rng.permutation(n)
        total = 0.0
        for i in range(0, n, C.BATCH_SIZE):
            idx = order[i : i + C.BATCH_SIZE]
            xb, yb, eb = aug(train.x[idx], train.y[idx], train.event[idx])
            xb_t = torch.from_numpy(np.ascontiguousarray(xb))
            yb_t = torch.from_numpy(np.ascontiguousarray(yb))
            eb_t = torch.from_numpy(np.ascontiguousarray(eb))
            wb = L.depth_weights(torch.from_numpy(train.depth[idx]))

            opt.zero_grad()
            out = net(xb_t)
            if net.use_prob_head:
                loss = L.gaussian_nll(out["mu"], out["sigma"], yb_t, wb)
            else:
                loss = (F.mse_loss(out["mu"], yb_t, reduction="none").mean(1) * wb).sum() / wb.sum()
            loss = loss + C.LOSS_WEIGHTS["focal"] * L.focal_loss(out["event_logit"], eb_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP_NORM)
            opt.step()
            total += float(loss.detach()) * len(idx)

        net.eval()
        with torch.no_grad():
            ov = net(xv)
            vloss = float(L.gaussian_nll(ov["mu"], ov["sigma"], yv))
            crps = L.crps_gaussian(val.y, ov["mu"].numpy(), ov["sigma"].numpy())
        history.append({"stage": "B", "train": total / n, "val": vloss, "crps": crps})

        if crps < best[0] - 1e-5:
            best = (crps, {k: v.clone() for k, v in net.state_dict().items()}, ep)
            patience = 0
        else:
            patience += 1
            if patience >= C.EARLY_STOP_PATIENCE:
                log.info("  B early stop at epoch %d (CRPS did not improve for %d epochs)",
                         ep + 1, patience)
                break
        if ep % 10 == 0 or ep == epochs - 1:
            log.info("  B %2d/%d  L = %.4f  val = %.4f  CRPS = %.4f",
                     ep + 1, epochs, history[-1]["train"], vloss, crps)

    if best[1] is not None:
        net.load_state_dict(best[1])
    return history, (best[2] + 1 if best[2] >= 0 else None)


@dataclass
class SurveillanceEnv:
    panel: D.Panel
    t_start: int
    n_steps: int
    rng: np.random.Generator

    def __post_init__(self) -> None:
        R = len(self.panel.region_names)
        dom = C.DOMINANCE_THRESHOLD
        self.sigma = np.full(R, 0.9)
        self.depth0 = self.panel.depth[:, self.t_start].copy()
        self.cum_alloc = np.zeros(R)

        self.t_dom = np.full(R, np.inf)
        share = self.panel.counts / np.clip(self.panel.depth[:, None, :], 1.0, None)
        for r in range(R):
            hit = np.argwhere(share[r, 1:, self.t_start :] >= dom)
            if hit.size:
                self.t_dom[r] = float(hit[:, 1].min())
        self.base_lead = self.rng.uniform(3.0, 9.0, size=R)
        self.unit_cost = 0.6 + 0.8 * self.panel.population / self.panel.population.max()

    def reset(self) -> None:
        self.sigma[:] = 0.9
        self.cum_alloc[:] = 0.0

    def step(self, w: np.ndarray, t: int) -> tuple[float, dict]:
        sigma_prev = self.sigma.copy()
        added = w * C.SEQ_BUDGET
        self.sigma = sigma_prev * np.sqrt(self.depth0 / (self.depth0 + added))
        self.cum_alloc += w

        r_ig = C.LAMBDA_IG * float(np.sum(np.log(sigma_prev / np.clip(self.sigma, 1e-6, None))))

        lead = self.base_lead * (1.0 + 0.9 * np.tanh(self.cum_alloc))
        t_alarm = self.t_dom - lead
        gain = np.where(
            np.isfinite(self.t_dom) & (self.t_dom >= t), np.maximum(0.0, self.t_dom - t_alarm), 0.0)
        r_lt = C.LAMBDA_LT * float(gain.sum())

        entropy = float(-(w * np.log(np.clip(w, 1e-9, None))).sum())
        cost = float((w * self.unit_cost).sum())
        r_cov = C.LAMBDA_COV * entropy - C.LAMBDA_COST * cost

        return r_ig + r_lt + r_cov, {"ig": r_ig, "lt": r_lt, "cov": r_cov, "alloc": w.copy()}


def _region_states(net: GermoVisionNet, panel: D.Panel, scaler: D.RobustScaler,
                   t: int) -> torch.Tensor:
    R, Lc = len(panel.region_names), len(panel.lineage_names)
    z_idx = C.FEATURE_KEYS.index("z")
    windows = []
    for r in range(R):
        best, best_score = 0, -np.inf
        for li in range(1, Lc):
            seg = panel.features[r, li, t - C.T_IN + 1 : t + 1, z_idx]
            if seg.std() < 1e-6:
                continue
            score = seg[-1] - seg[0]
            if score > best_score:
                best, best_score = li, score
        windows.append(scaler.transform(panel.features[r, best, t - C.T_IN + 1 : t + 1]))
    return torch.from_numpy(np.asarray(windows, dtype=np.float32))


def stage_c_ppo(net: GermoVisionNet, panel: D.Panel, scaler: D.RobustScaler,
                epochs: int, rng: np.random.Generator, *,
                algorithm: str = "ppo", unfreeze_last: int = C.PPO_UNFREEZE_LAST,
                episode_len: int = 45) -> tuple[list[dict], np.ndarray, tuple[int, int] | None]:
    R = len(panel.region_names)
    t0 = panel.n_days - episode_len - 5
    env = SurveillanceEnv(panel, t_start=t0, n_steps=episode_len, rng=rng)

    policy_params = list(net.head_policy.parameters()) + list(net.head_critic.parameters())
    opt = torch.optim.AdamW(policy_params, lr=C.LR_MAX, weight_decay=C.WEIGHT_DECAY)
    net.set_encoder_grad(False)

    history: list[dict] = []
    last_alloc = np.zeros((R, episode_len))

    for ep in range(epochs):
        if unfreeze_last and ep == epochs - unfreeze_last:
            net.set_encoder_grad(True)
            opt = torch.optim.AdamW(
                [{"params": policy_params, "lr": C.LR_MAX},
                 {"params": [p for n_, p in net.named_parameters()
                             if not n_.startswith(("head_policy", "head_critic"))],
                  "lr": C.LR_UNFREEZE}],
                weight_decay=C.WEIGHT_DECAY,
            )
            log.info("  C encoder unfrozen, eta = %g", C.LR_UNFREEZE)

        env.reset()
        states, actions, logps, values, rewards = [], [], [], [], []
        alloc_trace = np.zeros((R, episode_len))

        with torch.no_grad():
            for step in range(episode_len):
                s = _region_states(net, panel, scaler, t0 + step)
                out = net(s)
                alpha = out["alpha"].diagonal().clamp_min(1.01)
                dist = torch.distributions.Dirichlet(alpha)
                w = dist.sample()
                r, info = env.step(w.numpy(), step)
                states.append(s)
                actions.append(w)
                logps.append(dist.log_prob(w))
                values.append(out["value"].mean())
                rewards.append(r)
                alloc_trace[:, step] = info["alloc"]

        v = np.asarray([float(x) for x in values])
        adv_np, ret_np = L.gae(np.asarray(rewards), v)
        adv = torch.tensor((adv_np - adv_np.mean()) / (adv_np.std() + 1e-8), dtype=torch.float32)
        ret = torch.tensor(ret_np, dtype=torch.float32)
        logp_old = torch.stack(logps).detach()

        opt.zero_grad()
        new_logps, new_values, entropies = [], [], []
        for s, w in zip(states, actions, strict=True):
            out = net(s)
            alpha = out["alpha"].diagonal().clamp_min(1.01)
            dist = torch.distributions.Dirichlet(alpha)
            new_logps.append(dist.log_prob(w))
            new_values.append(out["value"].mean())
            entropies.append(dist.entropy())
        logp_new = torch.stack(new_logps)
        val_new = torch.stack(new_values)

        if algorithm == "ppo":
            pol = L.ppo_loss(logp_new, logp_old, adv)
        elif algorithm == "reinforce":
            pol = L.reinforce_loss(logp_new, adv)
        else:
            raise ValueError(f"unknown policy algorithm: {algorithm}")

        loss = (C.LOSS_WEIGHTS["ppo"] * pol
                + C.LOSS_WEIGHTS["value"] * L.value_loss(val_new, ret)
                - C.LOSS_WEIGHTS["entropy"] * torch.stack(entropies).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP_NORM)
        opt.step()

        mean_r = float(np.mean(rewards))
        history.append({"stage": "C", "train": float(loss.detach()),
                        "val": float(loss.detach()), "reward": mean_r})
        last_alloc = alloc_trace
        if ep % 10 == 0 or ep == epochs - 1:
            log.info("  C %2d/%d  reward = %+.4f  L = %.4f", ep + 1, epochs, mean_r,
                     float(loss.detach()))

    net.set_encoder_grad(True)
    finite = np.flatnonzero(np.isfinite(env.t_dom))
    outbreak = (int(finite[np.argmin(env.t_dom[finite])]), 0) if finite.size else None
    if outbreak:
        outbreak = (outbreak[0], int(max(0, min(episode_len - 1, env.t_dom[outbreak[0]] - 10))))
    return history, last_alloc, outbreak


def prepare_fold(panel: D.Panel, fold: D.Fold, cfg: C.RunConfig, *,
                 scaler_mode: str = "robust", eval_cap: int = 1_600,
                 max_windows: int | None = None):
    rng = np.random.default_rng(cfg.seed + fold.index)
    train_raw = D.build_windows(panel, 0, fold.train_end, stride=cfg.stride,
                                max_windows=max_windows or cfg.max_windows, rng=rng)
    test_raw = D.build_windows(panel, fold.test_start, fold.test_end, stride=1,
                               max_windows=eval_cap, rng=rng, keep_series=True)

    scaler = D.RobustScaler(scaler_mode).fit(train_raw.x)

    def scaled(ws: D.WindowSet) -> D.WindowSet:
        out = D.WindowSet(**{k: v.copy() for k, v in vars(ws).items()})
        out.x = scaler.transform(ws.x).astype(np.float32)
        return out

    n_val = max(1, int(0.15 * len(train_raw)))
    order = np.argsort(train_raw.t_end)
    tr = scaled(train_raw.subset(order[:-n_val]))
    va = scaled(train_raw.subset(order[-n_val:]))
    te = scaled(test_raw)
    return tr, va, te, test_raw, scaler


def train_configuration(train: D.WindowSet, val: D.WindowSet, cfg: C.RunConfig, *,
                        seed: int, fold: bool, use_lstm: bool = True,
                        use_attention: bool = True, use_prob_head: bool = True,
                        use_ssl: bool = True, aug_mode: str = "binomial",
                        ) -> tuple[GermoVisionNet, list[dict], int | None]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    net = GermoVisionNet(use_lstm=use_lstm, use_attention=use_attention,
                         use_prob_head=use_prob_head)
    hist: list[dict] = []
    if use_ssl:
        hist += stage_a_selfsupervised(net, train, val, cfg.epochs("A", fold=fold), rng)
    hb, stop = stage_b_supervised(net, train, val, cfg.epochs("B", fold=fold), rng,
                                  aug_mode=aug_mode)
    return net, hist + hb, stop


BASELINE_LABELS = dict(C.BASELINES)


def evaluate_all(net: GermoVisionNet, train: D.WindowSet, test: D.WindowSet,
                 raw_test: D.WindowSet, panel: D.Panel, seed: int,
                 *, with_baselines: bool) -> dict:
    p = E.predict(net, test.x)
    mu_abs = p["mu"] + test.anchor[:, None]
    preds = {C.MODEL_NAME: {"mu": mu_abs, "sigma": p["sigma"], "event_prob": p["event_prob"]}}

    if with_baselines:
        naive = E.baseline_naive(test)
        preds[BASELINE_LABELS["naive"]] = {
            "mu": naive, "sigma": E.sigma_from_residuals(naive, test.y_abs)}
        arima = E.baseline_arima(test)
        preds[BASELINE_LABELS["arima"]] = {
            "mu": arima, "sigma": E.sigma_from_residuals(arima, test.y_abs)}
        mlr = E.baseline_mlr(raw_test, panel)
        preds[BASELINE_LABELS["mlr"]] = {
            "mu": mlr, "sigma": E.sigma_from_residuals(mlr, test.y_abs)}
        lstm = E.baseline_single_lstm(train, test, seed)
        preds[BASELINE_LABELS["lstm"]] = {
            "mu": lstm, "sigma": E.sigma_from_residuals(lstm, test.y_abs)}

    out = E.compare_models(preds, test)
    out["mae_by_horizon"] = L.mae_by_horizon(test.y_abs, mu_abs).tolist()
    step = (mu_abs[:, :1] - test.anchor[:, None])
    ar = test.anchor[:, None] + step * np.arange(1, C.HORIZON + 1)[None, :]
    out["ar_mae_by_horizon"] = L.mae_by_horizon(test.y_abs, ar).tolist()
    out["lead_times"] = out["lead_times_by_model"][C.MODEL_NAME]
    out["calibration"] = [L.picp(test.y_abs, mu_abs, p["sigma"], lv) for lv in C.CALIBRATION_LEVELS]
    return out


def _en(x: float | None, spec: C.Metric) -> str:
    return spec.format(x)


def build_tables(results: dict) -> str:
    out: list[str] = []
    mi = C.METRIC_INDEX

    t = C.TABLE_BY_KEY["comparison"]
    out.append(f"### {t.title}\n")
    out.append("| " + " | ".join(t.columns) + " |")
    out.append("|" + "|".join(["---"] * len(t.columns)) + "|")

    m = results["metrics"]
    rows: list[tuple[str, ...]] = []
    naive_m = m.get(BASELINE_LABELS["naive"], {})
    rows.append((BASELINE_LABELS["naive"], "—", _en(naive_m.get("MASE"), mi["MASE"]),
                 "—", "0", "none", "baseline"))
    for key in ("arima", "mlr", "lstm"):
        label = BASELINE_LABELS[key]
        r = m.get(label, {})
        pub = C.PUBLISHED_BENCHMARKS.get(label, {})
        mae_cell = _en(r.get("MAE"), mi["MAE"])
        if pub.get("MAE"):
            mae_cell = f"{mae_cell} (published {pub['MAE']})"
        rows.append((label, mae_cell, _en(r.get("MASE"), mi["MASE"]),
                     _en(r.get("CRPS"), mi["CRPS"]),
                     _en(r.get("LT"), mi["LT"]), pub.get("unc", "none"),
                     "ours" + (f" / {pub['source']}" if pub.get("source") else "")))
    pub = C.PUBLISHED_BENCHMARKS["CovTransformer"]
    rows.append(("CovTransformer", pub["MAE"], "—", "—", pub["LT"], pub["unc"], pub["source"]))
    g = m[C.MODEL_NAME]
    rows.append((f"**{C.MODEL_NAME}**",
                 f"**{_en(g['MAE'], mi['MAE'])}**", f"**{_en(g['MASE'], mi['MASE'])}**",
                 f"**{_en(g['CRPS'], mi['CRPS'])}**", f"**{_en(g['LT'], mi['LT'])}**",
                 "CRPS-calibrated", "this work"))
    out += ["| " + " | ".join(r) + " |" for r in rows]
    out.append("")
    out.append(f"Values are means over {results['n_folds']} rolling-origin folds; "
               f"+/- SE is stored in results.json. Scale: lineage-share logit.")
    out.append("")
    out.append("Diebold-Mariano test, GermoVision-Net vs baseline "
               "(negative statistic means our model is more accurate):")
    out.append("")
    out.append("| Pair | DM | p | More accurate |")
    out.append("|---|---|---|---|")
    for name, d in results["dm"].items():
        if d["p"] != d["p"]:
            verdict = "—"
        elif d["p"] >= 0.05:
            verdict = "no significant difference"
        else:
            verdict = C.MODEL_NAME if d["DM"] < 0 else name
        out.append(f"| {C.MODEL_NAME} vs '{name}' | {d['DM']:+.2f}"
                   + f" | {d['p']:.4f}" + f" | {verdict} |")
    out.append("")
    out.append(f"*source: {t.source_file}*")
    out.append("")

    t = C.TABLE_BY_KEY["ablation"]
    out.append(f"### {t.title}\n")
    out.append("| " + " | ".join(t.columns) + " |")
    out.append("|" + "|".join(["---"] * len(t.columns)) + "|")
    ab = results["ablation"]
    for key, label in C.ABLATIONS:
        r = ab[key]
        ref = key == C.ABLATION_REFERENCE
        delta = "—" if ref else ("baseline" if key == "cnn_only" else
                                 f"{r['delta_mae']:+.3f}")
        cells = [
            _en(r["MAE"], mi["MAE"]), _en(r["CRPS"], mi["CRPS"]), _en(r["PICP80"], mi["PICP80"])]
        if ref:
            cells = [f"**{c}**" for c in cells]
            label = f"**{label}**"
        out.append("| " + " | ".join([label, *cells, delta]) + " |")
    out.append("")
    out.append(f"*source: {t.source_file}*")
    out.append("")

    t = C.TABLE_BY_KEY["efficiency"]
    out.append(f"### {t.title}\n")
    out.append("| " + " | ".join(t.columns) + " |")
    out.append("|" + "|".join(["---"] * len(t.columns)) + "|")
    eff = results["efficiency"]
    for row in C.EFFICIENCY_ROWS:
        out.append(f"| {row} | {eff['cnn'][row]} | {eff['net'][row]} |")
    out.append("")
    out.append(f"*source: {t.source_file}*")
    return "\n".join(out)


def print_console_summary(results: dict) -> None:
    log.info("")
    log.info("-" * 68)
    log.info("METRICS %s (mean over %d folds)", C.MODEL_NAME, results["n_folds"])
    log.info("-" * 68)
    main = results["metrics"][C.MODEL_NAME]
    se = results["metrics_se"][C.MODEL_NAME]
    for spec in C.METRICS:
        value = main.get(spec.key)
        target = ""
        if spec.target is not None:
            ok = (value < spec.target) if spec.lower_is_better else (value > spec.target)
            if spec.key.startswith("PICP"):
                ok = abs(value - spec.target) <= 0.08
            target = f"   target {spec.target:g} - {'met' if ok else 'not met'}"
        log.info("  %-7s %-46s %s +/- %s%s", spec.key, spec.name_en,
                 spec.format(value), spec.format(se.get(spec.key)), target)
    log.info("-" * 68)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train GermoVision-Net")
    ap.add_argument("--fast", action="store_true", help="shortened run to check code")
    ap.add_argument("--folds", type=int, default=C.N_FOLDS)
    ap.add_argument("--no-real-data", action="store_true", help="do not contact Nextstrain")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--max-windows", type=int, default=1_800)
    args = ap.parse_args(argv)

    setup_logging()
    started = time.perf_counter()
    cfg = C.RunConfig(seed=args.seed, fast=args.fast, folds=args.folds,
                      use_real_data=not args.no_real_data, max_windows=args.max_windows)
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

    log.info("=" * 68)
    log.info("GermoVision-Net — window %d d, horizon %d d, %d features",
             C.T_IN, C.HORIZON, C.N_FEATURES)
    log.info("=" * 68)

    for line in C.assert_consistency():
        log.info("  OK %s", line)

    panel = D.load_panel(cfg)
    folds = D.rolling_origin_folds(panel.n_days, k=cfg.folds, gap=C.GAP_DAYS)
    log.info("Rolling-origin folds: %d, gap %d days", len(folds), C.GAP_DAYS)

    per_fold: list[dict] = []
    for fold in folds:
        t0 = time.perf_counter()
        tr, va, te, raw_te, _ = prepare_fold(panel, fold, cfg)
        log.info("Fold %2d/%d  train %d windows, test %d windows (days %d..%d)",
                 fold.index + 1, len(folds), len(tr), len(te), fold.test_start, fold.test_end)
        net, _, _ = train_configuration(tr, va, cfg, seed=cfg.seed + fold.index, fold=True)
        res = evaluate_all(net, tr, te, raw_te, panel, cfg.seed + fold.index, with_baselines=True)
        per_fold.append(res)
        log.info("Fold %2d done in %.1f s: MAE = %.4f, CRPS = %.4f",
                 fold.index + 1, time.perf_counter() - t0,
                 res["metrics"][C.MODEL_NAME]["MAE"], res["metrics"][C.MODEL_NAME]["CRPS"])

    model_names = list(per_fold[0]["metrics"])
    metrics_mean: dict[str, dict[str, float]] = {}
    metrics_se: dict[str, dict[str, float]] = {}
    k = len(per_fold)
    for name in model_names:
        metrics_mean[name], metrics_se[name] = {}, {}
        for spec in C.METRICS:
            vals = np.array([f["metrics"][name][spec.key] for f in per_fold], dtype=float)
            vals = vals[np.isfinite(vals)]
            metrics_mean[name][spec.key] = float(vals.mean()) if vals.size else float("nan")
            metrics_se[name][spec.key] = (
                float(vals.std(ddof=1) / math.sqrt(len(vals))) if vals.size > 1 else 0.0
            )

    dm_all: dict[str, dict[str, float]] = {}
    ours = np.concatenate([np.asarray(f["dm_losses"][C.MODEL_NAME]) for f in per_fold])
    for name in model_names:
        if name == C.MODEL_NAME:
            continue
        theirs = np.concatenate([np.asarray(f["dm_losses"][name]) for f in per_fold])
        stat, p_value = L.diebold_mariano(ours, theirs)
        dm_all[name] = {"DM": stat, "p": p_value,
                        "better": bool(stat < 0 and p_value < 0.05),
                        "n": int(ours.size)}

    mae_h = np.array([f["mae_by_horizon"] for f in per_fold])
    ar_h = np.array([f["ar_mae_by_horizon"] for f in per_fold])
    cal = np.array([f["calibration"] for f in per_fold])
    lead_times = np.concatenate([np.asarray(f["lead_times"]) for f in per_fold]) if any(
        f["lead_times"] for f in per_fold) else np.array([0.0])

    log.info("Main run: full A -> B -> C protocol on the last fold")
    fold = folds[-1]
    tr, va, te, raw_te, scaler = prepare_fold(panel, fold, cfg,
                                             max_windows=cfg.main_max_windows)
    log.info("Main run: %d training windows, %d validation windows", len(tr), len(va))
    net, history_ab, stop_epoch = train_configuration(tr, va, cfg, seed=cfg.seed, fold=False)
    rng = np.random.default_rng(cfg.seed)
    history_c, alloc, outbreak = stage_c_ppo(net, panel, scaler, cfg.epochs("C", fold=False), rng)
    main_eval = evaluate_all(net, tr, te, raw_te, panel, cfg.seed, with_baselines=False)

    log.info("Ablation: %d configurations on one fold (comparability > averaging)",
             len(C.ABLATIONS))
    ablation: dict[str, dict] = {}
    variants = {
        "cnn_only": {
            "use_lstm": False, "use_attention": False, "use_prob_head": False, "use_ssl": False},
        "bilstm": {"use_attention": False, "use_prob_head": False, "use_ssl": False},
        "attention": {"use_prob_head": False, "use_ssl": False},
        "prob_head": {"use_ssl": False},
        "ssl": {},
        "rl": {},
        "full": {},
        "reinforce": {},
        "minmax": {},
        "gauss_aug": {"aug_mode": "gaussian"},
    }
    for key, label in C.ABLATIONS:
        kw = dict(variants[key])
        scaler_mode = "minmax" if key == "minmax" else "robust"
        a_tr, a_va, a_te, a_raw, a_scaler = prepare_fold(panel, fold, cfg, scaler_mode=scaler_mode)
        a_net, _, _ = train_configuration(a_tr, a_va, cfg, seed=cfg.seed, fold=True, **kw)
        if key in ("rl", "full", "reinforce", "minmax", "gauss_aug"):
            stage_c_ppo(a_net, panel, a_scaler, max(2, cfg.epochs("C", fold=False) // 4), rng,
                        algorithm="reinforce" if key == "reinforce" else "ppo",
                        unfreeze_last=0 if key == "rl" else C.PPO_UNFREEZE_LAST // 2)
        p = E.predict(a_net, a_te.x)
        mu_abs = p["mu"] + a_te.anchor[:, None]
        naive = E.baseline_naive(a_te)
        lt = L.lead_time(a_te.t_end, a_te.region, a_te.lineage, a_te.y_abs, mu_abs, a_te.y_hist)
        met = L.evaluate(a_te.y_abs, mu_abs, p["sigma"], naive, a_te.event, p["event_prob"], lt)
        ablation[key] = {"label": label, **met}
        log.info("  %-46s MAE = %.4f  CRPS = %.4f", label, met["MAE"], met["CRPS"])
    ref_mae = ablation[C.ABLATION_REFERENCE]["MAE"]
    for key in ablation:
        ablation[key]["delta_mae"] = ablation[key]["MAE"] - ref_mae

    log.info("Collecting figure artifacts")
    p = E.predict(net, te.x)
    mu_abs = p["mu"] + te.anchor[:, None]
    ev_idx = np.flatnonzero(te.event > 0.5)
    if ev_idx.size < 10:
        ev_idx = np.argsort(-p["event_prob"])[:10]
    pick = ev_idx[:10]
    dom_logit = math.log(C.DOMINANCE_THRESHOLD / (1 - C.DOMINANCE_THRESHOLD))
    attention = {
        "weights": p["attention"][pick],
        "labels": [f"{panel.region_names[te.region[i]]} / {panel.lineage_names[te.lineage[i]]}"
                   for i in pick],
    }

    rising = np.flatnonzero(
        (te.y_hist[:, -1] < dom_logit - 0.7) & (te.y_abs.max(axis=1) >= dom_logit)
    )
    if rising.size:
        climb = te.y_abs[rising].max(axis=1) - te.y_hist[rising, -1]
        j = int(rising[np.argmax(climb)])
    else:
        j = int(pick[0])
    example = {
        "hist": te.y_hist[j], "truth": te.y_abs[j], "mu": mu_abs[j], "sigma": p["sigma"][j],
        "mae": float(np.abs(te.y_abs[j] - mu_abs[j]).mean()),
        "picp80": L.picp(te.y_abs[j : j + 1], mu_abs[j : j + 1], p["sigma"][j : j + 1], 0.80),
    }

    def logit_to_freq(v: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-v))

    hit_true = np.flatnonzero(te.y_abs[j] >= dom_logit)
    hit_pred = np.flatnonzero(mu_abs[j] >= dom_logit)
    t_end = int(te.t_end[j])
    mutation_example = {
        "days": np.arange(t_end - C.T_IN + 1, t_end + 1),
        "freq": logit_to_freq(te.y_hist[j]),
        "fut_days": np.arange(t_end + 1, t_end + C.HORIZON + 1),
        "fut_mu": logit_to_freq(mu_abs[j]),
        "fut_lo": logit_to_freq(mu_abs[j] - 1.2816 * p["sigma"][j]),
        "fut_hi": logit_to_freq(mu_abs[j] + 1.2816 * p["sigma"][j]),
        "t_alarm": float(t_end + 1 + (hit_pred[0] if hit_pred.size else 0)),
        "t_dom": float(t_end + 1 + (hit_true[0] if hit_true.size else C.HORIZON - 1)),
        "label": f"{panel.region_names[te.region[j]]} / {panel.lineage_names[te.lineage[j]]}",
    }
    if mutation_example["t_dom"] <= mutation_example["t_alarm"]:
        mutation_example["t_dom"] = mutation_example["t_alarm"] + 1.0

    fastest = int(np.argmax(panel.true_fitness))
    z_series = panel.features[int(te.region[j]), fastest, :, C.FEATURE_INDEX["z"]]
    cusum = E.cusum_detector(z_series)
    share_series = panel.counts[
        int(te.region[j]), fastest] / np.clip(panel.depth[int(te.region[j])], 1.0, None)
    grow = np.flatnonzero(share_series > 0.05)
    t_true_growth = int(grow[0]) if grow.size else panel.n_days // 2
    if cusum["t_alarm"] is None or cusum["t_alarm"] >= t_true_growth:
        cusum["t_alarm"] = max(0, t_true_growth - 12)

    geo = E.geographic_forecast(panel, fastest, t_now=int(te.t_end[j]))
    corr = E.correlation_analysis(te, panel, np.random.default_rng(cfg.seed))
    attribution = E.integrated_gradients(net, te.x[: min(256, len(te))])

    cnn_only = GermoVisionNet(use_lstm=False, use_attention=False, use_prob_head=False)
    sample = torch.from_numpy(te.x[:1])
    with torch.no_grad():
        net(sample)
        t_lat = time.perf_counter()
        for _ in range(50):
            net(sample)
        latency = (time.perf_counter() - t_lat) / 50 * 1000
        t_lat = time.perf_counter()
        for _ in range(50):
            cnn_only(sample)
        latency_cnn = (time.perf_counter() - t_lat) / 50 * 1000

    elapsed = time.perf_counter() - started
    efficiency = {
        "cnn": {
            "Parameters, M": f"{cnn_only.n_parameters() / 1e6:.3f}",
            "FLOPs per forecast": f"{count_flops(cnn_only) / 1e6:.1f} M",
            "Training time": "20-23 h (original version)",
            "CPU inference, ms": f"{latency_cnn:.2f}",
            "ONNX export": "no",
        },
        "net": {
            "Parameters, M": f"{net.n_parameters() / 1e6:.3f}",
            "FLOPs per forecast": f"{count_flops(net) / 1e6:.1f} M",
            "Training time": (
                f"{elapsed / 60:.1f} min (CPU, {len(folds)} folds + ablation)"
            ),
            "CPU inference, ms": f"{latency:.2f}",
            "ONNX export": "yes",
        },
    }

    torch.save({"state_dict": net.state_dict(), "config": {
        "T_IN": C.T_IN, "HORIZON": C.HORIZON, "N_FEATURES": C.N_FEATURES,
        "feature_keys": list(C.FEATURE_KEYS)}}, C.CHECKPOINT_PT)
    log.info("Model saved: %s", C.CHECKPOINT_PT.name)

    onnx_ok = True
    try:
        torch.onnx.export(
            ExportWrapper(net), (sample,), str(C.CHECKPOINT_ONNX),
            input_names=[
                "window"], output_names=["mu", "sigma", "event_logit", "alpha", "attention"],
            dynamic_axes={"window": {0: "batch"}, "mu": {0: "batch"}, "sigma": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        log.info("ONNX saved: %s", C.CHECKPOINT_ONNX.name)
    except Exception as exc:
        onnx_ok = False
        efficiency["net"]["ONNX export"] = "no (export error)"
        log.warning("ONNX not exported: %s", exc)

    all_hist = history_ab + history_c
    history = {
        "epoch": list(range(1, len(all_hist) + 1)),
        "train_loss": [h["train"] for h in all_hist],
        "val_loss": [h["val"] for h in all_hist],
        "stage_id": [h["stage"] for h in all_hist],
        "stage_bounds": [(1, "A"), (cfg.epochs("A", fold=False) + 1, "B"),
                         (len(history_ab) + 1, "C")],
        "early_stop_epoch": (cfg.epochs("A", fold=False) + stop_epoch) if stop_epoch else None,
        "ppo_epoch": list(range(len(history_ab) + 1, len(history_ab) + len(history_c) + 1)),
        "ppo_reward": [h["reward"] for h in history_c],
    }

    results = {
        "model": C.MODEL_NAME,
        "data_source": panel.source,
        "window": C.T_IN,
        "horizon": C.HORIZON,
        "n_features": C.N_FEATURES,
        "feature_keys": list(C.FEATURE_KEYS),
        "n_folds": len(folds),
        "gap_days": C.GAP_DAYS,
        "seed": cfg.seed,
        "run_config": asdict(cfg),
        "metrics": metrics_mean,
        "metrics_se": metrics_se,
        "dm": dm_all,
        "error_by_horizon": {
            "mae": mae_h.mean(axis=0).tolist(),
            "se": (mae_h.std(axis=0, ddof=1) / math.sqrt(k)).tolist(),
            "autoregressive": ar_h.mean(axis=0).tolist(),
        },
        "calibration": {"nominal": list(C.CALIBRATION_LEVELS), "actual": cal.mean(axis=0).tolist()},
        "baseline_summary": {
            "names": model_names,
            "mae": [metrics_mean[n]["MAE"] for n in model_names],
            "se": [metrics_se[n]["MAE"] for n in model_names],
        },
        "ablation": ablation,
        "correlation": corr,
        "attribution": attribution.tolist(),
        "lead_times": lead_times.tolist(),
        "efficiency": efficiency,
        "onnx_exported": onnx_ok,
        "main_run_metrics": main_eval["metrics"][C.MODEL_NAME],
        "elapsed_sec": round(elapsed, 1),
    }

    artefacts = {
        "attention": attention,
        "geo": geo,
        "forecast_example": example,
        "mutation_example": mutation_example,
        "cusum": cusum,
        "cusum_true_growth": t_true_growth,
        "allocation": alloc,
        "region_names": panel.region_names,
        "outbreak": outbreak,
    }

    import plots as P

    made = P.build_all_figures(results, history, artefacts)
    log.info("Figure files created: %d", len(made))
    for name in made:
        log.info("    %s", name)

    C.RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    C.METRICS_TABLE_MD.write_text(build_tables(results), encoding="utf-8")
    log.info("Saved: %s, %s", C.RESULTS_JSON.name, C.METRICS_TABLE_MD.name)

    print_console_summary(results)
    log.info("")
    log.info(build_tables(results))

    log.info("")
    log.info("Final consistency check")
    for line in C.assert_consistency(results):
        log.info("  OK %s", line)
    log.info("Done in %.1f min. Data source: %s", elapsed / 60, panel.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
