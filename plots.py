from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config as C

log = logging.getLogger("germovision.plots")
C.apply_rcparams()


def _wrap(text: str, width: int = 95) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width))


def _finish(fig: plt.Figure, spec: C.FigureSpec, note: str | None = None) -> plt.Figure:
    y = -0.02
    if note:
        wrapped = _wrap(note)
        fig.text(0.5, y, wrapped, ha="center", va="top", fontsize=8.2, color=C.INK_SOFT)
        y -= 0.055 * (wrapped.count("\n") + 1)
    fig.text(0.5, y, spec.full_caption, ha="center", va="top", fontsize=9, color=C.INK_SOFT)
    C.FIGDIR.mkdir(parents=True, exist_ok=True)
    for path in (spec.path_png, spec.path_pdf):
        fig.savefig(path, facecolor=C.SURFACE, bbox_inches="tight")
    log.info("figure %2d -> %s", spec.number, spec.filename_png)
    return fig


def _rounded_barh(ax, y: float, width: float, height: float, color: str, alpha: float = 1.0):
    x0, x1 = ax.get_xlim()
    span = max(abs(x1 - x0), C.EPS)
    r = min(height * 0.42, abs(width) * 0.45)
    patch = FancyBboxPatch(
        (0, y - height / 2), width, height,
        boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_aspect=span / max(ax.get_ylim()[1] - ax.get_ylim()[0], C.EPS),
        linewidth=0, facecolor=color, alpha=alpha, zorder=3,
    )
    ax.add_patch(patch)
    return patch


def _slot(i: int) -> str:
    if i >= len(C.PALETTE):
        raise IndexError(
            f"slot {i} requested; palette has {len(C.PALETTE)}: collapse into 'Other' or split"
        )
    return C.PALETTE[i]


def _direct_label(ax, x, y, text, color, **kw):
    ax.annotate(text, (x, y), color=color, fontsize=8.5, fontweight="bold",
                va="center", **kw)


def fig01_architecture() -> plt.Figure:
    spec = C.figure(1)
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(14, 100)

    n = len(C.ARCH_BLOCKS)
    margin, gap = 2.0, 2.8
    box_w = (100 - 2 * margin - (n - 1) * gap) / n
    box_h, y_main = 15.0, 56.0
    xs = margin + np.arange(n) * (box_w + gap)

    for i, (label, _shape, fkey) in enumerate(C.ARCH_BLOCKS):
        x = xs[i]
        ax.add_patch(FancyBboxPatch(
            (x, y_main), box_w, box_h, boxstyle="round,pad=0.3,rounding_size=1.0",
            linewidth=1.2, edgecolor=_slot(0) if fkey else C.INK_SOFT,
            facecolor="#eef3fb" if fkey else "#f2f2f0", zorder=2))
        ax.text(x + box_w / 2, y_main + box_h * 0.63, label, ha="center", va="center",
                fontsize=7.4, fontweight="bold", color=C.INK, linespacing=1.35, zorder=3)
        if fkey:
            ax.text(x + box_w / 2, y_main + box_h * 0.17, C.FORMULAS[fkey], ha="center",
                    va="center", fontsize=7.0, color=_slot(0), zorder=3)
        if i:
            x0, x1 = xs[i - 1] + box_w, x
            ax.add_patch(FancyArrowPatch(
                (x0, y_main + box_h / 2), (x1, y_main + box_h / 2), arrowstyle="-|>",
                mutation_scale=10, linewidth=1.0, color=C.INK_SOFT, zorder=1))
            ax.text((x0 + x1) / 2, y_main - 1.6, C.ARCH_BLOCKS[i - 1][1], ha="center",
                    va="top", fontsize=6.6, color=C.INK_SOFT)


    nh = len(C.ARCH_HEADS)
    hgap = 3.0
    head_w = (100 - 2 * margin - (nh - 1) * hgap) / nh
    head_h, y_head = 14.0, 22.0
    hx = margin + np.arange(nh) * (head_w + hgap)

    hub_x = xs[-1] + box_w / 2
    bus_y = y_head + head_h + 8.0
    ax.plot([hub_x, hub_x], [y_main, bus_y], color=C.INK_SOFT, linewidth=1.0, zorder=1)
    ax.plot([hx[0] + head_w / 2, hub_x], [bus_y, bus_y], color=C.INK_SOFT,
            linewidth=1.0, zorder=1)
    ax.text(hub_x + 1.2, (y_main + bus_y) / 2, C.ARCH_BLOCKS[-1][1], ha="left",
            va="center", fontsize=6.8, color=C.INK_SOFT, fontweight="bold")

    for i, (label, shape, fkey) in enumerate(C.ARCH_HEADS):
        ax.add_patch(FancyBboxPatch(
            (hx[i], y_head), head_w, head_h, boxstyle="round,pad=0.3,rounding_size=1.0",
            linewidth=1.2, edgecolor=_slot(i + 1), facecolor="#fbfbfa", zorder=2))
        ax.text(hx[i] + head_w / 2, y_head + head_h * 0.62, label, ha="center", va="center",
                fontsize=7.6, fontweight="bold", color=C.INK, linespacing=1.35, zorder=3)
        ax.text(hx[i] + head_w / 2, y_head + head_h * 0.18, C.FORMULAS[fkey], ha="center",
                va="center", fontsize=7.0, color=_slot(i + 1), zorder=3)
        cx = hx[i] + head_w / 2
        ax.add_patch(FancyArrowPatch(
            (cx, bus_y), (cx, y_head + head_h), arrowstyle="-|>",
            mutation_scale=10, linewidth=1.0, color=C.INK_SOFT, zorder=1))
        ax.text(hx[i] + head_w / 2, y_head - 1.6, shape, ha="center", va="top",
                fontsize=6.8, color=C.INK_SOFT)

    ax.text(50, 97, "GermoVision-Net: tensor flow", ha="center", va="top",
            fontsize=12, fontweight="bold", color=C.INK)
    ax.text(50, 89, f"B = batch size; window {C.T_IN} days, {C.N_FEATURES} features, "
                    f"horizon {C.HORIZON} days; receptive field {C.RECEPTIVE_FIELD} days "
                    f"> {C.T_IN}",
            ha="center", va="top", fontsize=8.2, color=C.INK_SOFT)
    ax.text(50, 82, "Parenthesized labels reference documentation equations",
            ha="center", va="top", fontsize=7.6, color=C.INK_SOFT, style="italic")
    return _finish(fig, spec)


def fig02_attention_heatmap(attention: np.ndarray, labels: list[str]) -> plt.Figure:
    spec = C.figure(2)
    fig, ax = plt.subplots(figsize=(C.FIG_W, 4.0))
    ax.grid(False)

    im = ax.imshow(attention, aspect="auto", cmap=C.CMAP_SEQUENTIAL,
                   interpolation="nearest", vmin=0.0)
    peak = attention.argmax(axis=1)
    for row, col in enumerate(peak):
        ax.plot([col, col], [row - 0.5, row + 0.5], color=_slot(7), linewidth=2.0,
                solid_capstyle="butt", zorder=3)

    ax.set_xlabel(f"Day of observation window (total {C.T_IN})")
    ax.set_ylabel("Emergence cases")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(np.arange(0, C.T_IN, 6))
    ax.set_xticklabels([f"-{C.T_IN - 1 - t}" for t in np.arange(0, C.T_IN, 6)])
    ax.set_title("Where the model looks inside the observation window")

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label("attention weight a_t", fontsize=9, color=C.INK_SOFT)
    cb.outline.set_visible(False)
    ax.plot([], [], color=_slot(7), linewidth=2.0, label="peak-weight day")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncols=1)
    return _finish(fig, spec)


def fig03_geo_drift(geo: dict) -> plt.Figure:
    spec = C.figure(3)
    fig, ax = plt.subplots(figsize=(C.FIG_W, 5.0))
    ax.grid(False)
    ax.set_axis_off()

    coords, eta, share = geo["coords"], np.asarray(geo["eta"]), np.asarray(geo["share"])
    names = geo["names"]

    fmax = max((w for *_, w in geo["edges"]), default=1.0)
    for i, j, w in geo["edges"]:
        ax.plot(coords[[i, j], 0], coords[[i, j], 1], color="#c9ccd1",
                linewidth=0.4 + 2.6 * w / fmax, zorder=1, solid_capstyle="round")

    size = 90 + 1500 * share / max(share.max(), C.EPS)
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=size, c=eta, cmap=C.CMAP_SEQUENTIAL,
                    edgecolors=C.SURFACE, linewidths=2.0, zorder=3, vmin=0)

    for r in np.argsort(eta)[:5]:
        ax.annotate(f"{names[r]}\n{eta[r]:.0f} d", (coords[r, 0], coords[r, 1]),
                    textcoords="offset points", xytext=(0, 16), ha="center",
                    fontsize=8, fontweight="bold", color=C.INK, zorder=4)

    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("forecast arrival time, days", fontsize=9, color=C.INK_SOFT)
    cb.outline.set_visible(False)
    ax.set_title(f"Forecast arrival of lineage {geo['lineage']} (day {geo['t_now']})")
    return _finish(fig, spec,
                   note="Node size = forecast variant share; edge width = gravity-model flow "
                        "(20 strongest links); color = forecast arrival time.")


def fig04_training_curves(history: dict) -> plt.Figure:
    spec = C.figure(4)
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(C.FIG_W, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.14})

    ep = np.asarray(history["epoch"])
    tr = np.asarray(history["train_loss"], dtype=float)
    va = np.asarray(history["val_loss"], dtype=float)
    stages = np.asarray(history["stage_id"])

    def normalise(series: np.ndarray) -> np.ndarray:
        out = np.empty_like(series)
        for tag in np.unique(stages):
            m = stages == tag
            base = series[m][0]
            out[m] = series[m] / (abs(base) + C.EPS) * (1 if base >= 0 else -1)
        return out

    ax.plot(ep, normalise(tr), color=_slot(0), label="training loss")
    ax.plot(ep, normalise(va), color=_slot(1), label="validation loss")
    ax.axhline(1.0, color=C.GRID, linewidth=1.0, zorder=1)

    lo, hi = ax.get_ylim()
    for (start_ep, _tag), name in zip(history["stage_bounds"],
                                      ("self-supervision", "supervised", "PPO"), strict=False):
        ax.axvline(start_ep, color=C.INK_SOFT, linestyle=":", linewidth=1.0, zorder=1)
        ax.text(start_ep + 0.8, lo + (hi - lo) * 0.04, name, fontsize=8.4,
                color=C.INK_SOFT, va="bottom", ha="left", fontweight="bold")

    if history.get("early_stop_epoch"):
        e = int(history["early_stop_epoch"])
        pos = int(np.flatnonzero(ep == e)[0])
        y = normalise(va)[pos]
        ax.plot([e], [y], marker="o", markersize=C.MARKER_SIZE, color=_slot(1),
                markeredgecolor=C.SURFACE, markeredgewidth=1.5, zorder=5)
        ax.annotate(f"early stop, epoch {e}", (e, y), textcoords="offset points",
                    xytext=(10, 12), fontsize=8.2, color=C.INK, fontweight="bold")

    ax.set_ylabel("loss, fraction of stage's first epoch")
    ax.set_title("Training over three stages")
    ax.legend(loc="upper right", ncols=2)

    ax2.plot(history["ppo_epoch"], history["ppo_reward"], color=_slot(2))
    _direct_label(ax2, history["ppo_epoch"][-1], history["ppo_reward"][-1],
                  "  RL reward", _slot(2))
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mean reward")
    ax2.set_xlim(ax.get_xlim())
    return _finish(fig, spec,
                   note="Loss values are normalized to each stage's first epoch: the three "
                        "stages optimize different quantities and their absolute values are "
                        "incomparable. RL reward is shown only for stage C.")


def fig05_forecast_intervals(hist: np.ndarray, truth: np.ndarray, mu: np.ndarray,
                             sigma: np.ndarray, mae_value: float, picp80: float) -> plt.Figure:
    spec = C.figure(5)
    from scipy import stats as st

    fig, ax = plt.subplots()
    t_obs = np.arange(-C.T_IN + 1, 1)
    t_fut = np.arange(1, C.HORIZON + 1)

    ax.plot(t_obs, hist, color=C.INK, linewidth=C.LINE_WIDTH, label="actual trajectory")
    ax.plot(t_fut, truth, color=C.INK, linewidth=C.LINE_WIDTH)
    ax.plot(t_fut, mu, color=_slot(0), linewidth=C.LINE_WIDTH, label="median forecast")

    for level, alpha in zip(C.INTERVAL_LEVELS, (C.CI_ALPHA_INNER, C.CI_ALPHA_OUTER), strict=True):
        half = st.norm.ppf(0.5 + level / 2) * sigma
        ax.fill_between(t_fut, mu - half, mu + half, color=_slot(0), alpha=alpha,
                        linewidth=0, label=f"{int(level * 100)}% interval")

    ax.axvline(0.5, color=C.INK_SOFT, linestyle="--", linewidth=1.0)
    ylo, yhi = ax.get_ylim()
    ax.text(-1, yhi, "observation ", ha="right", va="top", fontsize=9, color=C.INK_SOFT)
    ax.text(2, yhi, " forecast", ha="left", va="top", fontsize=9, color=C.INK_SOFT)

    mae_spec = C.METRIC_INDEX["MAE"]
    picp_spec = C.METRIC_INDEX["PICP80"]
    ax.text(0.985, 0.04,
            f"{mae_spec.key} = {mae_spec.format(mae_value)}\n"
            f"{picp_spec.key} = {picp_spec.format(picp80)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=C.INK_SOFT, family="DejaVu Sans")

    ax.set_xlabel("day relative to forecast origin T")
    ax.set_ylabel(C.FEATURES[C.FEATURE_INDEX["z"]].label)
    ax.set_title("Forecast vs ground truth on a held-out variant")
    ax.legend(loc="upper left")
    return _finish(fig, spec)


def fig06_feature_attribution(attributions: np.ndarray) -> plt.Figure:
    spec = C.figure(6)
    fig, ax = plt.subplots(figsize=(C.FIG_W, 4.8))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    order = np.argsort(np.abs(attributions))
    vals = attributions[order]
    names = [C.FEATURES[i].name_en for i in order]

    lim = float(np.abs(vals).max()) * 1.32
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.8, len(vals) - 0.2)

    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = plt.get_cmap(C.CMAP_DIVERGING)
    for i, v in enumerate(vals):
        _rounded_barh(ax, i, v, 0.62, cmap(norm(v)))
        off = lim * 0.03
        ax.text(v + (off if v >= 0 else -off), i, f"{v:+.3f}",
                ha="left" if v >= 0 else "right", va="center", fontsize=8.2,
                color=C.INK_SOFT)

    ax.axvline(0, color=C.INK_SOFT, linewidth=1.0, zorder=2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.6)
    ax.set_xlabel("contribution to logit-share forecast, integrated gradients")
    ax.set_title("What drives the forecast")
    return _finish(fig, spec)


def fig07_mutation_forecast_example(
    days: np.ndarray, freq: np.ndarray, fut_days: np.ndarray, fut_mu: np.ndarray,
    fut_lo: np.ndarray, fut_hi: np.ndarray, t_alarm: float, t_dom: float,
    label: str,
) -> plt.Figure:
    spec = C.figure(7)
    fig, ax = plt.subplots(figsize=(C.FIG_W, 4.6))

    ax.plot(days, freq, color=C.INK, label="observed share")
    ax.plot(fut_days, fut_mu, color=_slot(0), label="forecast")
    ax.fill_between(fut_days, fut_lo, fut_hi, color=_slot(0), alpha=C.CI_ALPHA_OUTER,
                    linewidth=0, label="80% interval")

    ax.axhline(C.DOMINANCE_THRESHOLD, color=C.INK_SOFT, linestyle="--", linewidth=1.0)
    ax.text(fut_days[-1], C.DOMINANCE_THRESHOLD + 0.012,
            f"dominance threshold {C.DOMINANCE_THRESHOLD:.0%} ", fontsize=8.4,
            color=C.INK_SOFT, ha="right", va="bottom")

    ax.axvline(t_alarm, color=_slot(1), linewidth=1.6)
    ax.axvline(t_dom, color=_slot(7), linewidth=1.6)
    top = ax.get_ylim()[1]
    ax.text(t_alarm, top * 0.97, "alarm ", color=_slot(1), fontsize=8.5,
            fontweight="bold", ha="right", va="top")
    ax.text(t_dom, top * 0.88, " dominance", color=_slot(7), fontsize=8.5,
            fontweight="bold", ha="left", va="top")

    y_arrow = C.DOMINANCE_THRESHOLD * 0.45
    ax.annotate("", xy=(t_dom, y_arrow), xytext=(t_alarm, y_arrow),
                arrowprops={"arrowstyle": "<|-|>", "color": C.INK, "linewidth": 1.3,
                            "shrinkA": 0, "shrinkB": 0})
    ax.text((t_alarm + t_dom) / 2, y_arrow * 1.12,
            f"lead time = {t_dom - t_alarm:.0f} days", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=C.INK)

    ax.set_xlabel("observation day")
    ax.set_ylabel(C.FEATURES[C.FEATURE_INDEX["f"]].name_en)
    ax.set_ylim(0, max(1.0, float(np.max(fut_hi)) * 1.1))
    ax.set_title(f"Worked example: {label}")
    ax.legend(loc="upper left")
    return _finish(fig, spec)


def fig08_calibration(nominal: np.ndarray, actual: np.ndarray) -> plt.Figure:
    spec = C.figure(8)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))

    ax.plot([0, 1], [0, 1], color=C.INK_SOFT, linestyle="--", linewidth=1.2,
            label="ideal calibration")
    ax.plot(nominal, actual, color=_slot(0), marker="o", markersize=C.MARKER_SIZE,
            markeredgecolor=C.SURFACE, markeredgewidth=1.4, label="GermoVision-Net")

    ax.fill_between([0, 1], [0, 1], [1, 1], color=C.INK_SOFT, alpha=0.05, linewidth=0)
    ax.text(0.06, 0.9, "under-confident\n(intervals too wide)", fontsize=8.2,
            color=C.INK_SOFT, va="top")
    ax.text(0.94, 0.1, "over-confident\n(intervals too narrow)", fontsize=8.2,
            color=C.INK_SOFT, ha="right", va="bottom")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(axis="both")
    ax.set_xlabel("nominal confidence level")
    ax.set_ylabel("actual coverage PICP")
    ax.set_title("Reliability diagram")
    ax.legend(loc="lower right")
    return _finish(fig, spec)


def fig09_error_by_horizon(mae: np.ndarray, se: np.ndarray, ar_mae: np.ndarray) -> plt.Figure:
    spec = C.figure(9)
    fig, ax = plt.subplots()
    d = np.arange(1, len(mae) + 1)

    ax.plot(d, mae, color=_slot(0), label="GermoVision-Net, direct multi-horizon")
    ax.fill_between(d, mae - se, mae + se, color=_slot(0), alpha=C.CI_ALPHA_INNER,
                    linewidth=0, label=f"+/- SE across {C.N_FOLDS} folds")
    ax.plot(d, ar_mae, color=_slot(1), linestyle="--",
            label="autoregressive baseline")

    ax.annotate(f"{mae[-1]:.3f}", (d[-1], mae[-1]), textcoords="offset points",
                xytext=(6, -2), fontsize=8.4, color=C.INK_SOFT)
    ax.annotate(f"{ar_mae[-1]:.3f}", (d[-1], ar_mae[-1]), textcoords="offset points",
                xytext=(6, -2), fontsize=8.4, color=C.INK_SOFT)

    ax.set_xlabel("forecast horizon d, days")
    ax.set_ylabel(C.METRIC_INDEX["MAE"].name_en)
    ax.set_xlim(1, len(mae))
    ax.set_title("Direct multi-horizon scheme does not accumulate error")
    ax.legend(loc="upper left")
    return _finish(fig, spec)


def fig10_baselines(names: list[str], mae: list[float], se: list[float]) -> plt.Figure:
    spec = C.figure(10)
    fig, ax = plt.subplots(figsize=(C.FIG_W, 3.9))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    order = np.argsort(mae)[::-1]
    names = [names[i] for i in order]
    mae = [mae[i] for i in order]
    se = [se[i] for i in order]

    ax.set_xlim(0, max(m + s for m, s in zip(mae, se, strict=True)) * 1.24)
    ax.set_ylim(-0.7, len(names) - 0.3)

    for i, (n, m, s) in enumerate(zip(names, mae, se, strict=True)):
        ours = n == C.MODEL_NAME
        _rounded_barh(ax, i, m, 0.58, _slot(0) if ours else C.MUTED_BAR)
        ax.errorbar(m, i, xerr=s, fmt="none", ecolor=C.INK_SOFT, elinewidth=1.1,
                    capsize=3, zorder=4)
        ax.text(m + s + ax.get_xlim()[1] * 0.015, i, C.METRIC_INDEX["MAE"].format(m),
                va="center", fontsize=8.6, color=C.INK if ours else C.INK_SOFT,
                fontweight="bold" if ours else "normal")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f"{C.METRIC_INDEX['MAE'].name_en} {C.METRIC_INDEX['MAE'].arrow}, "
                  f"whiskers = +/- SE over {C.N_FOLDS} folds")
    ax.set_title("Comparison with baseline models")
    return _finish(fig, spec)


def fig11_ablation(labels: list[str], mae: list[float], delta: list[float]) -> plt.Figure:
    spec = C.figure(11)
    fig, ax = plt.subplots(figsize=(C.FIG_W, 5.0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    ax.set_xlim(0, max(mae) * 1.30)
    ax.set_ylim(-0.7, len(labels) - 0.3)

    for i, (lab, m, dl) in enumerate(zip(labels, mae, delta, strict=True)):
        ref = lab == dict(C.ABLATIONS)[C.ABLATION_REFERENCE]
        _rounded_barh(ax, i, m, 0.58, _slot(0) if ref else C.MUTED_BAR)
        text = C.METRIC_INDEX["MAE"].format(m)
        if not ref:
            text += f"   d {dl:+.3f}"
        ax.text(m + ax.get_xlim()[1] * 0.015, i, text, va="center", fontsize=8.4,
                color=C.INK if ref else C.INK_SOFT,
                fontweight="bold" if ref else "normal")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlabel(f"{C.METRIC_INDEX['MAE'].name_en} {C.METRIC_INDEX['MAE'].arrow}; "
                  "d = change vs full model")
    ax.set_title("Ablation study")
    return _finish(fig, spec)


def fig12_feature_correlation(corr: dict) -> plt.Figure:
    spec = C.figure(12)
    r = np.asarray(corr["r_matrix"])
    sig = np.asarray(corr["significant_matrix"])
    names = corr["names"]
    horizons = corr["horizons"]

    fig, ax = plt.subplots(figsize=(C.FIG_W, 5.2))
    ax.grid(False)

    lim = float(np.abs(r).max())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    ax.imshow(r, cmap=C.CMAP_DIVERGING, norm=norm, aspect="auto")

    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if not sig[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#e8e8e6",
                                           edgecolor=C.SURFACE, linewidth=1.0, zorder=2))
            ax.text(j, i, f"{r[i, j]:+.2f}", ha="center", va="center", fontsize=7.8,
                    color=C.INK_SOFT if not sig[i, j] else C.INK, zorder=3,
                    fontweight="normal" if not sig[i, j] else "bold")

    ctrl_row = corr["keys"].index(C.CONTROL_FEATURE.key)
    ax.add_patch(plt.Rectangle((-0.5, ctrl_row - 0.5), r.shape[1], 1, fill=False,
                               edgecolor=C.INK, linewidth=1.6, zorder=4))

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"d = {h}" for h in horizons])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.4)
    ax.set_xlabel("target-variable horizon")
    ax.set_title("Correlation with logit-share drift")
    return _finish(fig, spec,
                   note=f"Gray cells: not significant (Benjamini-Hochberg, alpha = 0.05, "
                        f"N = {corr['n']}). Boxed row: negative control — it must be "
                        "insignificant; otherwise the procedure is broken.")


def fig13_cusum(cusum: dict, t_true_growth: int) -> plt.Figure:
    spec = C.figure(13)
    fig, ax = plt.subplots()
    s = np.asarray(cusum["s"])
    t = np.arange(len(s))
    h = cusum["h"]

    ax.plot(t, s, color=_slot(0), label="statistic S_t")
    ax.fill_between(t, h, s, where=s > h, color=_slot(7), alpha=0.22, linewidth=0,
                    label="above threshold")
    ax.axhline(h, color=C.INK_SOFT, linestyle="--", linewidth=1.2)
    ax.text(t[0], h, f" threshold h = {h:.3f}", va="bottom", fontsize=8.4, color=C.INK_SOFT)

    ax.axvline(t_true_growth, color=C.INK, linewidth=1.4)
    _direct_label(ax, t_true_growth, ax.get_ylim()[1] * 0.9, " growth onset", C.INK)

    if cusum["t_alarm"] is not None:
        ta = cusum["t_alarm"]
        ax.axvline(ta, color=_slot(1), linewidth=1.6)
        _direct_label(ax, ta, ax.get_ylim()[1] * 0.78, " alarm", _slot(1))
        ax.annotate("", xy=(t_true_growth, ax.get_ylim()[1] * 0.5),
                    xytext=(ta, ax.get_ylim()[1] * 0.5),
                    arrowprops={"arrowstyle": "<|-|>", "color": C.INK, "linewidth": 1.2})
        ax.text((ta + t_true_growth) / 2, ax.get_ylim()[1] * 0.52,
                f"{t_true_growth - ta:.0f} days", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C.INK)

    ax.set_xlabel("observation day")
    ax.set_ylabel("cumulative statistic S_t")
    ax.set_title("CUSUM detector of drift acceleration")
    ax.legend(loc="upper left")
    return _finish(fig, spec)


def fig14_rl_allocation(alloc: np.ndarray, region_names: list[str],
                        outbreak: tuple[int, int] | None = None) -> plt.Figure:
    spec = C.figure(14)
    fig, ax = plt.subplots(figsize=(C.FIG_W, 4.4))
    ax.grid(False)

    im = ax.imshow(alloc, aspect="auto", cmap=C.CMAP_SEQUENTIAL, interpolation="nearest",
                   vmin=0.0)
    ax.set_yticks(range(len(region_names)))
    ax.set_yticklabels(region_names, fontsize=8.4)
    ax.set_xlabel("day")
    ax.set_ylabel("region")
    ax.set_title("Where the agent reallocates sequencing budget")

    if outbreak is not None:
        region, day = outbreak
        ax.add_patch(plt.Rectangle((day - 0.5, region - 0.5), alloc.shape[1] - day, 1,
                                   fill=False, edgecolor=_slot(7), linewidth=1.8, zorder=3))
        ax.annotate(f"emerging outbreak: {region_names[region]}",
                    (day, region), textcoords="offset points", xytext=(8, -18),
                    fontsize=8.4, fontweight="bold", color=_slot(7), zorder=4)

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label("share of daily budget", fontsize=9, color=C.INK_SOFT)
    cb.outline.set_visible(False)
    return _finish(fig, spec)


def fig15_lead_time_hist(lead_times: np.ndarray) -> plt.Figure:
    spec = C.figure(15)
    fig, ax = plt.subplots()

    ax.hist(lead_times, bins=18, color=_slot(0), alpha=0.85, edgecolor=C.SURFACE,
            linewidth=1.0, zorder=3)
    median = float(np.median(lead_times))

    ax.axvline(median, color=C.INK, linewidth=1.8, zorder=4)
    ax.text(median, ax.get_ylim()[1] * 0.97, f" median {median:.1f} d",
            fontsize=8.8, fontweight="bold", color=C.INK, va="top")

    ax.axvline(C.COVTRANSFORMER_LT, color=_slot(1), linestyle="--", linewidth=1.6, zorder=4)
    ax.text(C.COVTRANSFORMER_LT, ax.get_ylim()[1] * 0.80,
            f" CovTransformer {C.COVTRANSFORMER_LT} d",
            fontsize=8.8, color=_slot(1), va="top")

    ax.set_xlabel(C.METRIC_INDEX["LT"].name_en)
    ax.set_ylabel("number of held-out events")
    ax.set_title("Distribution of detection lead time")
    return _finish(fig, spec)


def build_all_figures(results: dict, history: dict, artefacts: dict) -> list[str]:
    C.FIGDIR.mkdir(parents=True, exist_ok=True)
    a = artefacts["attention"]
    ex = artefacts["forecast_example"]
    mx = artefacts["mutation_example"]
    cal = results["calibration"]
    eh = results["error_by_horizon"]
    bl = results["baseline_summary"]
    ab = results["ablation"]

    builders = {
        1: lambda: fig01_architecture(),
        2: lambda: fig02_attention_heatmap(a["weights"], a["labels"]),
        3: lambda: fig03_geo_drift(artefacts["geo"]),
        4: lambda: fig04_training_curves(history),
        5: lambda: fig05_forecast_intervals(ex["hist"], ex["truth"], ex["mu"], ex["sigma"],
                                            ex["mae"], ex["picp80"]),
        6: lambda: fig06_feature_attribution(np.asarray(results["attribution"])),
        7: lambda: fig07_mutation_forecast_example(
            mx["days"], mx["freq"], mx["fut_days"], mx["fut_mu"], mx["fut_lo"],
            mx["fut_hi"], mx["t_alarm"], mx["t_dom"], mx["label"]),
        8: lambda: fig08_calibration(np.asarray(cal["nominal"]), np.asarray(cal["actual"])),
        9: lambda: fig09_error_by_horizon(np.asarray(eh["mae"]), np.asarray(eh["se"]),
                                          np.asarray(eh["autoregressive"])),
        10: lambda: fig10_baselines(bl["names"], bl["mae"], bl["se"]),
        11: lambda: fig11_ablation([name for _, name in C.ABLATIONS],
                                   [ab[k]["MAE"] for k, _ in C.ABLATIONS],
                                   [ab[k]["delta_mae"] for k, _ in C.ABLATIONS]),
        12: lambda: fig12_feature_correlation(results["correlation"]),
        13: lambda: fig13_cusum(artefacts["cusum"], artefacts["cusum_true_growth"]),
        14: lambda: fig14_rl_allocation(np.asarray(artefacts["allocation"]),
                                        artefacts["region_names"], artefacts.get("outbreak")),
        15: lambda: fig15_lead_time_hist(np.asarray(results["lead_times"])),
    }

    made: list[str] = []
    for spec in C.FIGURES:
        builders[spec.number]()
        made.extend([spec.filename_png, spec.filename_pdf])
        plt.close("all")
    return made
