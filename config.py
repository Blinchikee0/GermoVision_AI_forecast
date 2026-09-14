from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures"
RESULTS_JSON = ROOT / "results.json"
METRICS_TABLE_MD = ROOT / "metrics_table.md"
CHECKPOINT_PT = ROOT / "net.pt"
CHECKPOINT_ONNX = ROOT / "net.onnx"
DATA_CACHE = ROOT / ".cache"

SEED = 20260910

T_IN = 42
HORIZON = 30
N_FEATURES = 12

DOMINANCE_THRESHOLD = 0.50
ALARM_PROB_THRESHOLD = 0.50

N_REGIONS = 12
N_LINEAGES = 8
N_DAYS = 900

N_FOLDS = 12
GAP_DAYS = 30

EPOCHS_SSL = 30
EPOCHS_SUP = 60
EPOCHS_PPO = 40
FREEZE_EPOCHS = 5
PPO_UNFREEZE_LAST = 10
EARLY_STOP_PATIENCE = 10

BATCH_SIZE = 256
LR_MAX = 3e-4
LR_UNFREEZE = 1e-5
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
GRAD_CLIP_NORM = 1.0
LLRD_GAMMA = 0.8

FOLD_EPOCHS_SSL = 10
FOLD_EPOCHS_SUP = 20

MASK_FRACTION = 0.15
MASK_SPAN_MIN = 3
MASK_SPAN_MAX = 7

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
DEPTH_WEIGHT_K = 50.0
PPO_CLIP_EPS = 0.2
GAE_GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_BETA = 0.01
L2_LAMBDA = 1e-4

LOSS_WEIGHTS = {
    "nll": 1.0,
    "focal": 0.5,
    "ppo": 1.0,
    "value": 0.5,
    "entropy": 0.01,
    "l2": 1e-4,
}

LAMBDA_IG = 1.0
LAMBDA_LT = 0.05
LAMBDA_COV = 0.3
LAMBDA_COST = 0.1
SEQ_BUDGET = 600.0

CNN_LAYERS = 5
CNN_KERNEL = 3
CNN_DILATIONS = (1, 2, 4, 8, 16)
CNN_CHANNELS = 64
CNN_DROPOUT = 0.2

LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_BIDIRECTIONAL = True
D_MODEL = LSTM_HIDDEN * (2 if LSTM_BIDIRECTIONAL else 1)

ATTN_HEADS = 8
ATTN_DK = 64

RECEPTIVE_FIELD = 1 + (CNN_KERNEL - 1) * (2 ** CNN_LAYERS - 1)

AUG_THIN_RHO = (0.1, 1.0)
AUG_SPLINE_KNOTS = 4
AUG_SPLINE_SIGMA = 0.1
AUG_MIXUP_ALPHA = 0.2
AUG_PROB = 0.5

EPS = 1e-8


@dataclass(frozen=True)
class Feature:
    key: str
    name_en: str
    unit: str
    lo: float
    hi: float

    @property
    def label(self) -> str:
        return f"{self.name_en}, {self.unit}" if self.unit else self.name_en


FEATURES: tuple[Feature, ...] = (
    Feature("f", "Lineage share", "", 0.0, 1.0),
    Feature("z", "Logit of share", "log-units", -12.0, 12.0),
    Feature("dz", "Logit growth rate", "1/day", -1.0, 1.0),
    Feature("n", "Sequencing depth", "reads", 0.0, 4000.0),
    Feature("H", "Lineage-distribution entropy", "bits", 0.0, 3.5),
    Feature("pi", "Nucleotide diversity", "x10^-3", 0.0, 5.0),
    Feature("omega", "Selection pressure dN/dS", "", 0.0, 4.0),
    Feature("ddG", "Folding free-energy change", "kcal/mol", -4.0, 4.0),
    Feature("E", "Immune-escape score", "", 0.0, 1.0),
    Feature("eps", "Epistasis score", "", -1.0, 1.0),
    Feature("Reff", "Effective reproduction number", "", 0.0, 4.0),
    Feature("m", "Population mobility index", "", 0.0, 2.0),
)

FEATURE_KEYS: tuple[str, ...] = tuple(f.key for f in FEATURES)
FEATURE_INDEX: dict[str, int] = {f.key: i for i, f in enumerate(FEATURES)}

RAW_SCALE_FEATURES: tuple[str, ...] = ("n",)

CONTROL_FEATURE = Feature("temp", "Temperature (control)", "C", -30.0, 40.0)

PALETTE: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)

LOW_CONTRAST_SLOTS: frozenset[int] = frozenset({2, 3, 4})
MAX_SLOTS_POINT_MARKS = 3

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#ffffff"
MUTED_BAR = "#b9b8b3"

CMAP_SEQUENTIAL = "Blues"
CMAP_DIVERGING = "RdBu_r"

FIG_W, FIG_H = 7.2, 4.2
FIG_DPI = 300
LINE_WIDTH = 2.0
MARKER_SIZE = 8.0
CI_ALPHA_INNER = 0.28
CI_ALPHA_OUTER = 0.15

INTERVAL_LEVELS = (0.50, 0.80)
CALIBRATION_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def apply_rcparams() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": (FIG_W, FIG_H),
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": FIG_DPI,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK_SOFT,
            "axes.facecolor": SURFACE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "text.color": INK,
            "figure.autolayout": False,
        }
    )


@dataclass(frozen=True)
class Metric:
    key: str
    name_en: str
    formula: str
    lower_is_better: bool
    fmt: str
    target: float | None = None
    target_note: str = ""

    @property
    def arrow(self) -> str:
        return "v" if self.lower_is_better else "^"

    def format(self, value: float | None) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        return format(value, self.fmt)

    @property
    def header(self) -> str:
        return f"{self.key} {self.arrow}"


METRICS: tuple[Metric, ...] = (
    Metric("MAE", "Mean absolute error", "mean|y - mu|", True, ".4f"),
    Metric("RMSE", "Root mean squared error", "sqrt(mean(y - mu)^2)", True, ".4f"),
    Metric(
        "MASE",
        "Mean absolute scaled error",
        "MAE / MAE of naive forecast",
        True,
        ".4f",
        target=1.0,
        target_note="target < 1",
    ),
    Metric(
        "CRPS",
        "Continuous ranked probability score",
        "sigma[z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]",
        True,
        ".4f",
    ),
    Metric("PICP50", "50% prediction-interval coverage", "share of y inside 50% CI", True, ".3f",
           target=0.50),
    Metric("PICP80", "80% prediction-interval coverage", "share of y inside 80% CI", True, ".3f",
           target=0.80),
    Metric("LT", "Detection lead time, days", "t_dom - t_alarm", False, ".1f"),
    Metric("AUPRC", "Area under precision-recall curve", "PR-curve area", False, ".3f"),
)

METRIC_INDEX: dict[str, Metric] = {m.key: m for m in METRICS}

FORBIDDEN_METRICS: frozenset[str] = frozenset({"MAPE", "SMAPE"})

COMPARISON_METRICS: tuple[str, ...] = ("MAE", "MASE", "CRPS", "LT")

PUBLISHED_BENCHMARKS: dict[str, dict] = {
    "Nextstrain MLR": {"MAE": "> 0.50", "LT": None, "source": "[9]", "unc": "regression CI"},
    "Single LSTM": {"MAE": "0.6098", "LT": None, "source": "[9]", "unc": "none"},
    "CovTransformer": {
        "MAE": "0.32-0.42", "LT": "48.7 (median 54)", "source": "[9]", "unc": "none"},
}
COVTRANSFORMER_LT = 48.7

MODEL_NAME = "GermoVision-Net"

BASELINES: tuple[tuple[str, str], ...] = (
    ("naive", "Naive forecast"),
    ("arima", "ARIMA(p,d,q)"),
    ("mlr", "Nextstrain MLR"),
    ("lstm", "Single LSTM"),
)

ABLATIONS: tuple[tuple[str, str], ...] = (
    ("cnn_only", "1D CNN only (original)"),
    ("bilstm", "+ BiLSTM"),
    ("attention", "+ Multi-head attention"),
    ("prob_head", "+ Probabilistic head (NLL)"),
    ("ssl", "+ Self-supervision (stage A)"),
    ("rl", "+ RL agent (PPO)"),
    ("full", "Full model"),
    ("reinforce", "REINFORCE instead of PPO"),
    ("minmax", "MinMax instead of robust scaling"),
    ("gauss_aug", "Gaussian noise instead of binomial augmentation"),
)
ABLATION_REFERENCE = "full"


@dataclass(frozen=True)
class FigureSpec:
    number: int
    stem: str
    caption: str
    section: str = ""

    @property
    def filename_png(self) -> str:
        return f"{self.stem}.png"

    @property
    def filename_pdf(self) -> str:
        return f"{self.stem}.pdf"

    @property
    def path_png(self) -> Path:
        return FIGDIR / self.filename_png

    @property
    def path_pdf(self) -> Path:
        return FIGDIR / self.filename_pdf

    @property
    def full_caption(self) -> str:
        return f"Figure {self.number} — {self.caption}"


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(1, "fig01_architecture",
               "GermoVision-Net architecture block diagram with tensor shapes", "2.4"),
    FigureSpec(2, "fig02_attention_heatmap",
               "temporal-attention weight heatmap", "2.4.4"),
    FigureSpec(3, "fig03_geo_drift",
               "geographic variant-drift forecast", "2.4.7"),
    FigureSpec(4, "fig04_training_curves",
               "training curves across three stages and RL reward", "2.7.2"),
    FigureSpec(5, "fig05_forecast_intervals",
               "forecast with 50%/80% intervals against ground truth", "2.7.2"),
    FigureSpec(6, "fig06_feature_attribution",
               "feature attribution via integrated gradients", "2.7.5"),
    FigureSpec(7, "fig07_mutation_forecast_example",
               "worked example: alarm and detection lead", "2.8.1"),
    FigureSpec(8, "fig08_calibration",
               "reliability diagram for probabilistic forecast", "2.7.1"),
    FigureSpec(9, "fig09_error_by_horizon",
               "error growth with forecast horizon", "2.7.1"),
    FigureSpec(10, "fig10_baselines",
               "MAE comparison against baseline models", "2.7.2"),
    FigureSpec(11, "fig11_ablation",
               "ablation study", "2.7.3"),
    FigureSpec(12, "fig12_feature_correlation",
               "feature-vs-target correlation matrix", "2.2.3"),
    FigureSpec(13, "fig13_cusum",
               "CUSUM detector of drift acceleration", "2.4.6"),
    FigureSpec(14, "fig14_rl_allocation",
               "RL sequencing-budget allocation trajectory", "2.4.8"),
    FigureSpec(15, "fig15_lead_time_hist",
               "distribution of detection lead time", "2.7.1"),
)

FIGURE_BY_NUMBER: dict[int, FigureSpec] = {f.number: f for f in FIGURES}
FIGURE_BY_STEM: dict[str, FigureSpec] = {f.stem: f for f in FIGURES}

FORMULAS: dict[str, str] = {
    "norm": "(16)",
    "cnn": "(17)-(20)",
    "lstm": "(21)-(23)",
    "attn": "(24)-(26)",
    "pool": "(27)",
    "prob": "(28)-(32)",
    "rl": "(33)-(41)",
    "geo": "(42)-(45)",
}


def tensor_shape(*dims: object) -> str:
    return "(" + ", ".join(str(d) for d in dims) + ")"


ARCH_BLOCKS: tuple[tuple[str, str, str | None], ...] = (
    ("Input\npanel", tensor_shape("B", T_IN, N_FEATURES), None),
    ("IQR\nnormalization", tensor_shape("B", T_IN, N_FEATURES), "norm"),
    (
        f"Causal CNN\n{CNN_LAYERS} layers, d 1..{CNN_DILATIONS[-1]}",
        tensor_shape("B", T_IN, CNN_CHANNELS), "cnn"),
    (f"BiLSTM\n{LSTM_LAYERS} x {LSTM_HIDDEN}", tensor_shape("B", T_IN, D_MODEL), "lstm"),
    (f"Attention\n{ATTN_HEADS} heads, d_k {ATTN_DK}", tensor_shape("B", T_IN, D_MODEL), "attn"),
    ("Pooling\nby a_t weights", tensor_shape("B", D_MODEL), "pool"),
)

ARCH_HEADS: tuple[tuple[str, str, str], ...] = (
    ("Probabilistic\nhead", tensor_shape("B", HORIZON, 2), "prob"),
    ("Emergence\nevent", tensor_shape("B", 1), "prob"),
    ("Dirichlet\npolicy", tensor_shape("B", N_REGIONS), "rl"),
    ("Critic V(s)", tensor_shape("B", 1), "rl"),
)


def figure(number: int) -> FigureSpec:
    if number not in FIGURE_BY_NUMBER:
        raise KeyError(f"figure {number} missing from FIGURES (available: {sorted(FIGURE_BY_NUMBER)})")
    return FIGURE_BY_NUMBER[number]


@dataclass(frozen=True)
class TableSpec:
    key: str
    title: str
    columns: tuple[str, ...]
    source_file: str


TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        "comparison",
        "Table 2.7.1 — Model comparison",
        ("Model", "MAE v", "MASE v", "CRPS v", "Lead time, days ^", "Uncertainty", "Source"),
        RESULTS_JSON.name,
    ),
    TableSpec(
        "ablation",
        "Table B.6 — Ablation study",
        ("Configuration", "MAE v", "CRPS v", "PICP (80%)", "Delta vs full"),
        RESULTS_JSON.name,
    ),
    TableSpec(
        "efficiency",
        "Table B.7 — Computational efficiency",
        ("Metric", "1D CNN baseline", "GermoVision-Net"),
        RESULTS_JSON.name,
    ),
)

TABLE_BY_KEY: dict[str, TableSpec] = {t.key: t for t in TABLES}

EFFICIENCY_ROWS: tuple[str, ...] = (
    "Parameters, M",
    "FLOPs per forecast",
    "Training time",
    "CPU inference, ms",
    "ONNX export",
)

NEXTSTRAIN_URL = "https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst"
NEXTSTRAIN_MAX_ROWS = 14_000_000
NEXTSTRAIN_TIMEOUT_S = 30
MIN_DAILY_DEPTH = 10
DATA_SOURCE_REAL = "Nextstrain open (GenBank derivatives)"
DATA_SOURCE_SYNTHETIC = "synthetic panel, multinomial logistic model"


@dataclass
class RunConfig:
    seed: int = SEED
    fast: bool = False
    folds: int = N_FOLDS
    use_real_data: bool = True
    stride: int = 2
    max_windows: int = 1_800
    main_max_windows: int = 4_500
    extras: dict = field(default_factory=dict)

    def epochs(self, stage: str, *, fold: bool) -> int:
        table = {
            "A": (FOLD_EPOCHS_SSL if fold else EPOCHS_SSL),
            "B": (FOLD_EPOCHS_SUP if fold else EPOCHS_SUP),
            "C": (0 if fold else EPOCHS_PPO),
        }
        n = table[stage]
        return max(1, n // 6) if self.fast else n


def assert_consistency(results: dict | None = None) -> list[str]:
    checks: list[str] = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(f"desync: {msg}")

    need(
        len(FEATURES) == N_FEATURES,
        f"FEATURES has {len(FEATURES)} but N_FEATURES = {N_FEATURES}",
    )
    need(len(set(FEATURE_KEYS)) == N_FEATURES, "feature keys not unique")
    for f in FEATURES:
        need(bool(f.name_en.strip()), f"feature {f.key} has empty English name")
        need(f.lo < f.hi, f"feature {f.key} has inverted range [{f.lo}, {f.hi}]")
    need(
        all(k in FEATURE_KEYS for k in RAW_SCALE_FEATURES),
        f"RAW_SCALE_FEATURES points outside FEATURES: {RAW_SCALE_FEATURES}",
    )
    need(
        CONTROL_FEATURE.key not in FEATURE_KEYS,
        "control feature must not be a model input")
    checks.append(f"features registry: {N_FEATURES} entries, unique keys, control outside model")

    need(len(CNN_DILATIONS) == CNN_LAYERS, "dilation count does not match CNN layer count")
    need(
        tuple(2**i for i in range(CNN_LAYERS)) == CNN_DILATIONS,
        f"dilations must double: expected "
        f"{tuple(2**i for i in range(CNN_LAYERS))}, got {CNN_DILATIONS}",
    )
    need(
        RECEPTIVE_FIELD > T_IN,
        f"receptive field {RECEPTIVE_FIELD} does not cover window {T_IN}: raise CNN_LAYERS",
    )
    need(
        LSTM_HIDDEN * (2 if LSTM_BIDIRECTIONAL else 1) == D_MODEL,
        "D_MODEL out of sync with BiLSTM hidden size")
    checks.append(f"receptive field {RECEPTIVE_FIELD} > window {T_IN}; d_model = {D_MODEL}")

    need(len(set(PALETTE)) == len(PALETTE), "palette has duplicate colors")
    need(
        all(c.startswith("#") and len(c) == 7 for c in PALETTE),
        "palette color not in #rrggbb form")
    need(max(LOW_CONTRAST_SLOTS) < len(PALETTE), "LOW_CONTRAST_SLOTS points outside palette")
    need(len(PALETTE) >= MAX_SLOTS_POINT_MARKS, "MAX_SLOTS_POINT_MARKS exceeds palette size")
    need("viridis" not in (CMAP_SEQUENTIAL, CMAP_DIVERGING), "rainbow colormaps forbidden")
    checks.append(
        f"palette: {len(PALETTE)} slots, {len(LOW_CONTRAST_SLOTS)} require direct labeling")

    need(len(set(m.key for m in METRICS)) == len(METRICS), "metric keys not unique")
    need(not (FORBIDDEN_METRICS & {m.key for m in METRICS}), "forbidden metric present in registry")
    for key in COMPARISON_METRICS:
        need(key in METRIC_INDEX, f"COMPARISON_METRICS references missing metric {key}")
    checks.append(f"metrics registry: {len(METRICS)} entries, MAPE absent")

    need(len(set(f.number for f in FIGURES)) == len(FIGURES), "figure numbers repeat")
    need(len(set(f.stem for f in FIGURES)) == len(FIGURES), "figure file names repeat")
    need(
        [f.number for f in FIGURES] == list(range(1, len(FIGURES) + 1)),
        "figure numbers must run sequentially from 1",
    )
    for f in FIGURES:
        need(bool(f.caption.strip()), f"figure {f.number} has empty caption")
    checks.append(f"figures registry: {len(FIGURES)} entries, sequential numbers, unique names")

    need(len(set(t.key for t in TABLES)) == len(TABLES), "table keys repeat")
    checks.append(f"tables registry: {len(TABLES)} entries")

    if results is None:
        return checks

    for number in FIGURE_BY_NUMBER:
        spec = figure(number)
        need(
            spec.path_png.exists(),
            f"figure {number} declared but file {spec.filename_png} missing",
        )
        need(
            spec.path_pdf.exists(),
            f"figure {number} declared but file {spec.filename_pdf} missing",
        )
    checks.append(f"all {len(FIGURES)} figures saved as PNG and PDF")

    need(
        results.get("horizon") == HORIZON,
        f"results have horizon {results.get('horizon')}, config has {HORIZON}",
    )
    need(
        results.get("window") == T_IN,
        f"results have window {results.get('window')}, config has {T_IN}",
    )
    need(
        results.get("n_features") == N_FEATURES,
        f"results have {results.get('n_features')} features, config has {N_FEATURES}",
    )
    need(
        list(results.get("feature_keys", ())) == list(FEATURE_KEYS),
        "feature key order in results differs from FEATURES",
    )
    checks.append(f"results consistent with config: window {T_IN}, horizon {HORIZON}")

    main = results.get("metrics", {}).get(MODEL_NAME, {})
    for key in COMPARISON_METRICS:
        need(
            key in main,
            f"metric {key} in COMPARISON_METRICS but not computed for {MODEL_NAME}",
        )
    need(
        len(results.get("error_by_horizon", {}).get("mae", [])) == HORIZON,
        f"error-by-horizon curve must have {HORIZON} points",
    )
    checks.append(
        f"metrics {', '.join(COMPARISON_METRICS)} computed; "
        f"horizon curve has {HORIZON} points"
    )

    abl = results.get("ablation", {})
    for key, _ in ABLATIONS:
        need(key in abl, f"ablation configuration '{key}' declared but not computed")
    checks.append(f"ablation: all {len(ABLATIONS)} configurations computed")

    corr = results.get("correlation", {})
    if corr:
        need(
            CONTROL_FEATURE.key in corr.get("significant", {}),
            "control feature missing from correlation matrix",
        )
        need(
            corr["significant"][CONTROL_FEATURE.key] is False,
            "control feature came out significant — procedure broken, not biology",
        )
        checks.append("control feature (temperature) insignificant, as required")

    return checks
