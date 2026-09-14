from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

import config as C

log = logging.getLogger("germovision.data")


@dataclass
class Panel:
    counts: np.ndarray
    depth: np.ndarray
    features: np.ndarray
    control: np.ndarray
    region_names: list[str]
    lineage_names: list[str]
    coords: np.ndarray
    population: np.ndarray
    true_fitness: np.ndarray
    arrival_day: np.ndarray
    source: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.counts.shape

    @property
    def n_days(self) -> int:
        return self.counts.shape[2]


def _try_load_nextstrain(rng: np.random.Generator) -> Panel | None:
    try:
        import io
        import urllib.request
        from collections import Counter, defaultdict

        import zstandard
    except ImportError as exc:
        log.info("Nextstrain unavailable: missing dependency (%s)", exc)
        return None

    try:
        req = urllib.request.Request(C.NEXTSTRAIN_URL, headers={"User-Agent": "GermoVision/2.0"})
        resp = urllib.request.urlopen(req, timeout=C.NEXTSTRAIN_TIMEOUT_S)
    except Exception as exc:
        log.info("Nextstrain unavailable: %s", exc)
        return None

    try:
        dctx = zstandard.ZstdDecompressor()
        stream = io.TextIOWrapper(dctx.stream_reader(resp), encoding="utf-8", errors="replace")
        header = stream.readline().rstrip("\n").split("\t")
        need = {"date": None, "country": None, "Nextstrain_clade": None}
        for name in need:
            if name not in header:
                log.info("Nextstrain: metadata has no column %s", name)
                return None
            need[name] = header.index(name)

        obs: dict[tuple[str, str], Counter] = defaultdict(Counter)
        countries: Counter = Counter()
        clades: Counter = Counter()
        rows = 0
        for line in stream:
            rows += 1
            if rows > C.NEXTSTRAIN_MAX_ROWS:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(need.values()):
                continue
            date = parts[need["date"]]
            if len(date) != 10 or "X" in date:
                continue
            country = parts[need["country"]]
            clade = parts[need["Nextstrain_clade"]]
            if not country or not clade or clade == "?":
                continue
            obs[(country, date)][clade] += 1
            countries[country] += 1
            clades[clade] += 1
    except Exception as exc:
        log.info("Nextstrain: read interrupted (%s)", exc)
        return None
    finally:
        resp.close()

    if len(countries) < C.N_REGIONS or len(clades) < C.N_LINEAGES:
        log.info(
            "Nextstrain: only %d countries and %d clades collected — too few for panel %dx%d",
            len(countries), len(clades), C.N_REGIONS, C.N_LINEAGES,
        )
        return None

    from datetime import date as _date

    per_day: dict[int, int] = defaultdict(int)
    for (_, ds), cnt in obs.items():
        try:
            per_day[_date.fromisoformat(ds).toordinal()] += sum(cnt.values())
        except ValueError:
            continue
    if not per_day:
        log.info("Nextstrain: no parseable dates")
        return None

    lo_o, hi_o = min(per_day), max(per_day)
    span = min(C.N_DAYS, hi_o - lo_o + 1)
    dense = np.array([per_day.get(o, 0) for o in range(lo_o, hi_o + 1)], dtype=np.int64)
    csum = np.concatenate([[0], np.cumsum(dense)])
    totals = csum[span:] - csum[:-span] if len(csum) > span else np.array([csum[-1]])
    start_o = lo_o + int(np.argmax(totals))
    ordinals = list(range(start_o, start_o + span))
    dates = [_date.fromordinal(o).isoformat() for o in ordinals]
    di = {ds: i for i, ds in enumerate(dates)}
    in_window = set(dates)

    win_countries: Counter = Counter()
    win_clades: Counter = Counter()
    for (country, ds), cnt in obs.items():
        if ds in in_window:
            if sum(cnt.values()) >= C.MIN_DAILY_DEPTH:
                win_countries[country] += 1
            for clade, k in cnt.items():
                win_clades[clade] += k
    if len(win_countries) < C.N_REGIONS or len(win_clades) < C.N_LINEAGES:
        log.info("Nextstrain: only %d countries and %d clades inside dense window",
                 len(win_countries), len(win_clades))
        return None

    top_c = [c for c, _ in win_countries.most_common(C.N_REGIONS)]
    top_l = [c for c, _ in win_clades.most_common(C.N_LINEAGES)]

    R, L, T = len(top_c), len(top_l), len(dates)
    counts = np.zeros((R, L, T), dtype=np.float64)
    li_of = {c: i for i, c in enumerate(top_l)}
    for ri, country in enumerate(top_c):
        for ds, t in di.items():
            cnt = obs.get((country, ds))
            if not cnt:
                continue
            for clade, k in cnt.items():
                li = li_of.get(clade)
                if li is not None:
                    counts[ri, li, t] = k

    depth = counts.sum(axis=1)
    coords, population, labels = _geo_for(top_c)
    log.info("Data source: %s (%d rows, %d countries, %d days, %s..%s)",
             C.DATA_SOURCE_REAL, rows, R, T, dates[0], dates[-1])
    log.info("Median daily depth per country: %s",
             ", ".join(f"{c} {np.median(depth[i]):.0f}" for i, c in enumerate(top_c)))
    log.info("Clades: %s", ", ".join(top_l))
    return _finalise_panel(
        counts, depth, labels, top_l, rng, source=C.DATA_SOURCE_REAL,
        true_fitness=None, coords=coords, population=population,
    )


REGION_NAMES = [
    "Aktobe", "Almaty", "Astana", "Shymkent", "Karaganda", "Pavlodar",
    "Atyrau", "Kostanay", "Taraz", "Aktau", "Semey", "Uralsk",
]

COUNTRY_GEO: dict[str, tuple[float, float, float]] = {
    "USA": (39.8, -98.6, 335.0), "United Kingdom": (54.0, -2.0, 67.3),
    "Germany": (51.2, 10.4, 83.3), "Denmark": (56.0, 10.0, 5.9),
    "France": (46.6, 2.2, 68.0), "Japan": (36.2, 138.3, 124.5),
    "Canada": (56.1, -106.3, 38.9), "India": (20.6, 79.0, 1417.0),
    "Sweden": (60.1, 18.6, 10.5), "Switzerland": (46.8, 8.2, 8.8),
    "Spain": (40.5, -3.7, 47.6), "Netherlands": (52.1, 5.3, 17.6),
    "Australia": (-25.3, 133.8, 26.0), "Brazil": (-14.2, -51.9, 215.3),
    "Italy": (41.9, 12.6, 59.0), "Belgium": (50.5, 4.5, 11.7),
    "Austria": (47.5, 14.6, 9.0), "Israel": (31.0, 34.9, 9.6),
    "Poland": (51.9, 19.1, 36.8), "Turkey": (39.0, 35.2, 85.3),
    "Mexico": (23.6, -102.6, 127.5), "Norway": (60.5, 8.5, 5.5),
    "Finland": (61.9, 25.7, 5.6), "Ireland": (53.4, -8.2, 5.1),
    "Portugal": (39.4, -8.2, 10.3), "Czech Republic": (49.8, 15.5, 10.5),
    "South Africa": (-30.6, 22.9, 60.4), "Singapore": (1.35, 103.8, 5.6),
    "Indonesia": (-0.8, 113.9, 275.5), "Russia": (61.5, 105.3, 144.2),
    "Slovenia": (46.2, 14.99, 2.1), "Luxembourg": (49.8, 6.1, 0.65),
    "Slovakia": (48.7, 19.7, 5.4), "Argentina": (-38.4, -63.6, 46.2),
    "Hungary": (47.2, 19.5, 9.6), "Greece": (39.1, 21.8, 10.4),
    "Romania": (45.9, 25.0, 19.0), "Croatia": (45.1, 15.2, 3.9),
    "Thailand": (15.9, 101.0, 71.7), "South Korea": (35.9, 127.8, 51.7),
    "New Zealand": (-40.9, 174.9, 5.1), "Chile": (-35.7, -71.5, 19.6),
    "Peru": (-9.2, -75.0, 34.0), "Nigeria": (9.1, 8.7, 218.5),
}


def _geo_for(names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    known = [COUNTRY_GEO[n] for n in names if n in COUNTRY_GEO]
    med_pop = float(np.median([k[2] for k in known])) if known else 20.0
    coords, pops, labels = [], [], []
    for n in names:
        if n in COUNTRY_GEO:
            lat, lon, pop = COUNTRY_GEO[n]
            coords.append((lon, lat))
            pops.append(pop)
            labels.append(n)
        else:
            coords.append((float(np.mean([c[1] for c in known])) if known else 0.0,
                           float(np.mean([c[0] for c in known])) if known else 0.0))
            pops.append(med_pop)
            labels.append(f"{n} (coords unknown)")
    return np.asarray(coords, dtype=float), np.asarray(pops, dtype=float), labels


def _gravity_matrix(coords: np.ndarray, population: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    flow = np.outer(population, population) / (d**2 + 1.0)
    np.fill_diagonal(flow, 0.0)
    return flow / flow.max()


def _simulate_arrival(flow: np.ndarray, seed_region: int, rng: np.random.Generator,
                      start_day: float, n_days: int) -> np.ndarray:
    R = flow.shape[0]
    arrival = np.full(R, np.inf)
    arrival[seed_region] = start_day
    for _ in range(R - 1):
        best = (np.inf, -1)
        for src in range(R):
            if not np.isfinite(arrival[src]):
                continue
            for dst in range(R):
                if np.isfinite(arrival[dst]) or flow[src, dst] <= 0:
                    continue
                delay = 6.0 / (flow[src, dst] + 0.02) * rng.uniform(0.7, 1.3)
                t = arrival[src] + delay
                if t < best[0]:
                    best = (t, dst)
        if best[1] < 0:
            break
        arrival[best[1]] = best[0]
    return np.clip(arrival, 0, n_days - 1)


def generate_synthetic_panel(rng: np.random.Generator) -> Panel:
    R, L, T = C.N_REGIONS, C.N_LINEAGES, C.N_DAYS
    days = np.arange(T, dtype=np.float64)

    coords = rng.uniform(0.0, 100.0, size=(R, 2))
    population = rng.uniform(0.3, 2.5, size=R)
    flow = _gravity_matrix(coords, population)

    true_fitness = np.concatenate([[0.0], np.sort(rng.uniform(0.008, 0.075, size=L - 1))])

    arrival = np.zeros((R, L))
    arrival[:, 0] = 0.0
    for li in range(1, L):
        seed = int(rng.integers(0, R))
        start = rng.uniform(0.05, 0.55) * T
        arrival[:, li] = _simulate_arrival(flow, seed, rng, start, T)

    z = np.zeros((R, L, T))
    for ri in range(R):
        for li in range(L):
            t0 = arrival[ri, li]
            base = rng.normal(-6.5, 0.8) if li else 0.0
            growth = true_fitness[li] * (days - t0)
            active = days >= t0
            wander = np.cumsum(rng.normal(0.0, 0.012, size=T))
            z[ri, li] = np.where(active, base + growth + wander, -30.0)

    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)

    weekly = 1.0 + 0.35 * np.sin(2 * np.pi * days / 7.0 - 1.2)
    trend = 1.0 + 0.6 * np.sin(2 * np.pi * days / 365.0)
    depth = np.zeros((R, T))
    for ri in range(R):
        mean = 55.0 * population[ri] * weekly * trend
        depth[ri] = rng.poisson(np.clip(mean * rng.gamma(6.0, 1 / 6.0, size=T), 1.0, None))

    counts = np.zeros((R, L, T))
    for ri in range(R):
        for t in range(T):
            counts[ri, :, t] = rng.multinomial(int(depth[ri, t]), p[ri, :, t])

    log.info(
        "Data source: %s (R=%d, L=%d, T=%d; true s_i known)",
        C.DATA_SOURCE_SYNTHETIC, R, L, T,
    )
    return _finalise_panel(
        counts, counts.sum(axis=1), REGION_NAMES[:R],
        [f"L{i}" for i in range(L)], rng,
        source=C.DATA_SOURCE_SYNTHETIC,
        true_fitness=true_fitness, coords=coords, population=population, arrival=arrival,
    )


def _finalise_panel(counts, depth, region_names, lineage_names, rng, *, source,
                    true_fitness=None, coords=None, population=None, arrival=None) -> Panel:
    counts = np.asarray(counts, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    keep = np.flatnonzero(np.median(depth, axis=1) >= C.MIN_DAILY_DEPTH)
    if keep.size >= 4 and keep.size < depth.shape[0]:
        dropped = [region_names[i] for i in range(depth.shape[0]) if i not in set(keep.tolist())]
        log.info("Dropped regions with depth < %d/day: %s",
                 C.MIN_DAILY_DEPTH, ", ".join(dropped))
        counts, depth = counts[keep], depth[keep]
        region_names = [region_names[i] for i in keep]
        if coords is not None:
            coords = np.asarray(coords)[keep]
        if population is not None:
            population = np.asarray(population)[keep]
        if arrival is not None:
            arrival = np.asarray(arrival)[keep]

    R, L, T = counts.shape
    days = np.arange(T, dtype=np.float64)
    n = depth[:, None, :]

    gap = depth <= 0
    if gap.any():
        filled = counts.copy()
        for ri in range(R):
            last = None
            for t in range(T):
                if gap[ri, t]:
                    if last is not None:
                        filled[ri, :, t] = counts[ri, :, last]
                else:
                    last = t
        share = np.where(
            gap[:, None, :], filled / np.clip(filled.sum(axis=1, keepdims=True), 1.0, None), 0.0)
        counts_for_f = np.where(gap[:, None, :], share * 30.0, counts)
        n_for_f = np.where(gap[:, None, :], 30.0, n)
        log.info("Days without sequences: %d of %d (frequency carried forward)",
                 int(gap.sum()), gap.size)
    else:
        counts_for_f, n_for_f = counts, n

    f = (counts_for_f + 0.5) / (n_for_f + 1.0)
    f = np.clip(f, 1e-6, 1 - 1e-6)
    var_f = f * (1 - f) / (n + 2.0)
    z = np.log(f / (1 - f))

    dz = np.gradient(z, axis=2)

    n_feat = np.repeat(depth[:, None, :], L, axis=1)

    p_lin = counts / np.clip(depth[:, None, :], 1.0, None)
    p_safe = np.clip(p_lin, 1e-12, 1.0)
    H = -(p_safe * np.log2(p_safe)).sum(axis=1)
    H_feat = np.repeat(H[:, None, :], L, axis=1)

    pi = 0.6 * H_feat + rng.normal(0.0, 0.12, size=(R, L, T))
    pi = np.clip(pi, 0.0, C.FEATURES[C.FEATURE_INDEX["pi"]].hi)

    if true_fitness is None:
        true_fitness = np.array(
            [np.polyfit(days, z[:, li, :].mean(axis=0), 1)[0] for li in range(L)]
        )
    s_norm = (true_fitness - true_fitness.min()) / (np.ptp(true_fitness) + C.EPS)

    def per_lineage(base: float, spread: float, corr: float, lo: float, hi: float) -> np.ndarray:
        core = base + spread * (corr * s_norm + (1 - corr) * rng.uniform(0, 1, size=L))
        arr = core[None, :, None] + rng.normal(0.0, 0.06 * spread, size=(R, L, T))
        return np.clip(arr, lo, hi)

    omega = per_lineage(0.6, 2.2, 0.75, 0.0, 4.0)
    ddG = per_lineage(-1.5, 3.0, 0.45, -4.0, 4.0)
    E = per_lineage(0.15, 0.7, 0.80, 0.0, 1.0)
    eps = per_lineage(-0.4, 0.9, 0.30, -1.0, 1.0)

    seasonal = np.sin(2 * np.pi * days / 365.0 - 0.7)
    Reff = 1.05 + 0.35 * seasonal[None, :] + rng.normal(0.0, 0.08, size=(R, T))
    Reff = np.clip(np.repeat(Reff[:, None, :], L, axis=1), 0.0, 4.0)

    phase = rng.uniform(0, 2 * np.pi, size=R)
    mob = 1.0 + 0.25 * np.sin(2 * np.pi * days[None, :] / 365.0 + phase[:, None])
    mob = mob + rng.normal(0.0, 0.05, size=(R, T))
    m_feat = np.clip(np.repeat(mob[:, None, :], L, axis=1), 0.0, 2.0)

    features = np.stack([f, z, dz, n_feat, H_feat, pi, omega, ddG, E, eps, Reff, m_feat], axis=-1)
    assert features.shape == (R, L, T, C.N_FEATURES), features.shape

    ctrl_rng = np.random.default_rng(C.SEED + 777)
    control = 12.0 + 18.0 * np.sin(2 * np.pi * days / 365.0 - 1.9)[None, :]
    control = control + ctrl_rng.normal(0.0, 3.0, size=(R, T))

    if coords is None:
        coords = rng.uniform(0.0, 100.0, size=(R, 2))
    if population is None:
        population = rng.uniform(0.3, 2.5, size=R)
    if arrival is None:
        arrival = np.zeros((R, L))
        for ri in range(R):
            for li in range(L):
                above = np.flatnonzero(counts[ri, li] > 0)
                arrival[ri, li] = above[0] if above.size else T - 1

    del var_f

    return Panel(
        counts=counts, depth=depth, features=features, control=control,
        region_names=list(region_names), lineage_names=list(lineage_names),
        coords=coords, population=population, true_fitness=np.asarray(true_fitness),
        arrival_day=arrival, source=source,
    )


def load_panel(cfg: C.RunConfig) -> Panel:
    rng = np.random.default_rng(cfg.seed)
    cache = C.DATA_CACHE / (
        f"nextstrain_{C.NEXTSTRAIN_MAX_ROWS}_"
        f"{C.N_REGIONS}x{C.N_LINEAGES}x{C.N_DAYS}.npz"
    )

    if cfg.use_real_data:
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            log.info("Data source: %s (from cache %s)", C.DATA_SOURCE_REAL, cache.name)
            names = [str(n).split(" (coords")[0] for n in z["regions"]]
            coords, population, labels = _geo_for(names)
            return _finalise_panel(
                z["counts"], z["depth"], labels, list(z["lineages"]), rng,
                source=C.DATA_SOURCE_REAL, true_fitness=None,
                coords=coords, population=population,
            )
        panel = _try_load_nextstrain(rng)
        if panel is not None:
            C.DATA_CACHE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache, counts=panel.counts, depth=panel.depth,
                regions=np.array([n.split(" (coords")[0] for n in panel.region_names],
                                 dtype=object),
                lineages=np.array(panel.lineage_names, dtype=object),
                coords=panel.coords, population=panel.population,
            )
            log.info("Aggregation cached: %s", cache.name)
            return panel
        log.info("Falling back to synthetic panel — pipeline must run end-to-end")
    return generate_synthetic_panel(rng)


class RobustScaler:
    def __init__(self, mode: str = "robust") -> None:
        self.mode = mode
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.raw_idx = [C.FEATURE_INDEX[k] for k in C.RAW_SCALE_FEATURES]

    def fit(self, x: np.ndarray) -> RobustScaler:
        flat = x.reshape(-1, x.shape[-1])
        if self.mode == "robust":
            q1, q2, q3 = np.percentile(flat, [25, 50, 75], axis=0)
            self.center, self.scale = q2, (q3 - q1) + C.EPS
        elif self.mode == "minmax":
            lo, hi = flat.min(axis=0), flat.max(axis=0)
            self.center, self.scale = lo, (hi - lo) + C.EPS
        else:
            raise ValueError(f"unknown scaler mode: {self.mode}")
        for i in self.raw_idx:
            self.center[i], self.scale[i] = 0.0, 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center is None:
            raise RuntimeError("scaler not fitted")
        return (x - self.center) / self.scale


WINDOW_FIELDS: tuple[str, ...] = (
    "x", "y", "y_abs", "anchor", "depth", "event", "region", "lineage", "t_end", "y_hist",
)


@dataclass
class WindowSet:
    x: np.ndarray
    y: np.ndarray
    y_abs: np.ndarray
    anchor: np.ndarray
    depth: np.ndarray
    event: np.ndarray
    region: np.ndarray
    lineage: np.ndarray
    t_end: np.ndarray
    y_hist: np.ndarray

    def __len__(self) -> int:
        return self.x.shape[0]

    def subset(self, idx: np.ndarray) -> WindowSet:
        return WindowSet(*[getattr(self, f)[idx] for f in WINDOW_FIELDS])


def build_windows(panel: Panel, t_lo: int, t_hi: int, stride: int = 1,
                  max_windows: int | None = None,
                  rng: np.random.Generator | None = None,
                  keep_series: bool = False) -> WindowSet:
    R, L, T, _ = panel.features.shape
    z_idx = C.FEATURE_INDEX["z"]
    span = C.T_IN + C.HORIZON

    starts = np.arange(t_lo, min(t_hi, T) - span + 1, stride)
    if starts.size == 0:
        raise ValueError(f"interval [{t_lo}, {t_hi}) shorter than {span}-day window")

    xs, ys, ds, evs, rs, ls, tes, yh = [], [], [], [], [], [], [], []
    dom_logit = math.log(C.DOMINANCE_THRESHOLD / (1 - C.DOMINANCE_THRESHOLD))

    for ri in range(R):
        for li in range(L):
            feat = panel.features[ri, li]
            for t in starts:
                win = feat[t : t + C.T_IN]
                tgt = feat[t + C.T_IN : t + span, z_idx]
                if win[:, z_idx].std() < 1e-6:
                    continue
                xs.append(win)
                ys.append(tgt)
                ds.append(panel.depth[ri, t + C.T_IN - 1])
                evs.append(float(tgt.max() >= dom_logit))
                rs.append(ri)
                ls.append(li)
                tes.append(t + C.T_IN - 1)
                yh.append(win[:, z_idx])

    y_abs = np.asarray(ys, dtype=np.float32)
    y_hist_arr = np.asarray(yh, dtype=np.float32)
    anchor = y_hist_arr[:, -1].copy()
    ws = WindowSet(
        x=np.asarray(xs, dtype=np.float32),
        y=(y_abs - anchor[:, None]).astype(np.float32),
        y_abs=y_abs, anchor=anchor,
        depth=np.asarray(ds, dtype=np.float32), event=np.asarray(evs, dtype=np.float32),
        region=np.asarray(rs, dtype=np.int64), lineage=np.asarray(ls, dtype=np.int64),
        t_end=np.asarray(tes, dtype=np.int64), y_hist=y_hist_arr,
    )
    if max_windows is not None and len(ws) > max_windows:
        gen = rng or np.random.default_rng(C.SEED)
        if keep_series:
            pairs = np.unique(np.stack([ws.region, ws.lineage], axis=1), axis=0)
            gen.shuffle(pairs)
            chosen, total = [], 0
            for r, li in pairs:
                idx = np.flatnonzero((ws.region == r) & (ws.lineage == li))
                if total and total + idx.size > max_windows:
                    continue
                chosen.append(idx)
                total += idx.size
                if total >= max_windows:
                    break
            keep = np.sort(np.concatenate(chosen))
        else:
            keep = np.sort(gen.choice(len(ws), size=max_windows, replace=False))
        ws = ws.subset(keep)
    return ws


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: int
    test_start: int
    test_end: int


def rolling_origin_folds(n_days: int, k: int = C.N_FOLDS, gap: int = C.GAP_DAYS) -> list[Fold]:
    span = C.T_IN + C.HORIZON
    test_len = span + 60
    first_train = int(n_days * 0.45)
    last_train = n_days - gap - test_len
    if last_train <= first_train:
        raise ValueError(f"panel of {n_days} days is insufficient for {k} folds with gap {gap}")
    ends = np.linspace(first_train, last_train, k).astype(int)
    return [
        Fold(index=i, train_end=int(e), test_start=int(e + gap), test_end=int(e + gap + test_len))
        for i, e in enumerate(ends)
    ]


class Augmenter:
    def __init__(self, rng: np.random.Generator, mode: str = "binomial") -> None:
        self.rng = rng
        self.mode = mode
        self.f_idx = C.FEATURE_INDEX["f"]
        self.z_idx = C.FEATURE_INDEX["z"]
        self.dz_idx = C.FEATURE_INDEX["dz"]
        self.n_idx = C.FEATURE_INDEX["n"]

    def binomial_resample(self, x: np.ndarray) -> np.ndarray:
        f = np.clip(x[..., self.f_idx], 1e-6, 1 - 1e-6)
        n = np.clip(x[..., self.n_idx], 1.0, None)
        k = self.rng.binomial(n.astype(np.int64), f)
        return self._rewrite_frequency(x, (k + 0.5) / (n + 1.0))

    def thin_depth(self, x: np.ndarray) -> np.ndarray:
        rho = self.rng.uniform(*C.AUG_THIN_RHO, size=(x.shape[0], 1))
        n = np.clip(x[..., self.n_idx], 1.0, None)
        n_new = np.floor(rho * n)
        n_new = np.clip(n_new, 1.0, None)
        f = np.clip(x[..., self.f_idx], 1e-6, 1 - 1e-6)
        k = self.rng.binomial(n_new.astype(np.int64), f)
        out = self._rewrite_frequency(x, (k + 0.5) / (n_new + 1.0))
        out[..., self.n_idx] = n_new
        return out

    def spline_warp(self, x: np.ndarray) -> np.ndarray:
        from scipy.interpolate import CubicSpline

        n, t, _ = x.shape
        knot_t = np.linspace(0, t - 1, C.AUG_SPLINE_KNOTS + 2)
        knot_v = self.rng.normal(1.0, C.AUG_SPLINE_SIGMA, size=(n, C.AUG_SPLINE_KNOTS + 2))
        grid = np.arange(t)
        out = x.copy()
        warp_idx = [i for i in range(x.shape[-1]) if i != self.n_idx]
        for i in range(n):
            curve = CubicSpline(knot_t, knot_v[i])(grid)[:, None]
            out[i, :, warp_idx] = (x[i, :, warp_idx].T * curve).T
        return out

    def mixup(self, x: np.ndarray, y: np.ndarray, ev: np.ndarray) -> tuple[np.ndarray, ...]:
        lam = self.rng.beta(
            C.AUG_MIXUP_ALPHA, C.AUG_MIXUP_ALPHA, size=(x.shape[0], 1, 1)).astype(np.float32)
        perm = self.rng.permutation(x.shape[0])
        xm = lam * x + (1 - lam) * x[perm]
        l2 = lam[:, :, 0]
        return xm, l2 * y + (1 - l2) * y[perm], (l2[:, 0] * ev + (1 - l2[:, 0]) * ev[perm])

    def gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        sigma = 0.05 * np.abs(x).mean(axis=(0, 1), keepdims=True)
        keep = np.ones(x.shape[-1], dtype=bool)
        keep[self.n_idx] = False
        out[
            ..., keep] = x[..., keep] + self.rng.normal(0, 1, size=x.shape)[..., keep] * sigma[...,
            keep]
        return out

    def _rewrite_frequency(self, x: np.ndarray, f_new: np.ndarray) -> np.ndarray:
        out = x.copy()
        f_new = np.clip(f_new, 1e-6, 1 - 1e-6)
        out[..., self.f_idx] = f_new
        z = np.log(f_new / (1 - f_new))
        out[..., self.z_idx] = z
        out[..., self.dz_idx] = np.gradient(z, axis=1)
        return out

    def __call__(self, x: np.ndarray, y: np.ndarray, ev: np.ndarray) -> tuple[np.ndarray, ...]:
        if self.mode == "gaussian":
            if self.rng.random() < C.AUG_PROB:
                x = self.gaussian_noise(x)
            return x, y, ev
        if self.rng.random() < C.AUG_PROB:
            x = self.binomial_resample(x)
        if self.rng.random() < C.AUG_PROB:
            x = self.thin_depth(x)
        if self.rng.random() < C.AUG_PROB:
            x = self.spline_warp(x)
        if self.rng.random() < C.AUG_PROB:
            x, y, ev = self.mixup(x, y, ev)
        return x.astype(np.float32), y.astype(np.float32), ev.astype(np.float32)


def make_masks(n: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((n, C.T_IN), dtype=bool)
    target = max(1, int(round(C.MASK_FRACTION * C.T_IN)))
    for i in range(n):
        covered = 0
        for _ in range(12):
            if covered >= target:
                break
            span = int(rng.integers(C.MASK_SPAN_MIN, C.MASK_SPAN_MAX + 1))
            start = int(rng.integers(0, C.T_IN - span + 1))
            mask[i, start : start + span] = True
            covered = int(mask[i].sum())
    return mask
