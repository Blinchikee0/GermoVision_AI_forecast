from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

MODEL_PATH = Path(__file__).parent / "geo_model.pkl"
EARTH_R_KM = 6371.0


@dataclass
class GeoTrainedBundle:
    arrival_regressor: object
    risk_classifier: object
    scaler: object
    feature_names: list[str]
    trained_n: int


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dla = la2 - la1
    dlo = lo2 - lo1
    h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(h))


def gravity_flow(pop_i: float, pop_j: float, km: float, alpha: float = 1.4) -> float:
    if km <= 0:
        return 0.0
    return (pop_i * pop_j) / ((km / 1000.0) ** alpha + 1.0)


def make_features(seed: dict, target: dict, r0: float, generation_time: float,
                  mutation_rate: float) -> list[float]:
    km = haversine_km(seed["lat"], seed["lng"], target["lat"], target["lng"])
    flow = gravity_flow(seed["pop"], target["pop"], km)
    lat_delta = abs(seed["lat"] - target["lat"])
    lng_delta = min(abs(seed["lng"] - target["lng"]), 360 - abs(seed["lng"] - target["lng"]))
    hemisphere_shift = 1.0 if (seed["lat"] * target["lat"] < 0) else 0.0
    return [
        km,
        math.log1p(flow),
        target["pop"],
        seed["pop"],
        lat_delta,
        lng_delta,
        hemisphere_shift,
        r0,
        generation_time,
        mutation_rate,
        r0 / max(generation_time, 1.0),
    ]


FEATURE_NAMES = [
    "distance_km", "log_flow", "target_pop_m", "seed_pop_m",
    "lat_delta", "lng_delta", "hemisphere_shift",
    "r0", "generation_time", "mutation_rate", "growth_rate_per_day",
]


def _synthesize_training_data(rng: np.random.Generator, n_samples: int = 4000,
                               cities: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cities is None:
        cities = [
            {"n": f"C{i}", "lat": rng.uniform(-60, 65), "lng": rng.uniform(-180, 180),
             "pop": max(0.3, rng.exponential(6.0))} for i in range(60)
        ]
    X, y_arrival, y_risk = [], [], []
    for _ in range(n_samples):
        seed = cities[rng.integers(len(cities))]
        target = cities[rng.integers(len(cities))]
        if seed is target:
            continue
        r0 = rng.uniform(1.2, 5.5)
        gt = rng.uniform(2.5, 10.0)
        mr = rng.uniform(0.005, 0.12)
        feats = make_features(seed, target, r0, gt, mr)
        km = feats[0]
        flow = math.expm1(feats[1])
        base_delay = 4.5 + km / 400.0 - math.log1p(flow) * 3.0
        base_delay = max(1.0, base_delay / max((r0 - 0.8) / gt, 0.05))
        base_delay += rng.normal(0, base_delay * 0.15)
        base_delay = max(0.5, base_delay)
        arrived_within_60 = 1 if base_delay < 60 else 0
        X.append(feats)
        y_arrival.append(base_delay)
        y_risk.append(arrived_within_60)
    return np.array(X), np.array(y_arrival), np.array(y_risk)


def train_or_load(cities: list[dict] | None = None, force: bool = False) -> GeoTrainedBundle | None:
    if not SKLEARN_OK:
        return None
    if MODEL_PATH.exists() and not force:
        try:
            with MODEL_PATH.open("rb") as f:
                b = pickle.load(f)
            if isinstance(b, GeoTrainedBundle):
                return b
        except Exception:
            pass
    rng = np.random.default_rng(2026)
    X, y_arr, y_risk = _synthesize_training_data(rng, cities=cities)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    reg = GradientBoostingRegressor(
        n_estimators=180, max_depth=4, learning_rate=0.06, random_state=2026,
        subsample=0.8, min_samples_leaf=8,
    ).fit(Xs, y_arr)
    clf = RandomForestClassifier(
        n_estimators=160, max_depth=8, min_samples_leaf=4, n_jobs=-1, random_state=2026,
    ).fit(Xs, y_risk)
    bundle = GeoTrainedBundle(
        arrival_regressor=reg, risk_classifier=clf, scaler=scaler,
        feature_names=FEATURE_NAMES, trained_n=len(X),
    )
    try:
        with MODEL_PATH.open("wb") as f:
            pickle.dump(bundle, f)
    except Exception:
        pass
    return bundle


def predict_arrival(bundle: GeoTrainedBundle, seed: dict, targets: list[dict],
                     r0: float, generation_time: float, mutation_rate: float) -> dict:
    X = np.array([make_features(seed, t, r0, generation_time, mutation_rate) for t in targets])
    Xs = bundle.scaler.transform(X)
    arrival = bundle.arrival_regressor.predict(Xs)
    risk = bundle.risk_classifier.predict_proba(Xs)[:, 1]
    return {"arrival_days": arrival.tolist(), "risk_60d": risk.tolist()}


def feature_importance(bundle: GeoTrainedBundle) -> list[dict]:
    reg = bundle.arrival_regressor
    clf = bundle.risk_classifier
    reg_imp = getattr(reg, "feature_importances_", np.zeros(len(bundle.feature_names)))
    clf_imp = getattr(clf, "feature_importances_", np.zeros(len(bundle.feature_names)))
    return [
        {"feature": n, "arrival_importance": float(reg_imp[i]),
         "risk_importance": float(clf_imp[i])}
        for i, n in enumerate(bundle.feature_names)
    ]


def derive_from_analysis(analysis: dict, base_r0: float, base_mut_rate: float) -> dict:
    if not analysis:
        return {
            "r0": base_r0, "mutation_rate": base_mut_rate,
            "r0_delta": 0.0, "mut_rate_delta": 0.0,
            "escape_pressure": 0.0, "high_impact_hits": 0,
            "hotspot_hits": 0, "novelty": 0.0,
            "notes": ["no genome supplied — using slider defaults"],
        }
    muts = analysis.get("mutations", [])
    hotspots = set(analysis.get("reference", {}).get("hotspots", []))
    novelty = float(analysis.get("novelty_score", 0.0))
    high_impact = [m for m in muts if m.get("impact", 0.0) >= 0.6]
    hotspot_hits = sum(1 for m in muts if m.get("hotspot_distance", 999) <= 5)
    escape_pressure = 0.0
    for m in muts[:20]:
        impact = m.get("impact", 0.0)
        near = 1.0 if m.get("hotspot_distance", 999) <= 5 else 0.4
        escape_pressure += impact * near
    escape_pressure = min(3.0, escape_pressure)

    r0_delta = round(0.08 * len(high_impact) + 0.05 * hotspot_hits + 0.12 * novelty, 3)
    r0_delta = max(-0.5, min(2.5, r0_delta))
    mut_rate_delta = round(0.005 * escape_pressure + 0.03 * novelty, 4)

    r0 = round(base_r0 + r0_delta, 2)
    mr = round(base_mut_rate + mut_rate_delta, 4)

    notes = []
    if high_impact:
        notes.append(f"{len(high_impact)} high-impact mutation(s) raise R₀ by {0.08 * len(high_impact):.2f}")
    if hotspot_hits:
        notes.append(f"{hotspot_hits} escape-hotspot hit(s) raise R₀ by {0.05 * hotspot_hits:.2f}")
    if novelty >= 0.3:
        notes.append(f"novelty index {novelty:.2f} raises R₀ by {0.12 * novelty:.2f} and mut. rate by {0.03 * novelty:.4f}")
    if not notes:
        notes.append("mutation profile is benign — simulation uses baseline sliders")

    return {
        "r0": r0, "mutation_rate": mr,
        "r0_delta": r0_delta, "mut_rate_delta": mut_rate_delta,
        "escape_pressure": round(escape_pressure, 3),
        "high_impact_hits": len(high_impact),
        "hotspot_hits": hotspot_hits,
        "novelty": novelty,
        "notes": notes,
        "base_r0": base_r0, "base_mut_rate": base_mut_rate,
    }


def great_circle_arc(seed: dict, target: dict, n: int = 24) -> list[list[float]]:
    la1, lo1 = math.radians(seed["lat"]), math.radians(seed["lng"])
    la2, lo2 = math.radians(target["lat"]), math.radians(target["lng"])
    d = 2 * math.asin(math.sqrt(
        math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    ))
    if d < 1e-9:
        return [[seed["lat"], seed["lng"]]]
    pts = []
    for i in range(n + 1):
        f = i / n
        A = math.sin((1 - f) * d) / math.sin(d)
        B = math.sin(f * d) / math.sin(d)
        x = A * math.cos(la1) * math.cos(lo1) + B * math.cos(la2) * math.cos(lo2)
        y = A * math.cos(la1) * math.sin(lo1) + B * math.cos(la2) * math.sin(lo2)
        z = A * math.sin(la1) + B * math.sin(la2)
        lat = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
        lng = math.degrees(math.atan2(y, x))
        pts.append([lat, lng])
    return pts
