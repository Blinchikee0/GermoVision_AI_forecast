"""Сколько именно метрик приписывает себе утечка версии 1.0.

Скрипт ставит численный эксперимент, соответствующий последней строке
таблицы абляций (§ 5.10 мастер-документа). Одна и та же модель обучается
на одних и тех же данных двумя способами:

    Протокол A (версия 1.0): аугментация → случайное разделение
    Протокол B (версия 2.0): разделение по кластерам → аугментация train

Разница между полученными метриками и есть та величина, которую утечка
добавляет к результату «бесплатно». Она не связана с качеством модели и
на новых данных не воспроизводится.

Запуск:  python examples/leakage_demo.py
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from germovision.core.metrics import evaluate_binary
from germovision.core.splitting import (
    LeakageGuard,
    augment_train_only,
    cluster_split,
)
from germovision.core.types import LeakageError, Split

RNG_SEED = 20260904
N_SAMPLES = 600
N_FEATURES = 12
N_AUGMENTED = 400
NOISE_SCALE = 0.01


def make_dataset(seed: int = RNG_SEED):
    """Синтетические данные с редким положительным классом (~12 %)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N_SAMPLES, N_FEATURES))
    logit = 1.2 * x[:, 0] - 0.8 * x[:, 1] + 0.5 * x[:, 2]
    y = (logit + rng.normal(scale=1.0, size=N_SAMPLES) > 2.0).astype(int)
    return x, y


def augment(x_pos_source, y_pos_source, rng, n: int = N_AUGMENTED):
    """Аугментация версии 1.0: дублирование редкого класса + гауссов шум."""
    pos = np.flatnonzero(y_pos_source == 1)
    if pos.size == 0:
        return np.empty((0, x_pos_source.shape[1])), np.empty(0, dtype=int)
    dup = rng.choice(pos, size=n, replace=True)
    noise = rng.normal(scale=NOISE_SCALE, size=(n, x_pos_source.shape[1]))
    return x_pos_source[dup] + noise, y_pos_source[dup]


def fit_and_score(x_train, y_train, x_test, y_test, label):
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, random_state=0
    )
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    return evaluate_binary(y_test, prob, label=label, threshold=0.5, n_boot=300)


def protocol_a_broken(x, y):
    """Как делала версия 1.0: аугментация до разделения выборки."""
    rng = np.random.default_rng(RNG_SEED)
    x_extra, y_extra = augment(x, y, rng)
    x_all = np.concatenate([x, x_extra])
    y_all = np.concatenate([y, y_extra])

    order = rng.permutation(x_all.shape[0])
    cut = int(0.75 * order.size)
    split = Split(train=order[:cut], test=order[cut:], strategy="v1_random")

    report = fit_and_score(
        x_all[split.train], y_all[split.train],
        x_all[split.test], y_all[split.test],
        "Протокол A (с утечкой)",
    )
    return split, x_all, report


def protocol_b_correct(x, y):
    """Как требует версия 2.0: разделение первым, аугментация только в train."""
    clusters = np.arange(x.shape[0])  # каждый объект независим
    split = cluster_split(clusters, test_size=0.25, seed=RNG_SEED)

    x_train, y_train = augment_train_only(split, x, y, augment, seed=RNG_SEED)
    report = fit_and_score(
        x_train, y_train, x[split.test], y[split.test], "Протокол B (корректный)"
    )
    return split, report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    x, y = make_dataset()
    print(f"Данные: {x.shape[0]} объектов, {x.shape[1]} признаков, "
          f"положительных {y.mean():.1%}\n")

    split_a, x_all, rep_a = protocol_a_broken(x, y)
    split_b, rep_b = protocol_b_correct(x, y)

    print("=" * 74)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 74)
    print(f"{'Метрика':<16} {'A (с утечкой)':<24} {'B (корректный)':<24} {'Δ':>8}")
    print("-" * 74)
    for name in ("roc_auc", "pr_auc", "sensitivity", "specificity"):
        a, b = getattr(rep_a, name), getattr(rep_b, name)
        print(f"{name:<16} {str(a):<24} {str(b):<24} {a.value - b.value:>+8.3f}")
    print("-" * 74)

    print(
        "\nВся разница получена без единого улучшения модели — только за счёт\n"
        "порядка операций. Обратите внимание, что сильнее всего расходятся\n"
        "PR-AUC и чувствительность: именно они отвечают за редкий класс,\n"
        "который и дублировался при аугментации. ROC-AUC маскирует проблему\n"
        "слабее прочих — ещё одна причина не отчитываться им в одиночку.\n"
        "\nИменно поэтому метрики версии 1.0 нельзя считать свидетельством\n"
        "качества модели: они измеряют утечку, а не предсказательную силу.\n"
    )

    print("=" * 74)
    print("ЧТО СКАЖЕТ ЗАЩИТА")
    print("=" * 74)

    print("\nПротокол A:")
    try:
        LeakageGuard(x=x_all).run(split_a)
        print("  защита ничего не нашла — этого быть не должно")
    except LeakageError as exc:
        print(f"  LeakageError: {exc}")

    print("\nПротокол B:")
    print("  " + str(LeakageGuard(x=x).run(split_b)).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
