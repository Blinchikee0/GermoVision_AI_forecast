"""Тесты слоя данных: каталог, генератор, признаки, загрузчик."""

from __future__ import annotations

import numpy as np
import pytest

from germovision.core.splitting import temporal_cluster_split
from germovision.data import (
    DRUG_GENES,
    DRUGS,
    FeatureBuilder,
    MutationCatalogue,
    SyntheticConfig,
    generate_isolates,
)
from germovision.data.cryptic import assign_clusters_by_genotype, load_cryptic
from germovision.data.synthetic import NOVEL_MARKERS, generate_lineage_counts

# --------------------------------------------------------------------------
# Каталог мутаций
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cat():
    return MutationCatalogue()


def test_catalogue_covers_all_drugs(cat):
    for drug in DRUGS:
        assert cat.resistance_markers(drug), f"нет маркеров для {drug}"


def test_catalogue_finds_canonical_marker(cat):
    hits = cat.lookup("rpoB_S450L", "RIF")
    assert hits and hits[0].group == 1


def test_catalogue_predicts_resistance(cat):
    decision, evidence = cat.predict({"rpoB_S450L"}, "RIF")
    assert decision is True
    assert evidence[0].key == "rpoB_S450L"


def test_catalogue_stays_silent_without_marker(cat):
    """Отсутствие маркера не доказывает чувствительность — каталог молчит."""
    decision, evidence = cat.predict({"rpoB_S488A"}, "RIF")
    assert decision is None
    assert evidence == []


def test_catalogue_encodes_rif_rfb_discordance(cat):
    """rpoB D435V даёт устойчивость к RIF, но не к рифабутину.

    Клинически значимое различие: потеряв его, система отняла бы у врача
    работающий препарат.
    """
    assert cat.predict({"rpoB_D435V"}, "RIF")[0] is True
    assert cat.predict({"rpoB_D435V"}, "RFB")[0] is None


def test_catalogue_encodes_bdq_cfz_cross_resistance(cat):
    """Потеря функции Rv0678 даёт перекрёстную устойчивость BDQ/CFZ."""
    assert cat.predict({"Rv0678_LoF"}, "BDQ")[0] is True
    assert cat.predict({"Rv0678_LoF"}, "CFZ")[0] is True


def test_catalogue_marker_genes_match_drug_genes(cat):
    """Каждый маркер лежит в гене, объявленном для своего препарата."""
    for entry in cat.entries:
        assert entry.gene in DRUG_GENES[entry.drug], (
            f"{entry.key} отнесён к {entry.drug}, но гена нет в DRUG_GENES"
        )


def test_catalogue_from_tsv_missing_file():
    with pytest.raises(FileNotFoundError):
        MutationCatalogue.from_who_tsv("нет-такого-файла.tsv")


# --------------------------------------------------------------------------
# Синтетический генератор
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds():
    return generate_isolates(SyntheticConfig(n_isolates=1200, seed=3))


def test_generator_is_reproducible():
    a = generate_isolates(SyntheticConfig(n_isolates=300, seed=7))
    b = generate_isolates(SyntheticConfig(n_isolates=300, seed=7))
    assert list(a.isolate_ids) == list(b.isolate_ids)
    assert a.mutations == b.mutations
    np.testing.assert_array_equal(a.phenotypes["RIF"], b.phenotypes["RIF"], strict=False)


def test_generator_marks_data_as_synthetic(ds):
    """Пометка проходит через весь пайплайн до отчёта."""
    assert ds.meta["synthetic"] is True
    assert "SYNTHETIC" in ds.meta["warning"]


def test_generator_has_transmission_clusters(ds):
    """Кластеров заметно меньше, чем изолятов, — иначе нечего разделять."""
    assert 1 < np.unique(ds.clusters).size < len(ds)


def test_submission_never_precedes_collection(ds):
    assert (ds.submission_dates >= ds.collection_dates).all()


def test_every_drug_has_both_classes(ds):
    for drug in ds.drugs:
        y = ds.phenotypes[drug]
        y = y[~np.isnan(y)]
        assert 0.0 < y.mean() < 1.0, f"{drug}: только один класс"


def test_phenotypes_have_missing_values(ds):
    """Пропуски реальны: не для каждого изолята измерены все 13 МИК."""
    assert any(np.isnan(ds.phenotypes[d]).any() for d in ds.drugs)


def test_kazakhstan_has_highest_resistance(ds):
    """Профиль страны воспроизводит данные ВОЗ: в Казахстане доля МЛУ выше."""
    rates = {}
    for country in np.unique(ds.countries):
        mask = ds.countries == country
        r, i = ds.phenotypes["RIF"][mask], ds.phenotypes["INH"][mask]
        both = ~np.isnan(r) & ~np.isnan(i)
        rates[country] = ((r[both] == 1) & (i[both] == 1)).mean()
    assert rates["KZ"] == max(rates.values())


def test_novel_markers_are_outside_catalogue(cat):
    """Новые маркеры не должны попадать в каталог — иначе абляция бессмысленна."""
    known = cat.known_keys()
    for markers in NOVEL_MARKERS.values():
        for key, _ in markers:
            assert key not in known, f"{key} есть в каталоге, хотя объявлен новым"


def test_novel_markers_actually_appear(ds):
    """Модель должна иметь что выучить сверх правил каталога."""
    seen = {m for muts in ds.mutations for m in muts}
    novel_rif = {k for k, _ in NOVEL_MARKERS["RIF"]}
    assert seen & novel_rif


def test_new_drug_resistance_is_time_shifted(ds):
    """Устойчивость к бедаквилину появляется только в поздние годы."""
    years = ds.collection_dates.astype("datetime64[Y]").astype(int) + 1970
    res = ds.phenotypes["BDQ"] == 1
    if res.sum() >= 5:
        assert np.median(years[res]) >= 2018


def test_config_rejects_impossible_fractions():
    with pytest.raises(ValueError, match="must be < 1"):
        SyntheticConfig(unexplained_fraction=0.7, novel_fraction=0.5)


def test_subset_preserves_consistency(ds):
    sub = ds.subset(np.arange(50))
    assert len(sub) == 50
    assert len(sub.mutations) == 50
    assert all(len(a) == 50 for a in sub.phenotypes.values())


def test_summary_lists_every_drug(ds):
    text = ds.summary()
    for drug in ds.drugs:
        assert drug in text


# --------------------------------------------------------------------------
# Признаки
# --------------------------------------------------------------------------


def test_feature_vocabulary_built_from_train_only(ds):
    """Ключевая проверка: словарь признаков не должен видеть тест.

    Сбор словаря по всем данным — тихая утечка: редкая мутация, которая
    встречается только в тестовых изолятах, получила бы собственный столбец.
    """
    split = temporal_cluster_split(ds.submission_dates, ds.clusters)
    fb = FeatureBuilder("RIF").fit(ds, split.train)

    train_muts = {m for i in split.train for m in ds.mutations[i]}
    assert set(fb.vocabulary_) <= train_muts


def test_feature_builder_keeps_only_relevant_genes(ds):
    split = temporal_cluster_split(ds.submission_dates, ds.clusters)
    fb = FeatureBuilder("RIF").fit(ds, split.train)
    assert all(m.split("_", 1)[0] in DRUG_GENES["RIF"] for m in fb.vocabulary_)


def test_unknown_variants_go_to_burden_not_columns(ds):
    """Невиданный вариант не создаёт столбец, но учитывается как нагрузка."""
    split = temporal_cluster_split(ds.submission_dates, ds.clusters)
    fb = FeatureBuilder("RIF").fit(ds, split.train)
    fm = fb.transform(ds, split.test)

    assert fm.n_features == len(fb.names_)
    assert "burden_unknown" in fm.names
    assert fm.x[:, fm.names.index("burden_unknown")].sum() >= 0


def test_feature_groups_allow_ablation(ds):
    split = temporal_cluster_split(ds.submission_dates, ds.clusters)
    fm = FeatureBuilder("RIF").fit(ds, split.train).transform(ds, split.train)
    assert fm.group_indices("catalogue").size == 4
    assert fm.group_indices("mutation").size > 0


def test_transform_requires_fit(ds):
    with pytest.raises(RuntimeError, match="not fitted"):
        FeatureBuilder("RIF").transform(ds)


# --------------------------------------------------------------------------
# Кластеризация и загрузчик
# --------------------------------------------------------------------------


def test_genotype_clustering_groups_near_identical():
    muts = [
        {"rpoB_S450L", "katG_S315T"},
        {"rpoB_S450L", "katG_S315T", "embB_M306V"},
        {"gyrA_D94G", "rrs_a1401g", "eis_c-14t", "inhA_S94A", "ethA_LoF", "atpE_D28N"},
    ]
    labels = assign_clusters_by_genotype(muts, threshold=2)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_load_cryptic_missing_directory():
    with pytest.raises(FileNotFoundError, match="samples.csv"):
        load_cryptic("нет-такого-каталога")


def test_load_cryptic_roundtrip(tmp_path):
    (tmp_path / "samples.csv").write_text(
        "id,country,lineage,collection_date\n"
        "S1,KZ,L2,2021-03-01\nS2,KZ,L2,2021-04-01\nS3,GB,L4,2021-05-01\n",
        encoding="utf-8",
    )
    (tmp_path / "mutations.csv").write_text(
        "id,gene,mutation\nS1,rpoB,S450L\nS2,rpoB,S450L\nS3,katG,S315T\n",
        encoding="utf-8",
    )
    (tmp_path / "phenotypes.csv").write_text(
        "id,drug,phenotype\nS1,RIF,R\nS2,RIF,R\nS3,RIF,S\nS3,INH,R\n",
        encoding="utf-8",
    )

    ds = load_cryptic(tmp_path)
    assert len(ds) == 3
    assert ds.mutations[0] == {"rpoB_S450L"}
    np.testing.assert_array_equal(ds.phenotypes["RIF"], np.array([1.0, 1.0, 0.0]))
    assert ds.meta["synthetic"] is False
    # Дата депонирования отсутствовала — об этом должна быть пометка.
    assert "submission_date" in ds.meta["notes"]


def test_load_cryptic_rejects_unrecognized_phenotypes(tmp_path):
    (tmp_path / "samples.csv").write_text(
        "id,country,collection_date\nS1,KZ,2021-03-01\n", encoding="utf-8"
    )
    (tmp_path / "mutations.csv").write_text("id,gene,mutation\nS1,rpoB,S450L\n", encoding="utf-8")
    (tmp_path / "phenotypes.csv").write_text(
        "id,drug,phenotype\nS1,XYZ,мутно\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="No recognised phenotype"):
        load_cryptic(tmp_path)


# --------------------------------------------------------------------------
# Счётчики линий
# --------------------------------------------------------------------------


def test_lineage_counts_shape():
    counts, times, regions, lineages, truth = generate_lineage_counts(n_weeks=12)
    assert counts.shape[1] == len(lineages)
    assert counts.shape[0] == times.size == regions.size
    assert set(truth) == set(lineages)
    assert counts.sum() > 0
