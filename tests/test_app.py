"""Тесты локального веб-приложения.

Проверяется контракт HTTP-слоя: что приложение поднимается без моделей,
что несколько файлов обрабатываются независимо, и что сбой одного файла
не отменяет остальные.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi", reason="веб-приложение требует pip install -e '.[app]'")
pytest.importorskip("httpx", reason="TestClient требует httpx")

from fastapi.testclient import TestClient  # noqa: E402

from germovision.app import MAX_FILE_BYTES, create_app  # noqa: E402
from germovision.core.splitting import temporal_cluster_split  # noqa: E402
from germovision.data import SyntheticConfig, generate_isolates  # noqa: E402
from germovision.models import GVResist  # noqa: E402
from germovision.persistence import ModelBundle, save_bundle  # noqa: E402

VCF = (
    "##fileformat=VCFv4.2\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
    "NC_000962.3\t761155\t.\tC\tT\t900\tPASS\t"
    "ANN=T|missense_variant|MODERATE|rpoB|Rv0667|transcript|c.1349C>T|p.Ser450Leu\tGT\t1/1\n"
    "NC_000962.3\t2155168\t.\tC\tG\t900\tPASS\t"
    "ANN=G|missense_variant|MODERATE|katG|Rv1908c|transcript|c.944G>C|p.Ser315Thr\tGT\t1/1\n"
)

COUNTS = "\n".join(
    ["region,week,lineage,count"]
    + [f"KZ,{w},L4,{max(1, 40 - w * 2)}" for w in range(12)]
    + [f"KZ,{w},L2,{max(1, 4 + w * 2)}" for w in range(12)]
)


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    """Реальные обученные модели: подделка не проверила бы интеграцию."""
    out = tmp_path_factory.mktemp("models")
    ds = generate_isolates(SyntheticConfig(n_isolates=1200, seed=31))
    sp = temporal_cluster_split(ds.submission_dates, ds.clusters)
    models = {d: GVResist(d, random_state=0).fit(ds, sp) for d in ("RIF", "INH")}
    save_bundle(
        ModelBundle(models=models, manifest={"source": "synthetic", "synthetic": True}),
        out,
    )
    return out


@pytest.fixture(scope="module")
def client(models_dir):
    return TestClient(create_app(models_dir))


def _upload(name: str, data: bytes | str):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return ("files", (name, data, "application/octet-stream"))


# --------------------------------------------------------------------------
# Базовые эндпоинты
# --------------------------------------------------------------------------


def test_index_serves_interface(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Перетащите" in r.text
    assert "/api/analyze" in r.text


def test_status_lists_formats_and_models(client):
    s = client.get("/api/status").json()
    assert s["models_loaded"] is True
    assert s["models_synthetic"] is True
    assert len(s["formats"]) >= 6
    assert s["max_file_mb"] == MAX_FILE_BYTES // (1024 * 1024)


def test_app_starts_without_models(tmp_path):
    """Отсутствие моделей не должно ронять запуск: два анализа из трёх работают."""
    c = TestClient(create_app(tmp_path / "нет"))
    s = c.get("/api/status").json()
    assert s["models_loaded"] is False
    assert "save-models" in s["models_error"] or "нет сохранённых" in s["models_error"]

    r = c.post("/api/analyze", files=[_upload("counts.csv", COUNTS)])
    assert r.json()["results"][0]["ok"] is True  # GV-Growth не зависит от моделей


# --------------------------------------------------------------------------
# Анализ
# --------------------------------------------------------------------------


def test_vcf_goes_to_resistance(client):
    r = client.post("/api/analyze", files=[_upload("v.vcf", VCF)])
    res = r.json()["results"][0]
    assert res["ok"] and res["model"] == "GV-Resist"
    assert any(t["name"] == "isolates" for t in res["tables"])


def test_counts_go_to_growth(client):
    r = client.post("/api/analyze", files=[_upload("c.csv", COUNTS)])
    res = r.json()["results"][0]
    assert res["ok"] and res["model"] == "GV-Growth"
    assert {t["name"] for t in res["tables"]} == {"growth", "forecast"}


def test_fasta_goes_to_escape(client):
    import numpy as np

    from germovision.models.escape import AMINO_ACIDS

    rng = np.random.default_rng(2)
    base = "".join(rng.choice(list(AMINO_ACIDS), 90))
    lines = []
    for i in range(60):
        s = list(base)
        s[int(rng.integers(0, 90))] = str(rng.choice(list(AMINO_ACIDS)))
        lines.append(f">s{i}\n" + "".join(s))
    r = client.post("/api/analyze", files=[_upload("p.fasta", "\n".join(lines))])
    res = r.json()["results"][0]
    assert res["ok"] and res["model"] == "GV-Escape"


def test_multiple_files_processed_independently(client):
    """Сбой одного файла не должен отменять остальные."""
    r = client.post(
        "/api/analyze",
        files=[
            _upload("v.vcf", VCF),
            _upload("плохой.csv", "нет,таких,столбцов\n1,2,3\n"),
            _upload("c.csv", COUNTS),
        ],
    )
    results = r.json()["results"]
    assert len(results) == 3
    assert [x["ok"] for x in results] == [True, False, True]
    assert "id, gene, mutation" in results[1]["error"]


def test_empty_file_reported_clearly(client):
    r = client.post("/api/analyze", files=[_upload("пусто.csv", b"")])
    res = r.json()["results"][0]
    assert res["ok"] is False and "пуст" in res["error"]


def test_binary_file_reported_clearly(client):
    r = client.post("/api/analyze", files=[_upload("a.bam", b"BAM" + bytes([0]) * 20)])
    res = r.json()["results"][0]
    assert res["ok"] is False and "двоичный" in res["error"]


def test_oversized_file_suggests_cli(client):
    big = b"id,gene,mutation\n" + b"A,rpoB,S450L\n" * 3_000_000
    assert len(big) > MAX_FILE_BYTES
    r = client.post("/api/analyze", files=[_upload("big.csv", big)])
    res = r.json()["results"][0]
    assert res["ok"] is False and "germovision.predict" in res["error"]


def test_genome_explains_pipeline(client):
    genome = ">chr\n" + "ACGT" * 5000 + "\n"
    r = client.post("/api/analyze", files=[_upload("g.fa", genome)])
    res = r.json()["results"][0]
    assert res["ok"] is False
    assert "TB-Profiler" in res["error"] and "snpEff" in res["error"]


def test_response_is_json_serialisable(client):
    """Ответ должен переживать сериализацию: в таблицах есть numpy-числа."""
    r = client.post("/api/analyze", files=[_upload("c.csv", COUNTS)])
    json.dumps(r.json())  # не должно бросить


def test_tables_have_matching_columns(client):
    r = client.post("/api/analyze", files=[_upload("v.vcf", VCF)])
    for t in r.json()["results"][0]["tables"]:
        for row in t["rows"]:
            assert len(row) == len(t["columns"])
