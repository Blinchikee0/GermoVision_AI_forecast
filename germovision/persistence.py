"""Сохранение и загрузка обученных моделей.

До появления этого модуля обученную модель нельзя было применить к новому
образцу: единственным способом получить предсказание было переобучение
с нуля. Для исследовательского прогона это терпимо, для работы в
лаборатории — нет: там модель обучают редко, а применяют ежедневно.

Вместе с моделями сохраняется происхождение: на каких данных обучено,
когда, при каком разделении, с какими метриками. Заключение, выданное
через полгода, должно быть восстановимо — какая именно версия модели его
породила и что она показывала на удержанной выборке.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["ModelBundle", "save_bundle", "load_bundle"]

#: Версия формата. Повышается при несовместимых изменениях состава модели.
BUNDLE_FORMAT = 2

_MODELS_FILE = "models.joblib"
_MANIFEST_FILE = "manifest.json"


@dataclass
class ModelBundle:
    """Набор обученных моделей с описанием происхождения.

    Args:
        models: словарь «код препарата → обученная GVResist».
        manifest: происхождение и сводка качества.
    """

    models: dict[str, Any]
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def drugs(self) -> list[str]:
        return sorted(self.models)

    def describe(self) -> str:
        """Краткое описание для журнала и для интерфейса."""
        m = self.manifest
        lines = [
            f"GermoVision, модели обучены {m.get('trained_at', 'дата неизвестна')}",
            f"источник данных: {m.get('source', 'не указан')}"
            + ("  ⚠ СИНТЕТИЧЕСКИЕ" if m.get("synthetic") else ""),
            f"препаратов: {len(self.models)}  ({', '.join(self.drugs)})",
            f"обучающая выборка: {m.get('n_train', '?')} изолятов, "
            f"разделение: {m.get('split_strategy', '?')}",
        ]
        needs = [d for d, q in m.get("quality", {}).items() if q.get("requires_confirmation")]
        if needs:
            lines.append(
                "требуют фенотипического подтверждения: " + ", ".join(sorted(needs))
            )
        return "\n".join(lines)


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    """Сохранить модели и манифест в каталог.

    Манифест пишется отдельным JSON рядом с моделями: его можно прочитать
    и проверить, не загружая сериализованные объекты. Это важно, поскольку
    загрузка joblib исполняет код — файл из недоверенного источника
    открывать нельзя, и решение об этом принимается по манифесту.

    Args:
        bundle: набор моделей.
        path: каталог назначения.

    Returns:
        Путь к каталогу.
    """
    import joblib

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    manifest = dict(bundle.manifest)
    manifest.setdefault("format", BUNDLE_FORMAT)
    manifest.setdefault(
        "trained_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    manifest["drugs"] = bundle.drugs

    joblib.dump(bundle.models, out / _MODELS_FILE, compress=3)
    (out / _MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def load_bundle(path: str | Path) -> ModelBundle:
    """Загрузить модели из каталога.

    Raises:
        FileNotFoundError: каталога или файлов нет.
        ValueError: формат несовместим с текущей версией кода.
    """
    import joblib

    root = Path(path)
    manifest_path = root / _MANIFEST_FILE
    models_path = root / _MODELS_FILE

    if not manifest_path.exists() or not models_path.exists():
        raise FileNotFoundError(
            f"в {root} нет сохранённых моделей. Сначала выполните: "
            "python -m germovision.train --save-models "
            + str(root)
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fmt = manifest.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValueError(
            f"формат моделей {fmt}, ожидается {BUNDLE_FORMAT}. "
            "Модели обучены другой версией кода — переобучите их, а не "
            "загружайте: состав признаков мог измениться, и предсказания "
            "были бы неверными без единого сообщения об ошибке"
        )

    return ModelBundle(models=joblib.load(models_path), manifest=manifest)
