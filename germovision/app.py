"""Локальное веб-приложение: перетащить файл — получить анализ мутаций.

    python -m germovision.app                 # http://127.0.0.1:8000
    python -m germovision.app --models models --port 8080

Приложение работает **на машине пользователя**. Это не оформительское
решение: файлы содержат данные пациентов, и отправлять их на чужой
сервер ради удобства недопустимо. Ничего никуда не уходит, результат
скачивается локально.

Модели устойчивости нужно обучить заранее:

    python -m germovision.train --save-models models

Анализ мутаций (GV-Escape) и динамики линий (GV-Growth) предобученных
весов не требует — эти модели подгоняются на присланных данных.
"""

# Модуль намеренно обходится без `from __future__ import annotations`:
# FastAPI разбирает аннотации обработчиков во время выполнения, а
# отложенные аннотации превращают их в строки, которые pydantic не может
# разрешить внутри функции. Аннотации, не участвующие в разборе запросов,
# взяты в кавычки вручную.

import argparse
import json
import sys
from pathlib import Path

__all__ = ["create_app", "main"]

#: Предел размера файла. Выравнивание и подгонка профиля — операции в
#: памяти, и стомегабайтный FASTA положит процесс раньше, чем пользователь
#: увидит ошибку. Лучше отказать сразу и объяснить.
MAX_FILE_BYTES = 32 * 1024 * 1024

WEBUI = Path(__file__).resolve().parent / "webui"


def create_app(models_dir: "str | Path | None" = "models"):
    """Собрать приложение FastAPI.

    Args:
        models_dir: каталог с сохранёнными моделями устойчивости. Если его
            нет, приложение всё равно запускается: анализ мутаций и
            динамики линий от него не зависит, а при загрузке файла с
            вариантами пользователь получит понятное объяснение вместо
            отказа при старте.
    """
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse

    from .analysis import AnalysisError, analyze
    from .formats import SUPPORTED, FormatError, detect_and_parse
    from .persistence import load_bundle

    bundle = None
    bundle_error = ""
    if models_dir:
        try:
            bundle = load_bundle(models_dir)
        except (FileNotFoundError, ValueError) as exc:
            bundle_error = str(exc)

    app = FastAPI(
        title="GermoVision — анализ мутаций патогена",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = WEBUI / "index.html"
        if not page.exists():
            return HTMLResponse(
                "<h1>Interface files not found</h1>"
                f"<p>Expected {page}</p>",
                status_code=500,
            )
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/app.css")
    def stylesheet():
        from fastapi.responses import Response

        path = WEBUI / "app.css"
        if not path.exists():
            return Response("/* app.css not found */", media_type="text/css")
        return Response(path.read_text(encoding="utf-8"), media_type="text/css")

    @app.get("/app.js")
    def script():
        from fastapi.responses import Response

        path = WEBUI / "app.js"
        if not path.exists():
            return Response("// app.js not found", media_type="text/javascript")
        return Response(
            path.read_text(encoding="utf-8"), media_type="text/javascript"
        )

    @app.get("/api/surveillance")
    def surveillance() -> JSONResponse:
        """Национальная картина надзора из последнего прогона обучения.

        Приложение работает и без неё: раздел анализа файлов не зависит от
        того, обучались ли модели на этой машине. Поэтому отсутствие
        отчёта возвращается как признак, а не как ошибка — интерфейс
        покажет объяснение вместо пустых графиков.
        """
        from .dashboard import build_payload

        candidates = (
            Path("reports/metrics.json"),
            WEBUI.parent.parent / "reports" / "metrics.json",
        )
        for candidate in candidates:
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    return JSONResponse({"available": True, **build_payload(data)})
                except (json.JSONDecodeError, KeyError) as exc:
                    return JSONResponse({
                        "available": False,
                        "reason": f"reports/metrics.json is unreadable: {exc}",
                    })
        return JSONResponse({
            "available": False,
            "reason": (
                "No training report found. Run: python -m germovision.train "
                "--save-models models"
            ),
        })

    @app.get("/api/status")
    def status() -> dict:
        """Что система умеет принимать и какие модели готовы."""
        return {
            "formats": SUPPORTED,
            "models_loaded": bundle is not None,
            "models_error": bundle_error,
            "models_info": bundle.describe() if bundle else "",
            "models_synthetic": bool(bundle and bundle.manifest.get("synthetic")),
            "max_file_mb": MAX_FILE_BYTES // (1024 * 1024),
        }

    @app.post("/api/analyze")
    async def analyze_files(
        files: list[UploadFile] = File(...),  # noqa: B008 — идиома FastAPI
    ) -> JSONResponse:
        """Разобрать и проанализировать один или несколько файлов.

        Файлы обрабатываются независимо, и сбой одного не отменяет
        остальные: пользователь, перетащивший десять файлов, должен
        получить девять результатов и одно объяснение, а не общий отказ.
        """
        results: list[dict] = []

        for upload in files:
            name = upload.filename or "файл"
            try:
                content = await upload.read()
            except Exception as exc:  # noqa: BLE001 — сеть и диск дают что угодно
                results.append(_failure(name, f"Could not read file: {exc}"))
                continue

            if not content:
                results.append(_failure(name, "File is empty."))
                continue
            if len(content) > MAX_FILE_BYTES:
                mb = len(content) / (1024 * 1024)
                results.append(_failure(
                    name,
                    f"File is {mb:.0f} MB, over the {MAX_FILE_BYTES // (1024 * 1024)} MB "
                    "limit. For datasets this size use the command line: "
                    "python -m germovision.predict",
                ))
                continue

            try:
                parsed = detect_and_parse(name, content)
            except FormatError as exc:
                results.append(_failure(name, str(exc)))
                continue

            try:
                result = analyze(parsed, bundle=bundle)
            except AnalysisError as exc:
                results.append(_failure(name, str(exc), kind=parsed.kind))
                continue
            except Exception as exc:  # noqa: BLE001
                results.append(_failure(
                    name, f"Internal analysis error: {exc}", kind=parsed.kind
                ))
                continue

            payload = result.to_dict()
            payload["file"] = name
            payload["ok"] = True
            payload["input_summary"] = parsed.summary
            results.append(payload)

        return JSONResponse({"results": results})

    return app


def _failure(filename: str, message: str, kind: str = "") -> dict:
    return {"file": filename, "ok": False, "error": message, "kind": kind}


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Локальное приложение GermoVision: анализ мутаций патогена"
    )
    parser.add_argument("--models", default="models", help="каталог с моделями устойчивости")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="перезапуск при правках кода")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        import uvicorn
    except ImportError:
        print(
            "Не установлены зависимости веб-приложения. Установите их:\n"
            '    pip install -e ".[app]"',
            file=sys.stderr,
        )
        return 1

    print(f"GermoVision: http://{args.host}:{args.port}")
    print(f"Модели устойчивости: {args.models}")
    print("Данные не покидают эту машину.\n")

    uvicorn.run(create_app(args.models), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
