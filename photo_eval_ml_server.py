from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import subprocess
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen

from import_public_feedback import DEFAULT_EXPORT_URL, fetch_records
from photo_eval_ml_core import (
    ROOT_DIR,
    _build_exif_summary_from_exif,
    _coerce_float,
    _parse_capture_datetime,
    build_feedback_statistics,
    ensure_database,
    export_feedback_records,
    get_model_status,
    load_feedback_rows,
    load_model,
    predict_total_score,
    save_feedback,
)


HOST = os.environ.get("PHOTO_EVAL_ML_HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT") or os.environ.get("PHOTO_EVAL_ML_PORT", "8788"))
DEFAULT_ENTRY = "photo-evaluator-training-lab.html"
DL_LAB_DIR = ROOT_DIR / "dl-lab"
DL_MODEL_PATH = DL_LAB_DIR / "models" / "dl_residual_model.pt"
DL_MODEL_META_PATH = DL_LAB_DIR / "models" / "dl_residual_model_meta.json"
DL_BETA_RESULTS_PATH = DL_LAB_DIR / "exports" / "dl_beta_public_results.jsonl"
GITHUB_PAGES_DIR = ROOT_DIR.parent / "github-pages-photo-evaluator"
DL_PUBLIC_STATS_URL = os.environ.get("PHOTO_EVAL_DL_PUBLIC_STATS_URL", "https://photo-evaluator-dl-api.onrender.com/api/dl/public-stats")
DL_RUNTIME_CACHE: dict[str, object] = {
    "mtime": None,
    "model": None,
    "transform": None,
    "metadata": None,
    "torch": None,
    "Image": None,
}
ADMIN_STATE_LOCK = threading.Lock()
ADMIN_RUNTIME_STATE: dict[str, object] = {
    "running": False,
    "currentAction": "",
    "startedAt": "",
    "lastAction": "",
    "lastFinishedAt": "",
    "lastOk": None,
}
RUNTIME_METRICS_LOCK = threading.Lock()
RUNTIME_METRICS: dict[str, float | int] = {
    "evaluationCount": 0,
    "evaluationFailureCount": 0,
    "evaluationTotalMs": 0.0,
    "saveFailureCount": 0,
    "lastEvaluationAt": "",
}


ADMIN_ACTIONS: dict[str, dict[str, object]] = {
    "train-sync-github": {
        "label": "学習して GitHub 用へ反映",
        "commands": [
            {"cmd": ["python3", "train_photo_eval_model.py"], "cwd": ROOT_DIR},
            {
                "cmd": ["cp", str(ROOT_DIR / "photo_eval_model.json"), str(GITHUB_PAGES_DIR / "photo_eval_model.json")],
                "cwd": ROOT_DIR,
            },
            {"cmd": ["python3", "export_github_stats.py"], "cwd": ROOT_DIR},
        ],
    },
    "merge-public-sync": {
        "label": "公開フィードバック統合 + GitHub 反映",
        "commands": [
            {"cmd": ["python3", "update_public_feedback_model.py"], "cwd": ROOT_DIR},
            {
                "cmd": ["cp", str(ROOT_DIR / "photo_eval_model.json"), str(GITHUB_PAGES_DIR / "photo_eval_model.json")],
                "cwd": ROOT_DIR,
            },
            {"cmd": ["python3", "export_github_stats.py"], "cwd": ROOT_DIR},
        ],
    },
    "export-dl-dataset": {
        "label": "DL データセット書き出し",
        "commands": [
            {"cmd": ["python3", "export_dl_dataset.py"], "cwd": DL_LAB_DIR},
        ],
    },
    "train-dl-model": {
        "label": "DL モデル学習",
        "commands": [
            {"cmd": ["python3", "train_dl_residual_model.py"], "cwd": DL_LAB_DIR},
        ],
    },
    "run-dl-pipeline": {
        "label": "DL パイプライン実行",
        "commands": [
            {"cmd": ["python3", "export_dl_dataset.py"], "cwd": DL_LAB_DIR},
            {"cmd": ["python3", "train_dl_residual_model.py"], "cwd": DL_LAB_DIR},
        ],
    },
    "reset-dl-state": {
        "label": "DL 学習状態を空にする",
        "commands": [
            {"cmd": ["python3", "reset_dl_state.py"], "cwd": DL_LAB_DIR},
        ],
    },
}


def _mark_metric(name: str, amount: int = 1) -> None:
    with RUNTIME_METRICS_LOCK:
        RUNTIME_METRICS[name] = int(RUNTIME_METRICS.get(name, 0)) + amount


def _record_evaluation_metric(duration_ms: float, ok: bool) -> None:
    with RUNTIME_METRICS_LOCK:
        RUNTIME_METRICS["evaluationCount"] = int(RUNTIME_METRICS.get("evaluationCount", 0)) + 1
        RUNTIME_METRICS["evaluationTotalMs"] = float(RUNTIME_METRICS.get("evaluationTotalMs", 0.0)) + max(0.0, duration_ms)
        if not ok:
            RUNTIME_METRICS["evaluationFailureCount"] = int(RUNTIME_METRICS.get("evaluationFailureCount", 0)) + 1
        RUNTIME_METRICS["lastEvaluationAt"] = datetime.now(timezone.utc).isoformat()


def _build_runtime_metrics() -> dict[str, object]:
    with RUNTIME_METRICS_LOCK:
        evaluation_count = int(RUNTIME_METRICS.get("evaluationCount", 0))
        evaluation_failure_count = int(RUNTIME_METRICS.get("evaluationFailureCount", 0))
        evaluation_total_ms = float(RUNTIME_METRICS.get("evaluationTotalMs", 0.0))
        save_failure_count = int(RUNTIME_METRICS.get("saveFailureCount", 0))
        last_evaluation_at = str(RUNTIME_METRICS.get("lastEvaluationAt") or "")
    average_ms = round(evaluation_total_ms / evaluation_count, 2) if evaluation_count else 0.0
    failure_rate = round((evaluation_failure_count / evaluation_count) * 100, 2) if evaluation_count else 0.0
    return {
        "evaluationCount": evaluation_count,
        "evaluationFailureCount": evaluation_failure_count,
        "evaluationAverageMs": average_ms,
        "evaluationFailureRate": failure_rate,
        "saveFailureCount": save_failure_count,
        "lastEvaluationAt": last_evaluation_at,
    }


class PhotoEvalHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.path = f"/{DEFAULT_ENTRY}"
            return super().do_GET()
        if parsed.path == "/api/health":
            return self._write_json({"success": True, "message": "ok"})
        if parsed.path == "/api/ml/status":
            return self._write_json({"success": True, **get_model_status()})
        if parsed.path == "/api/ml/stats":
            return self._write_json({"success": True, "stats": build_feedback_statistics()})
        if parsed.path == "/api/dl/status":
            return self._write_json({"success": True, **_get_dl_status()})
        if parsed.path == "/api/dl/stats":
            return self._write_json({"success": True, "stats": _build_dl_statistics()})
        if parsed.path == "/api/dl/public-stats":
            return self._write_json({"success": True, "stats": _get_dl_public_statistics()})
        if parsed.path == "/api/admin/status":
            return self._write_json({"success": True, **_build_admin_status()})
        if parsed.path == "/api/ml/export":
            params = parse_qs(parsed.query)
            fmt = (params.get("format") or ["json"])[0].lower()
            if fmt not in {"json", "csv"}:
                return self._write_json(
                    {"success": False, "message": "format は json または csv を指定してください"},
                    status=HTTPStatus.BAD_REQUEST,
                )
            body, content_type = export_feedback_records(fmt)
            file_name = f"photo-eval-dataset.{fmt}"
            return self._write_bytes(
                body.encode("utf-8"),
                content_type=content_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{file_name}"',
                },
            )
        if parsed.path == "/api/review/public-records":
            params = parse_qs(parsed.query)
            export_url = (params.get("url") or [DEFAULT_EXPORT_URL])[0]
            try:
                records = [_normalize_public_review_record(payload) for payload in fetch_records(export_url)]
            except Exception as error:
                return self._write_json(
                    {"success": False, "message": str(error)},
                    status=HTTPStatus.BAD_GATEWAY,
                )
            return self._write_json({
                "success": True,
                "source": "public",
                "count": len(records),
                "records": records,
            })
        if parsed.path == "/api/review/local-records":
            rows = load_feedback_rows()
            records = [_normalize_local_review_record(row) for row in rows]
            return self._write_json({
                "success": True,
                "source": "local",
                "count": len(records),
                "records": records,
            })
        if parsed.path == "/api/review/image":
            params = parse_qs(parsed.query)
            remote_url = (params.get("url") or [""])[0].strip()
            if not remote_url:
                return self._write_json(
                    {"success": False, "message": "画像URLが指定されていません"},
                    status=HTTPStatus.BAD_REQUEST,
                )
            return self._proxy_review_image(remote_url)

        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/ml/predict":
            return self._handle_predict()
        if parsed.path == "/api/dl/predict":
            return self._handle_dl_predict()
        if parsed.path == "/api/dl/evaluation":
            return self._handle_dl_evaluation()
        if parsed.path == "/api/ml/feedback":
            return self._handle_feedback()
        if parsed.path == "/api/admin/run":
            return self._handle_admin_run()
        return self._write_json(
            {"success": False, "message": "unknown endpoint"},
            status=HTTPStatus.NOT_FOUND,
        )

    def guess_type(self, path: str) -> str:
        if path.endswith(".webmanifest"):
            return "application/manifest+json"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"

    def _handle_predict(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        started_at = time.perf_counter()
        ok = False
        try:
            prediction = predict_total_score(
                payload.get("features") or {},
                payload.get("predicted_scores") or {},
            )
            ok = True
            self._write_json({"success": True, **prediction})
        finally:
            _record_evaluation_metric((time.perf_counter() - started_at) * 1000, ok)

    def _handle_dl_predict(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        started_at = time.perf_counter()
        ok = False
        image_data_url = str(payload.get("imageDataUrl") or "")
        try:
            rule_score = float(payload.get("ruleScore"))
        except (TypeError, ValueError):
            _record_evaluation_metric((time.perf_counter() - started_at) * 1000, False)
            return self._write_json(
                {"success": False, "message": "ruleScore が不正です"},
                status=HTTPStatus.BAD_REQUEST,
            )
        try:
            prediction = _predict_with_dl_model(image_data_url, rule_score)
            ok = True
            self._write_json({"success": True, **prediction})
        finally:
            _record_evaluation_metric((time.perf_counter() - started_at) * 1000, ok)

    def _handle_feedback(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            payload = _attach_full_exif_metadata(payload)
            saved = save_feedback(payload)
        except ValueError as error:
            _mark_metric("saveFailureCount")
            return self._write_json(
                {"success": False, "message": str(error)},
                status=HTTPStatus.BAD_REQUEST,
            )
        self._write_json(
            {
                "success": True,
                "message": "既存の学習データを更新しました" if saved.get("deduplicated") else "フィードバックを保存しました",
                "imageId": saved["image_id"],
                "savedAt": saved["updated_at"],
                "exifStored": bool((saved.get("image_metadata") or {}).get("exif")),
                "deduplicated": bool(saved.get("deduplicated")),
                "duplicateOf": saved.get("duplicate_of") or "",
            }
        )

    def _handle_dl_evaluation(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            payload = _attach_full_exif_metadata(payload)
            saved_record = _save_dl_public_evaluation(payload)
        except ValueError as error:
            return self._write_json(
                {"success": False, "message": str(error)},
                status=HTTPStatus.BAD_REQUEST,
            )
        self._write_json(
            {
                "success": True,
                "saved": True,
                "record": saved_record,
                "stats": _build_dl_public_statistics(),
            }
        )

    def _handle_admin_run(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        action = str(payload.get("action") or "").strip()
        if action not in ADMIN_ACTIONS:
            return self._write_json(
                {"success": False, "message": "unknown admin action"},
                status=HTTPStatus.BAD_REQUEST,
            )
        with ADMIN_STATE_LOCK:
            if ADMIN_RUNTIME_STATE["running"]:
                current_action = str(ADMIN_RUNTIME_STATE.get("currentAction") or "")
                return self._write_json(
                    {
                        "success": False,
                        "message": f"管理処理を実行中です: {current_action or 'unknown'}",
                        "status": _build_admin_status(),
                    },
                    status=HTTPStatus.CONFLICT,
                )
            ADMIN_RUNTIME_STATE.update({
                "running": True,
                "currentAction": action,
                "startedAt": datetime.now(timezone.utc).isoformat(),
            })
        try:
            result = _run_admin_action(action)
        except Exception as error:  # noqa: BLE001
            with ADMIN_STATE_LOCK:
                ADMIN_RUNTIME_STATE.update({
                    "running": False,
                    "currentAction": "",
                    "lastAction": action,
                    "lastFinishedAt": datetime.now(timezone.utc).isoformat(),
                    "lastOk": False,
                })
            return self._write_json(
                {
                    "success": False,
                    "message": f"管理処理の実行に失敗しました: {error}",
                    "status": _build_admin_status(),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        with ADMIN_STATE_LOCK:
            ADMIN_RUNTIME_STATE.update({
                "running": False,
                "currentAction": "",
                "lastAction": action,
                "lastFinishedAt": datetime.now(timezone.utc).isoformat(),
                "lastOk": bool(result.get("ok")),
            })
        return self._write_json({"success": True, **result})

    def _read_json_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._write_json(
                {"success": False, "message": "JSON body が不正です"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return None

    def _write_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        self._write_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            content_type="application/json; charset=utf-8",
            status=status,
        )

    def _write_bytes(
        self,
        payload: bytes,
        *,
        content_type: str,
        status: int = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _proxy_review_image(self, remote_url: str) -> None:
        try:
            request = Request(
                remote_url,
                headers={
                    "User-Agent": "PhotoEvalTrainingLab/1.0",
                    "Accept": "image/*,*/*;q=0.8",
                },
            )
            with urlopen(request, timeout=20) as response:
                payload = response.read()
                content_type = response.headers.get_content_type() or "application/octet-stream"
        except URLError as error:
            self._write_json(
                {"success": False, "message": f"画像の取得に失敗しました: {error.reason}"},
                status=HTTPStatus.BAD_GATEWAY,
            )
            return
        except Exception as error:
            self._write_json(
                {"success": False, "message": f"画像の取得に失敗しました: {error}"},
                status=HTTPStatus.BAD_GATEWAY,
            )
            return

        self._write_bytes(
            payload,
            content_type=content_type,
            headers={"Cache-Control": "no-store"},
        )


def _build_drive_thumbnail_url(file_id: str) -> str:
    return f"https://drive.google.com/thumbnail?id={quote(file_id)}&sz=w1200" if file_id else ""


def _build_drive_view_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=view&id={quote(file_id)}" if file_id else ""


def _extract_drive_file_id(drive_url: str) -> str:
    parsed = urlparse(drive_url or "")
    if not parsed.netloc:
        return ""
    query = parse_qs(parsed.query)
    if query.get("id"):
        return str(query["id"][0] or "")
    parts = [part for part in parsed.path.split("/") if part]
    if "d" in parts:
        index = parts.index("d")
        if index + 1 < len(parts):
            return parts[index + 1]
    return ""


def _build_review_image_proxy_url(remote_url: str) -> str:
    return f"/api/review/image?url={quote(remote_url, safe='')}" if remote_url else ""


def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_iso_datetime(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _get_dl_status() -> dict[str, object]:
    metadata = _read_json_file(DL_MODEL_META_PATH) if DL_MODEL_META_PATH.exists() else {}
    available = DL_MODEL_PATH.exists() and DL_MODEL_META_PATH.exists()
    reason = ""
    if not available:
        reason = "DLモデルがまだ学習されていません"
    return {
        "available": available,
        "modelPath": str(DL_MODEL_PATH),
        "metadataPath": str(DL_MODEL_META_PATH),
        "modelType": str(metadata.get("model_type") or ""),
        "sampleCount": int(metadata.get("sample_count") or 0),
        "trainCount": int(metadata.get("train_count") or 0),
        "validationCount": int(metadata.get("validation_count") or 0),
        "validationMae": metadata.get("validation_mae"),
        "trainedAt": str(metadata.get("trained_at") or ""),
        "reason": reason,
    }


def _decode_data_url_image(image_data_url: str) -> bytes:
    if not image_data_url.startswith("data:") or "," not in image_data_url:
        raise ValueError("画像データ形式が不正です")
    _header, encoded = image_data_url.split(",", 1)
    return base64.b64decode(encoded)


def _load_dl_runtime() -> tuple[object | None, object | None, dict, object | None, object | None, str]:
    if not DL_MODEL_PATH.exists() or not DL_MODEL_META_PATH.exists():
        return None, None, {}, None, None, "DLモデルがまだ学習されていません"

    mtime = DL_MODEL_PATH.stat().st_mtime
    if DL_RUNTIME_CACHE["mtime"] == mtime and DL_RUNTIME_CACHE["model"] is not None:
        return (
            DL_RUNTIME_CACHE["model"],
            DL_RUNTIME_CACHE["transform"],
            DL_RUNTIME_CACHE["metadata"] if isinstance(DL_RUNTIME_CACHE["metadata"], dict) else {},
            DL_RUNTIME_CACHE["torch"],
            DL_RUNTIME_CACHE["Image"],
            "",
        )

    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        from torch import nn  # type: ignore
        from torchvision import transforms  # type: ignore
    except Exception as error:
        return None, None, {}, None, None, f"PyTorch依存が未導入です: {error}"

    metadata = _read_json_file(DL_MODEL_META_PATH)
    output_names = metadata.get("output_names")
    if not isinstance(output_names, list) or not output_names:
        output_names = ["total"]
    genre_labels = metadata.get("genre_labels")
    if not isinstance(genre_labels, list):
        genre_labels = []

    class TinyScoreCNN(nn.Module):
        def __init__(self, output_dim: int, genre_dim: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.shared = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 96),
                nn.ReLU(inplace=True),
            )
            self.score_head = nn.Linear(96, output_dim)
            self.genre_head = nn.Linear(96, genre_dim) if genre_dim > 0 else None

        def forward(self, x):
            shared = self.shared(self.features(x))
            score_output = self.score_head(shared)
            genre_output = self.genre_head(shared) if self.genre_head is not None else None
            return score_output, genre_output

    image_size = int(metadata.get("image_size") or 224)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    checkpoint = torch.load(DL_MODEL_PATH, map_location="cpu")
    checkpoint_genre_labels = checkpoint.get("genre_labels")
    if isinstance(checkpoint_genre_labels, list) and checkpoint_genre_labels:
        genre_labels = checkpoint_genre_labels
    model = TinyScoreCNN(len(output_names), len(genre_labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    DL_RUNTIME_CACHE["mtime"] = mtime
    DL_RUNTIME_CACHE["model"] = model
    DL_RUNTIME_CACHE["transform"] = transform
    DL_RUNTIME_CACHE["metadata"] = metadata
    DL_RUNTIME_CACHE["torch"] = torch
    DL_RUNTIME_CACHE["Image"] = Image
    return model, transform, metadata, torch, Image, ""


def _predict_with_dl_model(image_data_url: str, rule_score: float) -> dict[str, object]:
    model, transform, metadata, torch, Image, error_message = _load_dl_runtime()
    if model is None or transform is None or torch is None or Image is None:
        return {
            "available": False,
            "usedModel": False,
            "predictedTotalScore": round(rule_score, 2),
            "predictedDelta": 0.0,
            "predictedScores": {},
            "predictedGenre": "",
            "genreProbabilities": {},
            "genreLabels": metadata.get("genre_labels") or [],
            "modelType": str(metadata.get("model_type") or ""),
            "sampleCount": int(metadata.get("sample_count") or 0),
            "validationMae": metadata.get("validation_mae"),
            "validationMaeByOutput": metadata.get("validation_mae_by_output") or {},
            "validationGenreAccuracy": metadata.get("validation_genre_accuracy"),
            "outputNames": metadata.get("output_names") or ["total"],
            "reason": error_message or "DLモデルを利用できません",
        }

    try:
        raw_bytes = _decode_data_url_image(image_data_url)
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            score_output, genre_output = model(tensor)
            raw_output = score_output.squeeze(0).detach().cpu().tolist()
            raw_genre_output = genre_output.squeeze(0).detach().cpu().tolist() if genre_output is not None else []
    except Exception as error:
        return {
            "available": True,
            "usedModel": False,
            "predictedTotalScore": round(rule_score, 2),
            "predictedDelta": 0.0,
            "predictedScores": {},
            "predictedGenre": "",
            "genreProbabilities": {},
            "genreLabels": metadata.get("genre_labels") or [],
            "modelType": str(metadata.get("model_type") or ""),
            "sampleCount": int(metadata.get("sample_count") or 0),
            "validationMae": metadata.get("validation_mae"),
            "validationMaeByOutput": metadata.get("validation_mae_by_output") or {},
            "validationGenreAccuracy": metadata.get("validation_genre_accuracy"),
            "outputNames": metadata.get("output_names") or ["total"],
            "reason": f"DL推論に失敗しました: {error}",
        }

    if not isinstance(raw_output, list):
        raw_output = [float(raw_output)]
    output_names = metadata.get("output_names")
    if not isinstance(output_names, list) or not output_names:
        output_names = ["total"]

    predicted_scores = {}
    for index, name in enumerate(output_names):
        try:
            value = float(raw_output[index])
        except (IndexError, TypeError, ValueError):
            continue
        predicted_scores[str(name)] = round(max(0.0, min(100.0, value)), 2)

    if "total" in predicted_scores:
        predicted_total = float(predicted_scores["total"])
        predicted_delta = predicted_total - rule_score
    else:
        residual = max(-25.0, min(25.0, float(raw_output[0] if raw_output else 0.0)))
        predicted_total = max(0.0, min(100.0, rule_score + residual))
        predicted_delta = residual

    predicted_genre = ""
    genre_probabilities = {}
    genre_labels = metadata.get("genre_labels")
    if isinstance(genre_labels, list) and genre_labels and isinstance(raw_genre_output, list) and raw_genre_output:
        probabilities_tensor = torch.softmax(torch.tensor(raw_genre_output, dtype=torch.float32), dim=0)
        probabilities = probabilities_tensor.tolist()
        best_index = int(probabilities_tensor.argmax().item())
        if 0 <= best_index < len(genre_labels):
            predicted_genre = str(genre_labels[best_index])
        genre_probabilities = {
            str(label): round(float(probability), 4)
            for label, probability in zip(genre_labels, probabilities)
        }

    return {
        "available": True,
        "usedModel": True,
        "predictedTotalScore": round(predicted_total, 2),
        "predictedDelta": round(predicted_delta, 2),
        "predictedScores": predicted_scores,
        "predictedGenre": predicted_genre,
        "genreProbabilities": genre_probabilities,
        "genreLabels": metadata.get("genre_labels") or [],
        "modelType": str(metadata.get("model_type") or ""),
        "sampleCount": int(metadata.get("sample_count") or 0),
        "validationMae": metadata.get("validation_mae"),
        "validationMaeByOutput": metadata.get("validation_mae_by_output") or {},
        "validationGenreAccuracy": metadata.get("validation_genre_accuracy"),
        "outputNames": metadata.get("output_names") or ["total"],
        "reason": "",
    }


def _build_admin_status() -> dict[str, object]:
    ml_status = get_model_status()
    dl_status = _get_dl_status()
    records = load_feedback_rows()
    model = load_model() or {}
    with ADMIN_STATE_LOCK:
        admin_runtime = dict(ADMIN_RUNTIME_STATE)
    model_trained_at = _parse_iso_datetime(model.get("trained_at"))
    new_learning_count = 0
    if model_trained_at is not None:
        for record in records:
            updated_at = _parse_iso_datetime(record.get("updated_at") or record.get("created_at"))
            if updated_at and updated_at > model_trained_at:
                new_learning_count += 1
    else:
        new_learning_count = len(records)
    return {
        "entryUrl": f"http://{HOST}:{PORT}/",
        "evaluateUrl": f"http://{HOST}:{PORT}/photo-evaluator-training-lab.html",
        "reviewUrl": f"http://{HOST}:{PORT}/photo-evaluator-training-lab.html#review",
        "statsUrl": f"http://{HOST}:{PORT}/photo-evaluator-training-lab.html#stats",
        "dlStatsUrl": f"http://{HOST}:{PORT}/photo-evaluator-training-lab.html#dl-stats",
        "recordCount": len(records),
        "newLearningCount": new_learning_count,
        "mlStatus": ml_status,
        "dlStatus": dl_status,
        "actions": [
            {"id": action_id, "label": str(config["label"])}
            for action_id, config in ADMIN_ACTIONS.items()
        ],
        "downloads": {
            "modelJson": "/photo_eval_model.json",
            "dbFile": "/photo_eval_ml.sqlite3",
            "datasetJson": "/api/ml/export?format=json",
            "datasetCsv": "/api/ml/export?format=csv",
            "dlDatasetJsonl": "/dl-lab/exports/dl_dataset.jsonl",
            "dlDatasetSummary": "/dl-lab/exports/dl_dataset_summary.json",
            "dlModelMeta": "/dl-lab/models/dl_residual_model_meta.json",
        },
        "adminRuntime": admin_runtime,
        "runtimeMetrics": _build_runtime_metrics(),
    }


def _build_dl_statistics() -> dict[str, object]:
    dataset_path = DL_LAB_DIR / "exports" / "dl_dataset.jsonl"
    summary_path = DL_LAB_DIR / "exports" / "dl_dataset_summary.json"
    metadata_path = DL_MODEL_META_PATH

    summary = _read_json_file(summary_path) if summary_path.exists() else {}
    metadata = _read_json_file(metadata_path) if metadata_path.exists() else {}
    records: list[dict] = []
    if dataset_path.exists():
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    corrected_hist = [{"label": f"{start}-{min(start + 9, 100)}", "count": 0} for start in range(0, 100, 10)]
    target_hist = [{"label": label, "count": 0} for label in ["-25〜-16", "-15〜-6", "-5〜5", "6〜15", "16〜25"]]
    genre_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}

    for record in records:
        corrected_score = record.get("corrected_score")
        if isinstance(corrected_score, (int, float)):
            bucket_index = min(int(float(corrected_score) // 10), len(corrected_hist) - 1)
            corrected_hist[bucket_index]["count"] += 1

        target_value = record.get("target")
        if isinstance(target_value, (int, float)):
            numeric = float(target_value)
            if numeric <= -16:
                target_hist[0]["count"] += 1
            elif numeric <= -6:
                target_hist[1]["count"] += 1
            elif numeric <= 5:
                target_hist[2]["count"] += 1
            elif numeric <= 15:
                target_hist[3]["count"] += 1
            else:
                target_hist[4]["count"] += 1

        genre = str(record.get("genre") or "").strip()
        if genre:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        evaluation_mode = str(record.get("evaluation_mode") or "").strip()
        if evaluation_mode:
            mode_counts[evaluation_mode] = mode_counts.get(evaluation_mode, 0) + 1

    corrected_hist_max = max((bucket["count"] for bucket in corrected_hist), default=0)
    target_hist_max = max((bucket["count"] for bucket in target_hist), default=0)
    top_genres = [
        {"label": label, "count": count}
        for label, count in sorted(genre_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    top_modes = [
        {"label": label, "count": count}
        for label, count in sorted(mode_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    return {
        "summary": {
            "sourceRows": int(summary.get("source_rows") or 0),
            "exportedRows": int(summary.get("exported_rows") or 0),
            "skippedMissingImage": int(summary.get("skipped_missing_image") or 0),
            "skippedMissingLabel": int(summary.get("skipped_missing_label") or 0),
            "target": str(summary.get("target") or metadata.get("target") or "residual"),
            "trainedSampleCount": int(metadata.get("sample_count") or 0),
            "trainCount": int(metadata.get("train_count") or 0),
            "validationCount": int(metadata.get("validation_count") or 0),
            "validationMae": metadata.get("validation_mae"),
            "trainedAt": metadata.get("trained_at"),
            "status": str(metadata.get("status") or ("trained" if DL_MODEL_PATH.exists() else "untrained")),
        },
        "correctedScoreHistogram": {
            "maxCount": corrected_hist_max,
            "buckets": corrected_hist,
        },
        "targetHistogram": {
            "maxCount": target_hist_max,
            "buckets": target_hist,
        },
        "genreCounts": top_genres,
        "evaluationModeCounts": top_modes,
    }


def _read_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _save_dl_public_evaluation(payload: dict) -> dict[str, object]:
    try:
        total_score = float(payload.get("totalScore"))
    except (TypeError, ValueError):
        raise ValueError("totalScore が不正です")

    required_score_keys = [
        "compositionScore",
        "lightScore",
        "colorScore",
        "technicalScore",
        "subjectScore",
        "impressionScore",
    ]
    normalized_scores: dict[str, float] = {}
    for key in required_score_keys:
        try:
            normalized_scores[key] = round(float(payload.get(key)), 2)
        except (TypeError, ValueError):
            raise ValueError(f"{key} が不正です")

    genre = str(payload.get("genre") or "other").strip() or "other"
    engine = str(payload.get("engine") or "rule").strip() or "rule"
    model_version = str(payload.get("modelVersion") or payload.get("modelType") or "").strip()
    image_metadata = dict(payload.get("image_metadata") or {})
    feedback = dict(payload.get("feedback") or {})
    score_feedback = str(feedback.get("scoreFeedback") or payload.get("scoreFeedback") or "").strip()
    genre_feedback = str(feedback.get("genreFeedback") or payload.get("genreFeedback") or "").strip()
    corrected_genre = str(feedback.get("correctedGenre") or payload.get("correctedGenre") or "").strip()
    record = {
        "imageId": str(payload.get("imageId") or payload.get("image_id") or ""),
        "anonymousUserId": str(payload.get("anonymousUserId") or payload.get("anonymous_user_id") or "").strip(),
        "createdAt": str(payload.get("createdAt") or datetime.now(timezone.utc).isoformat()),
        "engine": engine,
        "modelVersion": model_version,
        "genre": genre,
        "totalScore": round(total_score, 2),
        "fileName": str(payload.get("fileName") or image_metadata.get("originalFileName") or ""),
        "imageMetadata": image_metadata,
        "feedback": {
            "scoreFeedback": score_feedback,
            "genreFeedback": genre_feedback,
            "correctedGenre": corrected_genre,
            "savedAt": str(feedback.get("savedAt") or datetime.now(timezone.utc).isoformat()) if (score_feedback or genre_feedback) else "",
        },
        **normalized_scores,
    }
    DL_BETA_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_records = _read_jsonl_records(DL_BETA_RESULTS_PATH)
    image_id = str(record.get("imageId") or "").strip()
    updated = False
    if image_id:
        for index, existing in enumerate(existing_records):
            if str(existing.get("imageId") or "").strip() == image_id:
                existing_records[index] = {
                    **existing,
                    **record,
                    "feedback": {
                        **dict(existing.get("feedback") or {}),
                        **dict(record.get("feedback") or {}),
                    },
                }
                record = existing_records[index]
                updated = True
                break
    if not updated:
        existing_records.append(record)
    with DL_BETA_RESULTS_PATH.open("w", encoding="utf-8") as handle:
        for item in existing_records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return record


def _build_dl_public_statistics() -> dict[str, object]:
    records = _read_jsonl_records(DL_BETA_RESULTS_PATH)
    bins = [
        {"label": "0-49", "min": 0.0, "max": 49.999},
        {"label": "50-59", "min": 50.0, "max": 59.999},
        {"label": "60-69", "min": 60.0, "max": 69.999},
        {"label": "70-79", "min": 70.0, "max": 79.999},
        {"label": "80-89", "min": 80.0, "max": 89.999},
        {"label": "90-100", "min": 90.0, "max": 100.001},
    ]
    histogram = [{"label": bucket["label"], "count": 0} for bucket in bins]
    genre_counts: dict[str, int] = {}
    camera_counts: dict[str, int] = {}
    location_counts: dict[str, int] = {}
    lens_counts: dict[str, int] = {}
    hour_buckets = [{"label": f"{hour:02d}", "count": 0} for hour in range(24)]
    altitude_buckets = [
        {"label": "海面下", "min": None, "max": 0, "count": 0},
        {"label": "0-49m", "min": 0, "max": 50, "count": 0},
        {"label": "50-99m", "min": 50, "max": 100, "count": 0},
        {"label": "100-199m", "min": 100, "max": 200, "count": 0},
        {"label": "200-499m", "min": 200, "max": 500, "count": 0},
        {"label": "500m以上", "min": 500, "max": None, "count": 0},
    ]
    average_keys = [
        ("composition", "compositionScore"),
        ("light", "lightScore"),
        ("color", "colorScore"),
        ("technical", "technicalScore"),
        ("subject", "subjectScore"),
        ("impact", "impressionScore"),
        ("total", "totalScore"),
    ]
    exif_count = 0
    geo_count = 0
    altitude_count = 0
    estimated_user_ids: set[str] = set()

    for record in records:
        anonymous_user_id = str(record.get("anonymousUserId") or record.get("anonymous_user_id") or "").strip()
        if anonymous_user_id:
            estimated_user_ids.add(anonymous_user_id)
        total_score = float(record.get("totalScore") or 0.0)
        for index, bucket in enumerate(bins):
            if bucket["min"] <= total_score <= bucket["max"]:
                histogram[index]["count"] += 1
                break
        genre = str(record.get("genre") or "other").strip() or "other"
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

        image_metadata = dict(record.get("imageMetadata") or {})
        exif_summary = dict(image_metadata.get("exifSummary") or {})
        if not exif_summary and isinstance(image_metadata.get("exif"), dict):
          exif_summary = _build_exif_summary_from_exif(dict(image_metadata.get("exif") or {}))
        if any(value not in ("", None) for value in exif_summary.values()):
            exif_count += 1

        camera_name = str(exif_summary.get("cameraModel") or "").strip()
        if camera_name:
            camera_counts[camera_name] = camera_counts.get(camera_name, 0) + 1

        lens_name = str(exif_summary.get("lensModel") or "").strip()
        if lens_name:
            lens_counts[lens_name] = lens_counts.get(lens_name, 0) + 1

        location_label = str(exif_summary.get("locationLabel") or exif_summary.get("prefecture") or "").strip()
        if location_label:
            location_counts[location_label] = location_counts.get(location_label, 0) + 1

        capture_hour = exif_summary.get("captureHour")
        if capture_hour in ("", None):
            parsed_capture = _parse_capture_datetime(exif_summary.get("dateTimeOriginal"))
            capture_hour = parsed_capture.hour if parsed_capture else None
        if isinstance(capture_hour, int) and 0 <= capture_hour <= 23:
            hour_buckets[capture_hour]["count"] += 1

        latitude = _coerce_float(exif_summary.get("gpsLatitude"))
        longitude = _coerce_float(exif_summary.get("gpsLongitude"))
        if latitude is not None and longitude is not None:
            geo_count += 1

        altitude_value = _coerce_float(exif_summary.get("altitudeMeters"))
        if altitude_value is not None:
            altitude_count += 1
            for bucket in altitude_buckets:
                minimum = bucket["min"]
                maximum = bucket["max"]
                if minimum is None and altitude_value < float(maximum):
                    bucket["count"] += 1
                    break
                if maximum is None and altitude_value >= float(minimum):
                    bucket["count"] += 1
                    break
                if minimum is not None and maximum is not None and minimum <= altitude_value < maximum:
                    bucket["count"] += 1
                    break

    item_averages = []
    for label, key in average_keys:
        values = [float(record.get(key) or 0.0) for record in records if record.get(key) is not None]
        average = (sum(values) / len(values)) if values else 0.0
        item_averages.append({"label": label, "value": round(average, 2)})

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sampleCount": len(records),
        "summary": {
            "totalRecords": len(records),
            "estimatedUserCount": len(estimated_user_ids),
            "averageTotalScore": next((item["value"] for item in item_averages if item["label"] == "total"), 0.0),
            "exifRecords": exif_count,
            "exifCoverageRate": round((exif_count / len(records)) * 100, 1) if records else 0.0,
            "geoRecords": geo_count,
            "altitudeRecords": altitude_count,
        },
        "scoreHistogram": {
            "maxCount": max((bucket["count"] for bucket in histogram), default=0),
            "buckets": histogram,
        },
        "genreCounts": [
            {"label": label, "count": count}
            for label, count in sorted(genre_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "itemAverages": item_averages,
        "captureHourHistogram": hour_buckets,
        "topCameras": [
            {"label": label, "count": count}
            for label, count in sorted(camera_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "topLenses": [
            {"label": label, "count": count}
            for label, count in sorted(lens_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "locationCounts": [
            {"label": label, "count": count}
            for label, count in sorted(location_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "altitudeHistogram": {
            "maxCount": max((bucket["count"] for bucket in altitude_buckets), default=0),
            "buckets": altitude_buckets,
        },
        "storagePath": str(DL_BETA_RESULTS_PATH),
    }


def _fetch_remote_dl_public_statistics() -> dict[str, object] | None:
    if not DL_PUBLIC_STATS_URL:
        return None
    try:
        request = Request(DL_PUBLIC_STATS_URL, headers={"Accept": "application/json"})
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        stats = payload.get("stats") if isinstance(payload, dict) else None
        if isinstance(stats, dict):
            stats = dict(stats)
            stats["source"] = "remote"
            return stats
    except Exception:
        return None
    return None


def _get_dl_public_statistics() -> dict[str, object]:
    remote_stats = _fetch_remote_dl_public_statistics()
    if remote_stats:
        return remote_stats
    local_stats = _build_dl_public_statistics()
    local_stats["source"] = "local"
    return local_stats


def _run_admin_action(action: str) -> dict[str, object]:
    action_config = ADMIN_ACTIONS[action]
    outputs: list[str] = []
    ok = True
    for step in action_config["commands"]:
        step_cmd = step["cmd"]
        step_cwd = Path(step["cwd"])
        outputs.append(f"$ {' '.join(step_cmd)}")
        result = subprocess.run(
            step_cmd,
            cwd=step_cwd,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            outputs.append(result.stdout.strip())
        if result.stderr:
            outputs.append(result.stderr.strip())
        if result.returncode != 0:
            ok = False
            outputs.append(f"(exit {result.returncode})")
            break
    return {
        "action": action,
        "label": action_config["label"],
        "ok": ok,
        "output": "\n\n".join(chunk for chunk in outputs if chunk),
        "status": _build_admin_status(),
    }


def _pick_exif_value(exif: dict, *keys: str):
    for key in keys:
        if key in exif and exif[key] not in ("", None):
            return exif[key]
    for key in keys:
        lowered = key.lower()
        for existing_key, value in exif.items():
            existing = str(existing_key).lower()
            if existing == lowered or existing.endswith(f":{lowered}"):
                if value not in ("", None):
                    return value
    return None


def _build_exif_summary(exif: dict) -> dict:
    return {
        "cameraMake": _pick_exif_value(exif, "Make"),
        "cameraModel": _pick_exif_value(exif, "Model"),
        "lensModel": _pick_exif_value(exif, "LensModel", "LensID"),
        "iso": _pick_exif_value(exif, "ISO"),
        "aperture": _pick_exif_value(exif, "FNumber", "Aperture"),
        "exposureTime": _pick_exif_value(exif, "ExposureTime", "ShutterSpeed"),
        "focalLength": _pick_exif_value(exif, "FocalLength"),
        "focalLength35mm": _pick_exif_value(exif, "FocalLengthIn35mmFormat", "FocalLength35efl"),
        "dateTimeOriginal": _pick_exif_value(exif, "DateTimeOriginal", "SubSecDateTimeOriginal", "CreateDate"),
    }


def _extract_full_exif_metadata(path: Path) -> dict:
    command = [
        "exiftool",
        "-json",
        "-struct",
        "-a",
        "-u",
        "-U",
        "-n",
        "-G1",
        "-api",
        "RequestAll=3",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout or "[]")
    if not payload:
        return {}
    exif = dict(payload[0] or {})
    exif.pop("SourceFile", None)
    return exif


def _sanitize_file_stem(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value.strip())
    cleaned = cleaned.strip("-_")
    return cleaned or "image"


def _store_dl_source_image(raw_bytes: bytes, image_id: str, suffix: str) -> tuple[str, str]:
    dl_image_dir = ROOT_DIR / "dl-lab" / "images"
    dl_image_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{_sanitize_file_stem(image_id)}{suffix or '.jpg'}"
    file_path = dl_image_dir / file_name
    file_path.write_bytes(raw_bytes)
    relative_path = file_path.relative_to(ROOT_DIR)
    return str(file_path), str(relative_path)


def _attach_full_exif_metadata(payload: dict) -> dict:
    original_file = payload.get("originalFile") or payload.get("originalFileHeader") or {}
    base64_value = str(original_file.get("base64") or "")
    if not base64_value:
        return payload

    file_name = str(original_file.get("fileName") or payload.get("fileName") or "upload.jpg")
    mime_type = str(original_file.get("mimeType") or "")
    suffix = Path(file_name).suffix or mimetypes.guess_extension(mime_type) or ".jpg"
    image_metadata = dict(payload.get("image_metadata") or {})
    image_id = str(payload.get("imageId") or payload.get("image_id") or "")
    image_metadata["originalFileName"] = file_name
    image_metadata["originalMimeType"] = mime_type
    image_metadata["originalFileSize"] = int(original_file.get("size") or 0)

    temp_path = None
    try:
        raw_bytes = base64.b64decode(base64_value)
        if image_id and payload.get("originalFile"):
            stored_path, relative_path = _store_dl_source_image(raw_bytes, image_id, suffix)
            image_metadata["dlImagePath"] = stored_path
            image_metadata["dlImageRelativePath"] = relative_path
            image_metadata["dlImageStoredAt"] = datetime.now(timezone.utc).isoformat()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(raw_bytes)
            temp_path = Path(temp_file.name)
        exif = _extract_full_exif_metadata(temp_path)
        image_metadata["exif"] = exif
        image_metadata["exifSummary"] = _build_exif_summary(exif)
        if payload.get("originalFileHeader"):
            image_metadata["exifFromPartialHeader"] = True
    except Exception as error:  # noqa: BLE001
        image_metadata["exifError"] = str(error)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    enriched = dict(payload)
    enriched["image_metadata"] = image_metadata
    enriched.pop("originalFile", None)
    enriched.pop("originalFileHeader", None)
    return enriched


def _normalize_public_review_record(payload: dict) -> dict:
    feedback = payload.get("feedback") or {}
    predicted_scores = payload.get("predicted_scores") or {}
    displayed_scores = payload.get("displayed_scores") or {}
    image_metadata = payload.get("image_metadata") or {}
    review_asset = payload.get("reviewAsset") or {}
    drive_url = str(review_asset.get("driveUrl") or "")
    drive_file_id = str(review_asset.get("driveFileId") or _extract_drive_file_id(drive_url) or "")
    drive_url = str(review_asset.get("driveUrl") or "")
    thumbnail_url = str(review_asset.get("thumbnailUrl") or _build_drive_thumbnail_url(drive_file_id))
    direct_image_url = _build_drive_view_url(drive_file_id)
    proxy_source_url = direct_image_url or thumbnail_url or drive_url
    image_reason = ""
    if not drive_url and not drive_file_id:
        image_reason = "古い公開フィードバックのため画像参照情報がありません"
    elif not proxy_source_url:
        image_reason = "Drive 画像のサムネイルURLを生成できませんでした"
    total_score = displayed_scores.get("totalScore")
    if total_score in ("", None):
        total_score = predicted_scores.get("totalScore")
    return {
        "id": str(payload.get("imageId") or ""),
        "createdAt": str(payload.get("createdAt") or ""),
        "fileName": str(review_asset.get("fileName") or image_metadata.get("fileName") or ""),
        "totalScore": total_score,
        "evaluationMode": str(payload.get("evaluationMode") or ""),
        "modelVersion": str(payload.get("modelVersion") or ""),
        "fairness": str(feedback.get("fairness") or ""),
        "genre": str(feedback.get("genre") or ""),
        "savedAt": str(feedback.get("savedAt") or ""),
        "source": str(payload.get("source") or "public-pages"),
        "driveUrl": drive_url,
        "driveFileId": drive_file_id,
        "thumbnailUrl": thumbnail_url,
        "directImageUrl": direct_image_url,
        "imageProxyUrl": _build_review_image_proxy_url(proxy_source_url),
        "savedToDrive": bool(review_asset.get("savedToDrive") or drive_url or drive_file_id),
        "note": str(feedback.get("note") or ""),
        "hasImageAsset": bool(drive_url or drive_file_id),
        "imageAvailabilityReason": image_reason,
    }


def _normalize_local_review_record(row: dict) -> dict:
    user_feedback = row.get("user_feedback") or {}
    image_metadata = row.get("image_metadata") or {}
    displayed_scores = row.get("displayed_scores") or {}
    predicted_scores = row.get("predicted_scores") or {}
    total_score = row.get("displayed_total_score")
    if total_score in ("", None):
        total_score = displayed_scores.get("totalScore")
    if total_score in ("", None):
        total_score = predicted_scores.get("totalScore")
    return {
        "id": str(row.get("image_id") or ""),
        "createdAt": str(row.get("created_at") or ""),
        "fileName": str(image_metadata.get("fileName") or row.get("image_id") or ""),
        "totalScore": total_score,
        "evaluationMode": str(row.get("evaluation_mode") or ""),
        "modelVersion": str(row.get("model_version") or ""),
        "fairness": str(user_feedback.get("fairness") or row.get("feedback_label") or ""),
        "genre": str(user_feedback.get("genre") or row.get("genre") or ""),
        "savedAt": str(user_feedback.get("savedAt") or row.get("updated_at") or ""),
        "source": "local-db",
        "driveUrl": "",
        "driveFileId": "",
        "thumbnailUrl": "",
        "directImageUrl": "",
        "imageProxyUrl": "",
        "savedToDrive": False,
        "note": str(user_feedback.get("note") or ""),
        "hasImageAsset": False,
        "imageAvailabilityReason": "ローカル学習DBには元画像への参照が保存されていません",
    }


def main() -> None:
    ensure_database()
    server = ThreadingHTTPServer((HOST, PORT), PhotoEvalHandler)
    print(f"Photo Eval ML server running at http://{HOST}:{PORT}/{DEFAULT_ENTRY}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
