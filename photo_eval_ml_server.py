from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import os
import secrets
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from import_public_feedback import DEFAULT_EXPORT_URL, fetch_records
from photo_eval_ml_core import (
    ROOT_DIR,
    ensure_database,
    export_feedback_records,
    get_model_status,
    load_feedback_rows,
    predict_total_score,
    save_feedback,
)


HOST = os.environ.get("PHOTO_EVAL_ML_HOST", "127.0.0.1")
PORT = int(os.environ.get("PHOTO_EVAL_ML_PORT", "8787"))
DEFAULT_ENTRY = "local-main-app/photo-evaluator-pro-delivery-webhook-set.html"
DL_LAB_DIR = ROOT_DIR / "photo-evaluator-training-lab-app" / "dl-lab"
DL_MODEL_PATH = DL_LAB_DIR / "models" / "dl_residual_model.pt"
DL_MODEL_META_PATH = DL_LAB_DIR / "models" / "dl_residual_model_meta.json"
DL_RUNTIME_CACHE: dict[str, object] = {
    "mtime": None,
    "model": None,
    "transform": None,
    "metadata": None,
    "torch": None,
    "Image": None,
}
SNS_ACCOUNTS_PATH = ROOT_DIR / "SNS" / "sns_accounts.json"


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
        if parsed.path in {"/developer-review", "/developer-review/"}:
            self.path = "/developer-review.html"
            return super().do_GET()
        if parsed.path in {"/developer-review-app", "/developer-review-app/"}:
            self.path = "/developer-review.html"
            return super().do_GET()
        if parsed.path == "/developer-review.html":
            self.path = "/developer-review.html"
            return super().do_GET()
        if parsed.path == "/developer-review-app/index.html":
            self.path = "/developer-review.html"
            return super().do_GET()
        if parsed.path == "/api/health":
            return self._write_json({"success": True, "message": "ok"})
        if parsed.path == "/api/ml/status":
            return self._write_json({"success": True, **get_model_status()})
        if parsed.path == "/api/dl/status":
            return self._write_json({"success": True, **_get_dl_status()})
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

        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/ml/predict":
            return self._handle_predict()
        if parsed.path == "/api/dl/predict":
            return self._handle_dl_predict()
        if parsed.path == "/api/ml/feedback":
            return self._handle_feedback()
        if parsed.path == "/api/sns/register":
            return self._handle_sns_register()
        if parsed.path == "/api/sns/login":
            return self._handle_sns_login()
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
        prediction = predict_total_score(
            payload.get("features") or {},
            payload.get("predicted_scores") or {},
        )
        self._write_json({"success": True, **prediction})

    def _handle_dl_predict(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        image_data_url = str(payload.get("imageDataUrl") or "")
        try:
            rule_score = float(payload.get("ruleScore"))
        except (TypeError, ValueError):
            return self._write_json(
                {"success": False, "message": "ruleScore が不正です"},
                status=HTTPStatus.BAD_REQUEST,
            )
        prediction = _predict_with_dl_model(image_data_url, rule_score)
        self._write_json({"success": True, **prediction})

    def _handle_feedback(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            saved = save_feedback(payload)
        except ValueError as error:
            return self._write_json(
                {"success": False, "message": str(error)},
                status=HTTPStatus.BAD_REQUEST,
            )
        self._write_json(
            {
                "success": True,
                "message": "フィードバックを保存しました",
                "imageId": saved["image_id"],
                "savedAt": saved["updated_at"],
            }
        )

    def _handle_sns_register(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        email = str(payload.get("email") or "").strip().lower()
        password = str(payload.get("password") or "")
        profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
        if not email or "@" not in email:
            return self._write_json(
                {"success": False, "message": "メールアドレスが不正です"},
                status=HTTPStatus.BAD_REQUEST,
            )
        if len(password) < 8:
            return self._write_json(
                {"success": False, "message": "パスワードは8文字以上にしてください"},
                status=HTTPStatus.BAD_REQUEST,
            )
        accounts = _load_sns_accounts()
        existing = next((item for item in accounts if str(item.get("email") or "").lower() == email), None)
        normalized_profile = _normalize_sns_profile(profile)
        if existing:
            if not _verify_password(password, str(existing.get("passwordSalt") or ""), str(existing.get("passwordHash") or "")):
                return self._write_json(
                    {"success": False, "message": "このメールアドレスはすでに登録されています"},
                    status=HTTPStatus.CONFLICT,
                )
            updated = {
                **existing,
                "profile": normalized_profile,
            }
            _save_sns_accounts([updated if item.get("id") == updated["id"] else item for item in accounts])
            return self._write_json({
                "success": True,
                "mode": "updated",
                "account": _public_sns_account(updated),
            })

        salt, password_hash = _hash_password(password)
        account = {
            "id": f"acct_{secrets.token_hex(8)}",
            "email": email,
            "passwordSalt": salt,
            "passwordHash": password_hash,
            "profile": normalized_profile,
        }
        _save_sns_accounts([*accounts, account])
        return self._write_json({
            "success": True,
            "mode": "created",
            "account": _public_sns_account(account),
        })

    def _handle_sns_login(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        email = str(payload.get("email") or "").strip().lower()
        password = str(payload.get("password") or "")
        accounts = _load_sns_accounts()
        account = next((item for item in accounts if str(item.get("email") or "").lower() == email), None)
        if account is None or not _verify_password(password, str(account.get("passwordSalt") or ""), str(account.get("passwordHash") or "")):
            return self._write_json(
                {"success": False, "message": "メールアドレスまたはパスワードが一致しません"},
                status=HTTPStatus.UNAUTHORIZED,
            )
        return self._write_json({
            "success": True,
            "account": _public_sns_account(account),
        })

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


def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_sns_accounts() -> list[dict]:
    try:
        parsed = json.loads(SNS_ACCOUNTS_PATH.read_text(encoding="utf-8")) if SNS_ACCOUNTS_PATH.exists() else []
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _save_sns_accounts(accounts: list[dict]) -> None:
    SNS_ACCOUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNS_ACCOUNTS_PATH.write_text(
        json.dumps(accounts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_sns_profile(profile: dict) -> dict[str, str]:
    display_name = str(profile.get("displayName") or "").strip() or "Pulse User"
    handle = str(profile.get("handle") or "").strip().replace(" ", "")
    if not handle:
        handle = "@pulse"
    if not handle.startswith("@"):
        handle = f"@{handle}"
    location = str(profile.get("location") or "").strip() or "Japan"
    bio = str(profile.get("bio") or "").strip() or "Pulse に参加しました。"
    avatar_src = str(profile.get("avatarSrc") or "")
    return {
        "displayName": display_name,
        "handle": handle,
        "location": location,
        "bio": bio,
        "avatarSrc": avatar_src,
    }


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    actual_salt = salt or secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(actual_salt),
        200_000,
    ).hex()
    return actual_salt, password_hash


def _verify_password(password: str, salt: str, password_hash: str) -> bool:
    if not salt or not password_hash:
        return False
    _salt, computed_hash = _hash_password(password, salt)
    return secrets.compare_digest(computed_hash, password_hash)


def _public_sns_account(account: dict) -> dict[str, object]:
    return {
        "id": str(account.get("id") or ""),
        "email": str(account.get("email") or ""),
        "profile": _normalize_sns_profile(account.get("profile") if isinstance(account.get("profile"), dict) else {}),
    }


def _get_dl_status() -> dict[str, object]:
    metadata = _read_json_file(DL_MODEL_META_PATH) if DL_MODEL_META_PATH.exists() else {}
    available = DL_MODEL_PATH.exists() and DL_MODEL_META_PATH.exists()
    reason = "" if available else "DLモデルがまだ学習されていません"
    return {
        "available": available,
        "modelPath": str(DL_MODEL_PATH),
        "metadataPath": str(DL_MODEL_META_PATH),
        "modelType": str(metadata.get("model_type") or ""),
        "sampleCount": int(metadata.get("sample_count") or 0),
        "trainCount": int(metadata.get("train_count") or 0),
        "validationCount": int(metadata.get("validation_count") or 0),
        "validationMae": metadata.get("validation_mae"),
        "validationMaeByOutput": metadata.get("validation_mae_by_output") or {},
        "validationGenreAccuracy": metadata.get("validation_genre_accuracy"),
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


def _build_drive_thumbnail_url(file_id: str) -> str:
    return f"https://drive.google.com/thumbnail?id={quote(file_id)}&sz=w1200" if file_id else ""


def _normalize_public_review_record(payload: dict) -> dict:
    feedback = payload.get("feedback") or {}
    predicted_scores = payload.get("predicted_scores") or {}
    displayed_scores = payload.get("displayed_scores") or {}
    image_metadata = payload.get("image_metadata") or {}
    review_asset = payload.get("reviewAsset") or {}
    drive_file_id = str(review_asset.get("driveFileId") or "")
    drive_url = str(review_asset.get("driveUrl") or "")
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
        "thumbnailUrl": str(review_asset.get("thumbnailUrl") or _build_drive_thumbnail_url(drive_file_id)),
        "savedToDrive": bool(review_asset.get("savedToDrive") or drive_url or drive_file_id),
        "note": str(feedback.get("note") or ""),
        "hasImageAsset": bool(drive_url or drive_file_id),
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
        "savedToDrive": False,
        "note": str(user_feedback.get("note") or ""),
        "hasImageAsset": False,
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
