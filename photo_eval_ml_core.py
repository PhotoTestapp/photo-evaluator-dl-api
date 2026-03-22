from __future__ import annotations

import csv
import io
import json
import math
import os
import base64
import hashlib
import mimetypes
import re
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("PHOTO_EVAL_DB_PATH") or (ROOT_DIR / "photo_eval_ml.sqlite3"))
MODEL_PATH = Path(os.environ.get("PHOTO_EVAL_MODEL_PATH") or (ROOT_DIR / "photo_eval_model.json"))
MIN_ML_SAMPLES = 30
RECOMMENDED_ML_SAMPLES = 100
ALLOWED_GENRE_CODES = ("portrait", "landscape", "animal", "flora", "food", "other")
MAX_GENRE_SELECTIONS = 2
PREFECTURE_CAPITALS = (
    ("北海道", 43.06417, 141.34694),
    ("青森県", 40.82444, 140.74),
    ("岩手県", 39.70361, 141.1525),
    ("宮城県", 38.26889, 140.87194),
    ("秋田県", 39.71861, 140.1025),
    ("山形県", 38.24056, 140.36333),
    ("福島県", 37.75, 140.46778),
    ("茨城県", 36.34139, 140.44667),
    ("栃木県", 36.56583, 139.88361),
    ("群馬県", 36.39111, 139.06083),
    ("埼玉県", 35.85694, 139.64889),
    ("千葉県", 35.60472, 140.12333),
    ("東京都", 35.68944, 139.69167),
    ("神奈川県", 35.44778, 139.6425),
    ("新潟県", 37.90222, 139.02361),
    ("富山県", 36.69528, 137.21139),
    ("石川県", 36.59444, 136.62556),
    ("福井県", 36.06528, 136.22194),
    ("山梨県", 35.66389, 138.56833),
    ("長野県", 36.65139, 138.18111),
    ("岐阜県", 35.39111, 136.72222),
    ("静岡県", 34.97694, 138.38306),
    ("愛知県", 35.18028, 136.90667),
    ("三重県", 34.73028, 136.50861),
    ("滋賀県", 35.00444, 135.86833),
    ("京都府", 35.02139, 135.75556),
    ("大阪府", 34.68639, 135.52),
    ("兵庫県", 34.69139, 135.18306),
    ("奈良県", 34.68528, 135.83278),
    ("和歌山県", 34.22611, 135.1675),
    ("鳥取県", 35.50361, 134.23833),
    ("島根県", 35.47222, 133.05056),
    ("岡山県", 34.66167, 133.935),
    ("広島県", 34.39639, 132.45944),
    ("山口県", 34.18583, 131.47139),
    ("徳島県", 34.06583, 134.55944),
    ("香川県", 34.34028, 134.04333),
    ("愛媛県", 33.84167, 132.76611),
    ("高知県", 33.55972, 133.53111),
    ("福岡県", 33.60639, 130.41806),
    ("佐賀県", 33.24944, 130.29889),
    ("長崎県", 32.74472, 129.87361),
    ("熊本県", 32.78972, 130.74167),
    ("大分県", 33.23806, 131.6125),
    ("宮崎県", 31.91111, 131.42389),
    ("鹿児島県", 31.56028, 130.55806),
    ("沖縄県", 26.2125, 127.68111),
)
GLOBAL_LOCATION_SEEDS = (
    ("香港", "香港", 22.3193, 114.1694),
    ("イギリス", "ロンドン", 51.5074, -0.1278),
    ("フランス", "パリ", 48.8566, 2.3522),
    ("アメリカ", "ニューヨーク", 40.7128, -74.006),
    ("アメリカ", "ロサンゼルス", 34.0522, -118.2437),
    ("韓国", "ソウル", 37.5665, 126.978),
    ("台湾", "台北", 25.033, 121.5654),
    ("中国", "上海", 31.2304, 121.4737),
    ("タイ", "バンコク", 13.7563, 100.5018),
    ("シンガポール", "シンガポール", 1.3521, 103.8198),
    ("ベトナム", "ホーチミン", 10.8231, 106.6297),
    ("オーストラリア", "シドニー", -33.8688, 151.2093),
    ("ドイツ", "ベルリン", 52.52, 13.405),
    ("イタリア", "ローマ", 41.9028, 12.4964),
    ("スペイン", "マドリード", 40.4168, -3.7038),
    ("カナダ", "トロント", 43.6532, -79.3832),
)

FEATURE_FIELDS = [
    "brightnessMean",
    "brightnessStd",
    "contrast",
    "overexposedRatio",
    "underexposedRatio",
    "entropy",
    "saturationRatio",
    "sharpnessRaw",
    "noiseEstimate",
    "meanEdge",
    "dominantEdge",
    "leftRightBalance",
    "topBottomBalance",
    "thirdsAlignment",
    "subjectConcentration",
    "subjectCenterSupport",
    "subjectXRatio",
    "subjectYRatio",
    "backgroundInformation",
    "backgroundClutter",
    "compositionScore",
    "lightScore",
    "colorScore",
    "technicalScore",
    "subjectScore",
    "impactScore",
    "ruleTotalScore",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_initial_model() -> dict[str, Any]:
    return {
        "model_type": "linear_residual_regression_v2",
        "model_version": "linear-v2-untrained",
        "trained_at": "",
        "sample_count": 0,
        "minimum_ml_samples": MIN_ML_SAMPLES,
        "recommended_ml_samples": RECOMMENDED_ML_SAMPLES,
        "target": "delta = corrected_score - rule_score",
        "feature_names": FEATURE_FIELDS[:],
        "means": [0.0 for _ in FEATURE_FIELDS],
        "scales": [1.0 for _ in FEATURE_FIELDS],
        "weights": [0.0 for _ in FEATURE_FIELDS],
        "bias": 0.0,
        "training_mae": 0.0,
    }


def write_initial_model() -> dict[str, Any]:
    model = build_initial_model()
    MODEL_PATH.write_text(_json_dump(model), encoding="utf-8")
    return model


def ensure_database() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_records (
              image_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              image_metadata_json TEXT NOT NULL,
              feature_vector_json TEXT NOT NULL,
              predicted_scores_json TEXT NOT NULL,
              displayed_scores_json TEXT NOT NULL,
              comments_json TEXT NOT NULL,
              user_feedback_json TEXT NOT NULL,
              corrected_score REAL,
              feedback_label TEXT,
              comment_helpfulness TEXT,
              genre TEXT,
              review_mode TEXT,
              evaluation_mode TEXT,
              model_version TEXT,
              file_fingerprint TEXT,
              rule_score REAL,
              ml_score REAL,
              displayed_total_score REAL,
              score_gap REAL
            )
            """
        )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(feedback_records)").fetchall()
        }
        if "file_fingerprint" not in columns:
            connection.execute("ALTER TABLE feedback_records ADD COLUMN file_fingerprint TEXT")
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_file_fingerprint
            ON feedback_records(file_fingerprint)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_created_at
            ON feedback_records(created_at DESC)
            """
        )
        connection.commit()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_load(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def normalize_genre_value(raw_value: str | None) -> str:
    if raw_value in ("", None):
        return ""

    parts = [part.strip() for part in str(raw_value).split("|") if part.strip()]
    if not parts:
        return ""

    if len(parts) != len(set(parts)):
        raise ValueError("genre に重複した値があります")
    if any(part not in ALLOWED_GENRE_CODES for part in parts):
        raise ValueError("genre に不正なカテゴリコードがあります")
    if "other" in parts and len(parts) > 1:
        raise ValueError("other は単独選択のみ許可されています")
    if len(parts) > MAX_GENRE_SELECTIONS:
        raise ValueError("genre は最大2項目までです")
    return "|".join(parts)


def _pick_exif_value(exif: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in exif and exif[key] not in ("", None):
            return exif[key]
    for key in keys:
        lowered = str(key).lower()
        for existing_key, value in exif.items():
            existing = str(existing_key).lower()
            if existing == lowered or existing.endswith(f":{lowered}"):
                if value not in ("", None):
                    return value
    return None


def _normalize_prefecture_name(raw_value: Any) -> str | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    compact = text.replace(" ", "").replace("　", "")
    for prefecture, _, _ in PREFECTURE_CAPITALS:
        if prefecture in compact:
            return prefecture
    return None


def _haversine_distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2
    )
    return radius * (2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a))))


def _infer_prefecture_from_coordinates(latitude: float | None, longitude: float | None) -> str | None:
    if latitude is None or longitude is None:
        return None
    if not (20.0 <= latitude <= 46.5 and 122.0 <= longitude <= 154.0):
        return None
    nearest = min(
        PREFECTURE_CAPITALS,
        key=lambda item: _haversine_distance_km(latitude, longitude, item[1], item[2]),
    )
    return str(nearest[0])


def _compose_overseas_location_label(country: Any, city: Any) -> str | None:
    country_text = str(country or "").strip()
    city_text = str(city or "").strip()
    if country_text and city_text:
        return f"{country_text} / {city_text}"
    if country_text:
        return country_text
    if city_text:
        return city_text
    return None


def _infer_overseas_location_from_coordinates(latitude: float | None, longitude: float | None) -> tuple[str | None, str | None]:
    if latitude is None or longitude is None:
        return (None, None)
    if 20.0 <= latitude <= 46.5 and 122.0 <= longitude <= 154.0:
        return (None, None)
    nearest_country = None
    nearest_city = None
    nearest_distance = None
    for country, city, seed_lat, seed_lng in GLOBAL_LOCATION_SEEDS:
        distance = _haversine_distance_km(latitude, longitude, seed_lat, seed_lng)
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_country = country
            nearest_city = city
    if nearest_distance is None or nearest_distance > 400:
        return (None, None)
    return (nearest_country, nearest_city)


def _build_exif_summary_from_exif(exif: dict[str, Any]) -> dict[str, Any]:
    latitude = _coerce_float(_pick_exif_value(exif, "GPSLatitude", "Composite:GPSLatitude"))
    longitude = _coerce_float(_pick_exif_value(exif, "GPSLongitude", "Composite:GPSLongitude"))
    altitude = _coerce_float(_pick_exif_value(exif, "GPSAltitude", "Composite:GPSAltitude"))
    country = _pick_exif_value(exif, "Country", "Country-PrimaryLocationName", "IPTCCountry-PrimaryLocationName")
    city = _pick_exif_value(exif, "City", "Sub-location", "Location")
    inferred_country, inferred_city = _infer_overseas_location_from_coordinates(latitude, longitude)
    country = country or inferred_country
    city = city or inferred_city
    prefecture = (
        _normalize_prefecture_name(_pick_exif_value(exif, "State", "Province", "Province-State", "RegionName"))
        or _normalize_prefecture_name(_pick_exif_value(exif, "Sub-location", "City", "Country-PrimaryLocationName"))
        or _infer_prefecture_from_coordinates(latitude, longitude)
    )
    location_label = prefecture or _compose_overseas_location_label(country, city)
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
        "gpsLatitude": latitude,
        "gpsLongitude": longitude,
        "altitudeMeters": altitude,
        "prefecture": prefecture,
        "country": country,
        "city": city,
        "locationLabel": location_label,
    }


def _extract_full_exif_metadata(path: Path) -> dict[str, Any]:
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


def _enrich_payload_with_exif(payload: dict[str, Any]) -> dict[str, Any]:
    image_metadata = dict(payload.get("image_metadata") or {})
    existing_exif = image_metadata.get("exif")
    if isinstance(existing_exif, dict) and existing_exif:
        return payload

    source = payload.get("originalFile") or payload.get("originalFileHeader") or {}
    base64_value = str(source.get("base64") or "")
    if not base64_value:
        return payload

    file_name = str(source.get("fileName") or payload.get("fileName") or image_metadata.get("fileName") or "upload.jpg")
    mime_type = str(source.get("mimeType") or image_metadata.get("mimeType") or "")
    suffix = Path(file_name).suffix or mimetypes.guess_extension(mime_type) or ".jpg"
    image_metadata["originalFileName"] = file_name
    image_metadata["originalMimeType"] = mime_type
    image_metadata["originalFileSize"] = int(source.get("size") or image_metadata.get("fileSize") or 0)

    temp_path: Path | None = None
    try:
        raw_bytes = base64.b64decode(base64_value)
        if payload.get("originalFile"):
            image_metadata["fileFingerprint"] = hashlib.sha256(raw_bytes).hexdigest()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(raw_bytes)
            temp_path = Path(temp_file.name)
        exif = _extract_full_exif_metadata(temp_path)
        if exif:
            image_metadata["exif"] = exif
            image_metadata["exifSummary"] = _build_exif_summary_from_exif(exif)
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


def sanitize_feedback_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _enrich_payload_with_exif(payload)
    feedback = payload.get("feedback") or {}
    predicted_scores = payload.get("predicted_scores") or {}
    displayed_scores = payload.get("displayed_scores") or predicted_scores
    feature_vector = payload.get("features") or {}
    image_metadata = payload.get("image_metadata") or {}
    comments = payload.get("comments") or {}
    created_at = payload.get("createdAt") or now_iso()
    image_id = str(payload.get("imageId") or payload.get("image_id") or "")

    if not image_id:
      raise ValueError("imageId is required")

    corrected_score = feedback.get("correctedScore")
    corrected_score = None if corrected_score in ("", None) else float(corrected_score)
    genre = normalize_genre_value(feedback.get("genre") or "")

    return {
        "image_id": image_id,
        "created_at": created_at,
        "updated_at": now_iso(),
        "image_metadata": image_metadata,
        "feature_vector": feature_vector,
        "predicted_scores": predicted_scores,
        "displayed_scores": displayed_scores,
        "comments": comments,
        "user_feedback": feedback,
        "corrected_score": corrected_score,
        "feedback_label": str(feedback.get("fairness") or ""),
        "comment_helpfulness": str(feedback.get("commentUsefulness") or ""),
        "genre": genre,
        "review_mode": "general",
        "evaluation_mode": str(payload.get("evaluationMode") or "Rule-based"),
        "model_version": str(payload.get("modelVersion") or ""),
        "file_fingerprint": str(image_metadata.get("fileFingerprint") or ""),
        "rule_score": _coerce_float(predicted_scores.get("totalScore")),
        "ml_score": _coerce_float(payload.get("mlScore")),
        "displayed_total_score": _coerce_float(displayed_scores.get("totalScore")),
        "score_gap": _coerce_float(payload.get("scoreGap")),
    }


def save_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_database()
    row = sanitize_feedback_payload(payload)
    source = str(payload.get("source") or "")
    original_image_id = row["image_id"]
    duplicate_of = ""
    with sqlite3.connect(DB_PATH) as connection:
        if source in {"developer-training-lab", "user-photo-recheck"} and row["file_fingerprint"]:
            existing = connection.execute(
                """
                SELECT image_id
                FROM feedback_records
                WHERE file_fingerprint = ?
                LIMIT 1
                """,
                (row["file_fingerprint"],),
            ).fetchone()
            if existing and existing[0]:
                duplicate_of = str(existing[0])
                row["image_id"] = duplicate_of
        connection.execute(
            """
            INSERT INTO feedback_records (
              image_id, created_at, updated_at,
              image_metadata_json, feature_vector_json, predicted_scores_json,
              displayed_scores_json, comments_json, user_feedback_json,
              corrected_score, feedback_label, comment_helpfulness, genre,
              review_mode, evaluation_mode, model_version, file_fingerprint, rule_score,
              ml_score, displayed_total_score, score_gap
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
              updated_at = excluded.updated_at,
              image_metadata_json = excluded.image_metadata_json,
              feature_vector_json = excluded.feature_vector_json,
              predicted_scores_json = excluded.predicted_scores_json,
              displayed_scores_json = excluded.displayed_scores_json,
              comments_json = excluded.comments_json,
              user_feedback_json = excluded.user_feedback_json,
              corrected_score = excluded.corrected_score,
              feedback_label = excluded.feedback_label,
              comment_helpfulness = excluded.comment_helpfulness,
              genre = excluded.genre,
              review_mode = excluded.review_mode,
              evaluation_mode = excluded.evaluation_mode,
              model_version = excluded.model_version,
              file_fingerprint = excluded.file_fingerprint,
              rule_score = excluded.rule_score,
              ml_score = excluded.ml_score,
              displayed_total_score = excluded.displayed_total_score,
              score_gap = excluded.score_gap
            """,
            (
                row["image_id"],
                row["created_at"],
                row["updated_at"],
                _json_dump(row["image_metadata"]),
                _json_dump(row["feature_vector"]),
                _json_dump(row["predicted_scores"]),
                _json_dump(row["displayed_scores"]),
                _json_dump(row["comments"]),
                _json_dump(row["user_feedback"]),
                row["corrected_score"],
                row["feedback_label"],
                row["comment_helpfulness"],
                row["genre"],
                row["review_mode"],
                row["evaluation_mode"],
                row["model_version"],
                row["file_fingerprint"],
                row["rule_score"],
                row["ml_score"],
                row["displayed_total_score"],
                row["score_gap"],
            ),
        )
        connection.commit()
    row["deduplicated"] = bool(duplicate_of and duplicate_of != original_image_id)
    row["duplicate_of"] = duplicate_of
    row["original_image_id"] = original_image_id
    return row


def load_feedback_rows() -> list[dict[str, Any]]:
    ensure_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT *
            FROM feedback_records
            ORDER BY datetime(created_at) DESC
            """
        ).fetchall()

    result = []
    for row in rows:
        result.append(
            {
                "image_id": row["image_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "image_metadata": _json_load(row["image_metadata_json"], {}),
                "feature_vector": _json_load(row["feature_vector_json"], {}),
                "predicted_scores": _json_load(row["predicted_scores_json"], {}),
                "displayed_scores": _json_load(row["displayed_scores_json"], {}),
                "comments": _json_load(row["comments_json"], {}),
                "user_feedback": _json_load(row["user_feedback_json"], {}),
                "corrected_score": row["corrected_score"],
                "feedback_label": row["feedback_label"] or "",
                "comment_helpfulness": row["comment_helpfulness"] or "",
                "genre": row["genre"] or "",
                "review_mode": row["review_mode"] or "",
                "evaluation_mode": row["evaluation_mode"] or "Rule-based",
                "model_version": row["model_version"] or "",
                "file_fingerprint": row["file_fingerprint"] or "",
                "rule_score": row["rule_score"],
                "ml_score": row["ml_score"],
                "displayed_total_score": row["displayed_total_score"],
                "score_gap": row["score_gap"],
            }
        )
    return result


def export_feedback_records(fmt: str) -> tuple[str, str]:
    records = load_feedback_rows()
    dataset = [build_export_record(record) for record in records]

    if fmt == "json":
        return _json_dump(dataset), "application/json; charset=utf-8"

    output = io.StringIO()
    fieldnames = [
        "image_id",
        "created_at",
        "feedback_label",
        "comment_helpfulness",
        "corrected_score",
        "genre",
        "review_mode",
        "evaluation_mode",
        "model_version",
        "rule_score",
        "ml_score",
        "displayed_total_score",
        "score_gap",
        "features_json",
        "predicted_scores_json",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in dataset:
        writer.writerow(
            {
                "image_id": row["image_id"],
                "created_at": row["created_at"],
                "feedback_label": row["feedback_label"],
                "comment_helpfulness": row["comment_helpfulness"],
                "corrected_score": row["corrected_score"],
                "genre": row["genre"],
                "review_mode": row["review_mode"],
                "evaluation_mode": row["evaluation_mode"],
                "model_version": row["model_version"],
                "rule_score": row["predicted_scores"].get("totalScore"),
                "ml_score": row["ml_score"],
                "displayed_total_score": row["displayed_scores"].get("totalScore"),
                "score_gap": row["score_gap"],
                "features_json": _json_dump(row["features"]),
                "predicted_scores_json": _json_dump(row["predicted_scores"]),
            }
        )
    return output.getvalue(), "text/csv; charset=utf-8"


def build_export_record(record: dict[str, Any]) -> dict[str, Any]:
    corrected_score = derive_corrected_score(record)
    return {
        "image_id": record["image_id"],
        "created_at": record["created_at"],
        "features": normalize_feature_vector(record["feature_vector"], record["predicted_scores"]),
        "predicted_scores": record["predicted_scores"],
        "displayed_scores": record["displayed_scores"],
        "corrected_score": corrected_score,
        "feedback_label": record["feedback_label"],
        "comment_helpfulness": record["comment_helpfulness"],
        "genre": record["genre"],
        "review_mode": record["review_mode"],
        "evaluation_mode": record["evaluation_mode"],
        "model_version": record["model_version"],
        "ml_score": record["ml_score"],
        "score_gap": record["score_gap"],
    }


def derive_corrected_score(record: dict[str, Any]) -> float | None:
    if record["corrected_score"] is not None:
        return float(record["corrected_score"])

    base_score = _coerce_float(record["predicted_scores"].get("totalScore"))
    if base_score is None:
        return None

    fairness = record["feedback_label"]
    if fairness == "too_high":
        return max(0.0, base_score - 8.0)
    if fairness == "too_low":
        return min(100.0, base_score + 8.0)
    if fairness == "valid":
        return base_score
    return None


def normalize_feature_vector(feature_vector: dict[str, Any], predicted_scores: dict[str, Any]) -> dict[str, float]:
    merged = {**feature_vector, **predicted_scores}
    merged["ruleTotalScore"] = predicted_scores.get("totalScore")
    merged["impactScore"] = (
        predicted_scores.get("impactScore")
        or predicted_scores.get("impressionScore")
        or merged.get("impactScore")
        or 0.0
    )
    merged["subjectCenterSupport"] = feature_vector.get("subjectCenterSupport") or merged.get("subjectCenterSupport") or 0.0
    normalized = {}
    for key in FEATURE_FIELDS:
        normalized[key] = _coerce_float(merged.get(key), default=0.0) or 0.0
    return normalized


def build_training_rows() -> list[dict[str, Any]]:
    rows = []
    for record in load_feedback_rows():
        corrected_score = derive_corrected_score(record)
        rule_score = _coerce_float(record["predicted_scores"].get("totalScore"))
        if corrected_score is None or rule_score is None:
            continue
        features = normalize_feature_vector(record["feature_vector"], record["predicted_scores"])
        rows.append(
            {
                "image_id": record["image_id"],
                "features": features,
                "target_residual": corrected_score - rule_score,
                "corrected_score": corrected_score,
                "rule_score": rule_score,
            }
        )
    return rows


def train_linear_residual_model() -> dict[str, Any]:
    rows = build_training_rows()
    if len(rows) < 4:
        raise ValueError("学習データが不足しています。最低4件以上必要です。")

    feature_names = FEATURE_FIELDS[:]
    feature_matrix = [[row["features"].get(name, 0.0) for name in feature_names] for row in rows]
    targets = [row["target_residual"] for row in rows]

    means = []
    scales = []
    normalized_matrix = []
    for column_index, feature_name in enumerate(feature_names):
        column = [vector[column_index] for vector in feature_matrix]
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        scale = math.sqrt(variance) or 1.0
        means.append(mean)
        scales.append(scale)

    for vector in feature_matrix:
        normalized_matrix.append(
            [(value - means[index]) / scales[index] for index, value in enumerate(vector)]
        )

    weights = [0.0 for _ in feature_names]
    bias = 0.0
    learning_rate = 0.03
    l2_penalty = 0.015
    sample_count = len(normalized_matrix)

    for _ in range(2200):
        predictions = [
            bias + sum(weight * value for weight, value in zip(weights, vector))
            for vector in normalized_matrix
        ]
        residuals = [prediction - target for prediction, target in zip(predictions, targets)]
        bias_gradient = (2.0 / sample_count) * sum(residuals)
        weight_gradients = []
        for column_index in range(len(feature_names)):
            gradient = (2.0 / sample_count) * sum(
                residuals[row_index] * normalized_matrix[row_index][column_index]
                for row_index in range(sample_count)
            )
            gradient += 2.0 * l2_penalty * weights[column_index]
            weight_gradients.append(gradient)

        bias -= learning_rate * bias_gradient
        weights = [
            weight - learning_rate * gradient
            for weight, gradient in zip(weights, weight_gradients)
        ]

    mae = sum(
        abs(
            (bias + sum(weight * value for weight, value in zip(weights, vector))) - target
        )
        for vector, target in zip(normalized_matrix, targets)
    ) / sample_count

    model = {
        "model_type": "linear_residual_regression_v2",
        "model_version": f"linear-v2-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "trained_at": now_iso(),
        "sample_count": sample_count,
        "minimum_ml_samples": MIN_ML_SAMPLES,
        "recommended_ml_samples": RECOMMENDED_ML_SAMPLES,
        "target": "delta = corrected_score - rule_score",
        "feature_names": feature_names,
        "means": means,
        "scales": scales,
        "weights": weights,
        "bias": bias,
        "training_mae": mae,
    }
    MODEL_PATH.write_text(_json_dump(model), encoding="utf-8")
    return model


def load_model() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    try:
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        if model.get("model_type") != "linear_residual_regression_v2":
            return None
        return model
    except json.JSONDecodeError:
        return None


def get_model_status() -> dict[str, Any]:
    model = load_model()
    feedback_count = len(load_feedback_rows())
    if not model:
        return {
            "available": False,
            "mode": "Rule-based",
            "modelVersion": "",
            "sampleCount": 0,
            "feedbackCount": feedback_count,
            "minimumMlSamples": MIN_ML_SAMPLES,
            "recommendedMlSamples": RECOMMENDED_ML_SAMPLES,
            "reason": "学習済みモデルがありません",
        }

    sample_count = int(model.get("sample_count") or 0)
    available = sample_count >= int(model.get("minimum_ml_samples") or MIN_ML_SAMPLES)
    return {
        "available": available,
        "mode": "ML-assisted" if available else "Rule-based",
        "modelVersion": model.get("model_version") or "",
        "sampleCount": sample_count,
        "feedbackCount": feedback_count,
        "minimumMlSamples": int(model.get("minimum_ml_samples") or MIN_ML_SAMPLES),
        "recommendedMlSamples": int(model.get("recommended_ml_samples") or RECOMMENDED_ML_SAMPLES),
        "trainingMae": round(float(model.get("training_mae") or 0.0), 4),
        "reason": "" if available else "学習データ数が30件に達していないため Rule-based にフォールバックしています",
    }


def predict_total_score(feature_vector: dict[str, Any], predicted_scores: dict[str, Any]) -> dict[str, Any]:
    status = get_model_status()
    rule_score = _coerce_float(predicted_scores.get("totalScore"), default=0.0) or 0.0
    if not status["available"]:
        return {
            "mode": "Rule-based",
            "modelVersion": status["modelVersion"],
            "sampleCount": status["sampleCount"],
            "minimumMlSamples": status["minimumMlSamples"],
            "recommendedMlSamples": status.get("recommendedMlSamples", RECOMMENDED_ML_SAMPLES),
            "predictedTotalScore": rule_score,
            "ruleScore": rule_score,
            "predictedDelta": 0.0,
            "residualApplied": 0.0,
            "usedModel": False,
            "fallbackReason": status["reason"],
        }

    model = load_model()
    if not model:
        return {
            "mode": "Rule-based",
            "modelVersion": "",
            "sampleCount": 0,
            "minimumMlSamples": MIN_ML_SAMPLES,
            "recommendedMlSamples": RECOMMENDED_ML_SAMPLES,
            "predictedTotalScore": rule_score,
            "ruleScore": rule_score,
            "predictedDelta": 0.0,
            "residualApplied": 0.0,
            "usedModel": False,
            "fallbackReason": "モデル読込に失敗しました",
        }

    features = normalize_feature_vector(feature_vector, predicted_scores)
    feature_names = model.get("feature_names") or FEATURE_FIELDS
    means = model.get("means") or [0.0 for _ in feature_names]
    scales = model.get("scales") or [1.0 for _ in feature_names]
    weights = model.get("weights") or [0.0 for _ in feature_names]
    bias = float(model.get("bias") or 0.0)

    normalized_values = []
    for index, name in enumerate(feature_names):
        scale = float(scales[index] or 1.0)
        value = _coerce_float(features.get(name), default=0.0) or 0.0
        normalized_values.append((value - float(means[index] or 0.0)) / scale)

    residual = bias + sum(weight * value for weight, value in zip(weights, normalized_values))
    residual = max(-25.0, min(25.0, residual))
    predicted_total = max(0.0, min(100.0, rule_score + residual))

    return {
        "mode": "ML-assisted",
        "modelVersion": model.get("model_version") or "",
        "sampleCount": int(model.get("sample_count") or 0),
        "minimumMlSamples": int(model.get("minimum_ml_samples") or MIN_ML_SAMPLES),
        "recommendedMlSamples": int(model.get("recommended_ml_samples") or RECOMMENDED_ML_SAMPLES),
        "predictedTotalScore": round(predicted_total, 2),
        "ruleScore": round(rule_score, 2),
        "predictedDelta": round(residual, 2),
        "residualApplied": round(residual, 2),
        "usedModel": True,
        "fallbackReason": "",
    }


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _find_exif_value(exif: dict[str, Any], *keys: str) -> Any:
    if not isinstance(exif, dict):
        return None
    for key in keys:
        if key in exif and exif[key] not in ("", None):
            return exif[key]
    lowered = {str(key).lower(): key for key in exif.keys()}
    for key in keys:
        lower_key = str(key).lower()
        for existing_lower, existing_key in lowered.items():
            if existing_lower == lower_key or existing_lower.endswith(f":{lower_key}"):
                value = exif.get(existing_key)
                if value not in ("", None):
                    return value
    return None


def _parse_capture_datetime(value: Any) -> datetime | None:
    if value in ("", None):
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _parse_exposure_seconds(value: Any) -> float | None:
    numeric = _coerce_float(value)
    if numeric is not None:
        return numeric if numeric > 0 else None
    text = str(value or "").strip()
    if not text:
        return None
    if "/" in text:
        left, right = text.split("/", 1)
        left_num = _coerce_float(left)
        right_num = _coerce_float(right)
        if left_num is not None and right_num not in (None, 0):
            return left_num / right_num
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return _coerce_float(match.group(1))
    return None


def _build_exif_summary(image_metadata: dict[str, Any]) -> dict[str, Any]:
    summary = dict((image_metadata or {}).get("exifSummary") or {})
    exif = (image_metadata or {}).get("exif") or {}
    if not summary and not exif:
        return {}

    def pick(*keys: str) -> Any:
        if summary:
            for key in keys:
                if summary.get(key) not in ("", None):
                    return summary.get(key)
        return _find_exif_value(exif, *keys)

    capture_at = pick("dateTimeOriginal", "SubSecDateTimeOriginal", "EXIF:DateTimeOriginal", "QuickTime:CreateDate", "XMP:DateCreated")
    iso_value = pick("iso", "ISO", "EXIF:ISO")
    aperture_value = pick("fNumber", "FNumber", "EXIF:FNumber", "Composite:Aperture")
    shutter_value = pick("exposureTime", "ExposureTime", "EXIF:ExposureTime", "Composite:ShutterSpeed")
    focal_length = pick("focalLength", "FocalLength", "EXIF:FocalLength", "Composite:FocalLength")
    focal_length_35 = pick("focalLength35mm", "FocalLengthIn35mmFormat", "EXIF:FocalLengthIn35mmFormat", "Composite:FocalLength35efl")
    latitude = _coerce_float(pick("gpsLatitude", "GPSLatitude", "Composite:GPSLatitude", "GPS:GPSLatitude"))
    longitude = _coerce_float(pick("gpsLongitude", "GPSLongitude", "Composite:GPSLongitude", "GPS:GPSLongitude"))
    altitude = _coerce_float(pick("altitudeMeters", "GPSAltitude", "Composite:GPSAltitude", "GPS:GPSAltitude"))
    country = pick("country", "Country", "Country-PrimaryLocationName", "IPTC:Country-PrimaryLocationName", "QuickTime:Country")
    city = pick("city", "City", "Sub-location", "Location")
    inferred_country, inferred_city = _infer_overseas_location_from_coordinates(latitude, longitude)
    country = country or inferred_country
    city = city or inferred_city
    prefecture = (
        _normalize_prefecture_name(pick("prefecture", "State", "Province", "Province-State", "RegionName"))
        or _normalize_prefecture_name(pick("City", "Sub-location", "Country-PrimaryLocationName"))
        or _infer_prefecture_from_coordinates(latitude, longitude)
    )
    location_label = prefecture or _compose_overseas_location_label(country, city)
    return {
        "cameraMake": pick("cameraMake", "Make", "EXIF:Make"),
        "cameraModel": pick("cameraModel", "Model", "EXIF:Model", "QuickTime:Model"),
        "lensModel": pick("lensModel", "LensModel", "EXIF:LensModel", "Composite:LensID"),
        "iso": _coerce_float(iso_value) if iso_value not in ("", None) else None,
        "aperture": _coerce_float(aperture_value) if aperture_value not in ("", None) else None,
        "exposureTime": shutter_value,
        "exposureSeconds": _parse_exposure_seconds(shutter_value),
        "focalLength": _coerce_float(focal_length) if focal_length not in ("", None) else None,
        "focalLength35mm": _coerce_float(focal_length_35) if focal_length_35 not in ("", None) else None,
        "dateTimeOriginal": capture_at,
        "captureHour": _parse_capture_datetime(capture_at).hour if _parse_capture_datetime(capture_at) else None,
        "gpsLatitude": latitude,
        "gpsLongitude": longitude,
        "altitudeMeters": altitude,
        "prefecture": prefecture,
        "country": country,
        "city": city,
        "locationLabel": location_label,
    }


def build_feedback_statistics() -> dict[str, Any]:
    rows = load_feedback_rows()
    totals = []
    fairness_counts = {"valid": 0, "too_high": 0, "too_low": 0, "blank": 0}
    genre_counts = {code: 0 for code in ALLOWED_GENRE_CODES}
    item_keys = {
        "composition": "compositionScore",
        "light": "lightScore",
        "color": "colorScore",
        "technical": "technicalScore",
        "subject": "subjectScore",
        "impact": "impressionScore",
    }
    item_values: dict[str, list[float]] = {key: [] for key in item_keys}
    histogram = [{"label": f"{start}-{min(start + 9, 100)}", "min": start, "max": min(start + 9, 100), "count": 0} for start in range(0, 100, 10)]
    hour_buckets = [{"label": f"{hour:02d}", "count": 0} for hour in range(24)]
    camera_counts: dict[str, int] = {}
    lens_counts: dict[str, int] = {}
    location_counts: dict[str, int] = {}
    iso_labels = ["100以下", "101-400", "401-800", "801-1600", "1601-3200", "3201以上"]
    shutter_labels = ["1/1000以下", "1/500-1/999", "1/125-1/499", "1/30-1/124", "1/8-1/29", "1/8より遅い"]
    heatmap = [[0 for _ in shutter_labels] for _ in iso_labels]
    altitude_buckets = [
        {"label": "海面下", "min": None, "max": 0, "count": 0},
        {"label": "0-49m", "min": 0, "max": 50, "count": 0},
        {"label": "50-99m", "min": 50, "max": 100, "count": 0},
        {"label": "100-199m", "min": 100, "max": 200, "count": 0},
        {"label": "200-499m", "min": 200, "max": 500, "count": 0},
        {"label": "500m以上", "min": 500, "max": None, "count": 0},
    ]
    exif_count = 0
    geo_count = 0
    altitude_count = 0

    for row in rows:
        displayed_scores = row.get("displayed_scores") or {}
        total_score = _coerce_float(displayed_scores.get("totalScore"), _coerce_float(row.get("displayed_total_score")))
        if total_score is not None:
            totals.append(total_score)
            bucket_index = min(9, max(0, int(total_score // 10)))
            histogram[bucket_index]["count"] += 1

        fairness = row.get("feedback_label") or ""
        fairness_counts[fairness if fairness in fairness_counts else "blank"] += 1

        for genre in normalize_genre_value(row.get("genre") or "").split("|"):
            if genre:
                genre_counts[genre] += 1

        for key, score_key in item_keys.items():
            score = _coerce_float(displayed_scores.get(score_key))
            if score is not None:
                item_values[key].append(score)

        exif_summary = _build_exif_summary(row.get("image_metadata") or {})
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

        iso_value = _coerce_float(exif_summary.get("iso"))
        exposure_seconds = _coerce_float(exif_summary.get("exposureSeconds"))
        if iso_value is not None and exposure_seconds is not None:
            if iso_value <= 100:
                iso_index = 0
            elif iso_value <= 400:
                iso_index = 1
            elif iso_value <= 800:
                iso_index = 2
            elif iso_value <= 1600:
                iso_index = 3
            elif iso_value <= 3200:
                iso_index = 4
            else:
                iso_index = 5

            if exposure_seconds <= 1 / 1000:
                shutter_index = 0
            elif exposure_seconds <= 1 / 500:
                shutter_index = 1
            elif exposure_seconds <= 1 / 125:
                shutter_index = 2
            elif exposure_seconds <= 1 / 30:
                shutter_index = 3
            elif exposure_seconds <= 1 / 8:
                shutter_index = 4
            else:
                shutter_index = 5
            heatmap[iso_index][shutter_index] += 1

    histogram_max = max((bucket["count"] for bucket in histogram), default=0)
    total_count = len(rows)
    mean_total = round(_mean(totals), 2) if totals else 0.0
    item_averages = {
        key: round(_mean(values), 2) if values else 0.0
        for key, values in item_values.items()
    }
    top_cameras = [
        {"label": label, "count": count}
        for label, count in sorted(camera_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    top_lenses = [
        {"label": label, "count": count}
        for label, count in sorted(lens_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    top_locations = [
        {"label": label, "count": count}
        for label, count in sorted(location_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    altitude_histogram_max = max((bucket["count"] for bucket in altitude_buckets), default=0)

    return {
        "summary": {
            "totalRecords": total_count,
            "exifRecords": exif_count,
            "exifCoverageRate": round((exif_count / total_count) * 100, 1) if total_count else 0.0,
            "averageTotalScore": mean_total,
            "mlSampleCount": int((load_model() or {}).get("sample_count") or 0),
            "geoRecords": geo_count,
            "altitudeRecords": altitude_count,
        },
        "scoreHistogram": {
            "maxCount": histogram_max,
            "buckets": histogram,
        },
        "genreCounts": [{"label": code, "count": genre_counts[code]} for code in ALLOWED_GENRE_CODES],
        "fairnessCounts": [{"label": label, "count": fairness_counts[label]} for label in ("valid", "too_high", "too_low", "blank")],
        "itemAverages": [{"label": key, "value": item_averages[key]} for key in ("composition", "light", "color", "technical", "subject", "impact")],
        "captureHourHistogram": hour_buckets,
        "topCameras": top_cameras,
        "topLenses": top_lenses,
        "locationCounts": top_locations,
        "altitudeHistogram": {
            "maxCount": altitude_histogram_max,
            "buckets": altitude_buckets,
        },
        "isoShutterHeatmap": {
            "xLabels": shutter_labels,
            "yLabels": iso_labels,
            "values": heatmap,
            "maxValue": max((value for row_values in heatmap for value in row_values), default=0),
        },
    }
