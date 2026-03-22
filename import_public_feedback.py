from __future__ import annotations

import json
import sys
import urllib.request

from photo_eval_ml_core import save_feedback


DEFAULT_EXPORT_URL = (
    "https://script.google.com/macros/s/"
    "AKfycbyT_-5dbbSbPeWu7nKDG7JWN-lzx_PgXbY-7NWy0GlelRRDrec3WUWcF49_txNrqR9B/exec"
    "?action=export"
)


def fetch_records(url: str) -> list[dict]:
    with urllib.request.urlopen(url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("success"):
        raise ValueError(payload.get("message") or "collector export failed")
    return payload.get("records") or []


def import_public_feedback(url: str = DEFAULT_EXPORT_URL) -> dict[str, object]:
    records = fetch_records(url)
    imported = 0
    skipped = 0
    for record in records:
        try:
            save_feedback(record)
            imported += 1
        except ValueError as error:
            skipped += 1
            print(f"Skipped record: {error}")
    return {
        "source_url": url,
        "imported_count": imported,
        "skipped_count": skipped,
    }


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXPORT_URL
    result = import_public_feedback(url)
    print(f"Imported {result['imported_count']} public feedback records.")
    print(f"Skipped {result['skipped_count']} invalid records.")


if __name__ == "__main__":
    main()
