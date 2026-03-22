from __future__ import annotations

from _root_module_loader import APP_DIR, load_root_module


ROOT_DIR = APP_DIR
DB_PATH = ROOT_DIR / "photo_eval_ml.sqlite3"
MODEL_PATH = ROOT_DIR / "photo_eval_model.json"

_shared_core = load_root_module("photo_eval_ml_core.py", "_shared_photo_eval_ml_core")

for _name in dir(_shared_core):
    if _name.startswith("_") or _name in {"ROOT_DIR", "DB_PATH", "MODEL_PATH"}:
        continue
    globals()[_name] = getattr(_shared_core, _name)
