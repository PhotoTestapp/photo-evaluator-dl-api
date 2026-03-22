from __future__ import annotations

from _root_module_loader import load_root_module


_shared_importer = load_root_module("import_public_feedback.py", "_shared_import_public_feedback")

DEFAULT_EXPORT_URL = _shared_importer.DEFAULT_EXPORT_URL
fetch_records = _shared_importer.fetch_records
import_public_feedback = _shared_importer.import_public_feedback
main = _shared_importer.main


if __name__ == "__main__":
    main()
