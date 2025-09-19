"""Pipeline wrapper.

For now, this module reuses the robust pipeline implemented in `aimeta.py`.
Future refactors can migrate the functions here fully.
"""
from __future__ import annotations
from pathlib import Path
import aimeta  # reuse existing implementation

# Re-export selected helpers for convenience
parse_exts = aimeta.parse_exts
scan_images = aimeta.scan_images
write_filelists = aimeta.write_filelists
run_exiftool_batches = aimeta.run_exiftool_batches
stream_entries = aimeta.stream_entries
process_entry = aimeta.process_entry
fast_dump_bytes = aimeta.fast_dump_bytes
DEFAULT_EXTS = getattr(aimeta, "DEFAULT_EXTS", "jpg,jpeg,png")
DEFAULT_EXECUTOR = getattr(aimeta, "DEFAULT_EXECUTOR", "thread")
DEFAULT_EXIF_WORKERS = getattr(aimeta, "DEFAULT_EXIF_WORKERS", 4)
PROGRESS_EVERY = getattr(aimeta, "PROGRESS_EVERY", 2000)

# Expose additional defaults and helpers used by the unified UI
DEFAULT_INPUT_DIR = getattr(aimeta, "DEFAULT_INPUT_DIR", r"D:/AI")
DEFAULT_PARSE_WORKERS = getattr(aimeta, "DEFAULT_PARSE_WORKERS", 4)

# Compute default boxed output locations relative to this file.
# Layout:
# Dumby/
#   Run_AI_Toolkit.bat
#   Boxxed_Data/
#     01_Batches/         (filelist_*.txt, metadata_batch_*.json)
#     02_NDJSON/          (structured_metadata*.ndjson)
#     03_PerFile_JSON/    (*.json per image if enabled)
#     04_CSV/             (exports)
#     05_RenameMap/       (map files)
_DUMBY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOX_DIR = _DUMBY_ROOT / "Boxxed_Data"
DEFAULT_BATCHES_DIR = DEFAULT_BOX_DIR / "01_Batches"
DEFAULT_NDJSON_DIR = DEFAULT_BOX_DIR / "02_NDJSON"
DEFAULT_PERFILE_DIR = DEFAULT_BOX_DIR / "03_PerFile_JSON"
DEFAULT_CSV_DIR = DEFAULT_BOX_DIR / "04_CSV"
DEFAULT_RENAMEMAP_DIR = DEFAULT_BOX_DIR / "05_RenameMap"
DEFAULT_OUT_PATH = str(DEFAULT_NDJSON_DIR / "structured_metadata.ndjson")
DEFAULT_WORK_DIR = str(DEFAULT_BOX_DIR)  # root for outputs

which = aimeta.which
try_install_exiftool = aimeta.try_install_exiftool
