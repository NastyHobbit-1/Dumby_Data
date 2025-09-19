AI Metadata Toolkit
===================

Structure
- Dumby/Run_AI_Toolkit.bat: single Windows launcher (double-clickable)
- Dumby/Bulked_AI: all Python sources
- Dumby/Boxxed_Data: all outputs (created as needed)
  - 01_Batches: filelist_*.txt, metadata_batch_*.json (ExifTool)
  - 02_NDJSON: structured_metadata*.ndjson
  - 03_PerFile_JSON: per-image JSON entries (optional)
  - 04_CSV: exported tables
  - 05_RenameMap: rename maps (02_NDJSON/04_CSV)

Quick Start (Interactive)
1) Double-click: Dumby\Run_AI_Toolkit.bat
2) Choose an action:
   - Process images (extract + normalize)
   - Export (04_CSV / Rename Map / Fixed 02_NDJSON)
   - Edit (Replace / Relocate / Delete)

Notes
- ExifTool is auto-detected/installed when needed (no admin required).
- Defaults write under Dumby\Boxxed_Data with per-type subfolders.
- Persistent 00_Rules are stored at Dumby\Boxxed_Data\00_Rules\export_rules.json.


