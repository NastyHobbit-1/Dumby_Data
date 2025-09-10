AI Metadata Toolkit
===================

Structure
- Dumby/Run_AI_Toolkit.bat: single Windows launcher (double-clickable)
- Dumby/Bulked_AI: all Python sources
- Dumby/Boxxed_Data: all outputs (created as needed)
  - Batches: filelist_*.txt, metadata_batch_*.json (ExifTool)
  - NDJSON: structured_metadata*.ndjson
  - PerFile_JSON: per-image JSON entries (optional)
  - CSV: exported tables
  - RenameMap: rename maps (NDJSON/CSV)

Quick Start (Interactive)
1) Double-click: Dumby\Run_AI_Toolkit.bat
2) Choose an action:
   - Process images (extract + normalize)
   - Export (CSV / Rename Map / Fixed NDJSON)
   - Edit (Replace / Relocate / Delete)

Notes
- ExifTool is auto-detected/installed when needed (no admin required).
- Defaults write under Dumby\Boxxed_Data with per-type subfolders.
- Persistent rules are stored at Dumby\Boxxed_Data\Rules\export_rules.json.

