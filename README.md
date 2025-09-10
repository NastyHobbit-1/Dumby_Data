# Dumby_Data
Ai Image Meta Data Reader and Exporter

Structure & Usage
- Double-click `Dumby/Run_AI_Toolkit.bat` for the interactive menu.
- Sources: `Dumby/Bulked_AI` (main.py, aimeta.py, meta2csv.py, scripts/)
- Outputs: `Dumby/Boxxed_Data`
  - `00_Rules`: persistent rules (`export_rules.json`)
  - `01_Batches`: ExifTool list + batch JSON
  - `02_NDJSON`: normalized metadata
  - `03_PerFile_JSON`: per-image JSONs (optional)
  - `04_CSV`: exported tables
  - `05_RenameMap`: rename maps (NDJSON/CSV)

Git
- `.gitignore` excludes `Dumby/Boxxed_Data/` and common caches.
