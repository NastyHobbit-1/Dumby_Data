"""
AI Metadata Toolkit

This program processes images to generate rich AI metadata, then creates
structured files like CSV or NDJSON. Afterward, you can perform actions
such as renaming files, replacing/relocating/deleting data, or exporting
lean CSVs for quick analysis.

Structure
- scripts/pipeline: scan images, run ExifTool, normalize to NDJSON
- scripts/meta_export: export CSV, rename map (CSV/NDJSON), fixed NDJSON
- scripts/edits: replace/relocate/delete interactive utilities; apply rename map
- scripts/rules: persistent rules used by exports and edits
- scripts/ui: standardized interactive prompts and menus
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

from scripts import ui
from scripts.rules import RULES_DEFAULT_PATH, load_rules
from scripts.meta_export import (
    export_ndjson_to_csv,
    export_folder_to_csv,
    write_fixed_ndjson_from_ndjson,
    write_fixed_ndjson_from_folder,
    export_ndjson_rename_map,
    export_folder_rename_map,
)
from scripts.edits import (
    find_entry_by_image,
    replace_value,
    relocate_value,
    delete_value,
    apply_rename_map_and_update_data,
)


def run_pipeline_interactive() -> None:
    # Wrapper that defers to existing pipeline (backed by aimeta implementation)
    from scripts import pipeline as pipeline

    print("Process images — extract and normalize metadata to NDJSON. (Recommended for initial setup)")
    in_dir = Path(ui.strip_quotes(ui.prompt_input("Image source directory", getattr(pipeline, "DEFAULT_INPUT_DIR", "D:/AI"))))
    # Root for all outputs
    work_dir = Path(ui.strip_quotes(ui.prompt_input("Output root directory (Boxxed_Data)", str(getattr(pipeline, "DEFAULT_BOX_DIR", Path.cwd() / "Boxxed_Data")))))
    work_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = work_dir / "01_Batches"
    ndjson_dir = work_dir / "02_NDJSON"
    perfile_dir = work_dir / "03_PerFile_JSON"
    csv_dir = work_dir / "04_CSV"
    rename_dir = work_dir / "05_RenameMap"
    batches_dir.mkdir(parents=True, exist_ok=True)
    ndjson_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    rename_dir.mkdir(parents=True, exist_ok=True)

    # Batch size selection
    bs_opts = [100, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    sel = ui.menu_select("Select batch size (files per ExifTool batch)", [str(x) for x in bs_opts])
    batch_size = bs_opts[sel - 1]

    pretty = ui.prompt_yes_no("Pretty-print NDJSON lines?", False)
    wrap = 0
    if ui.prompt_yes_no("Wrap long text fields?", False):
        wrap = int(ui.prompt_input("Wrap width", "100"))

    out_name = ui.prompt_input("NDJSON output filename", Path(getattr(pipeline, "DEFAULT_OUT_PATH", ndjson_dir / "structured_metadata.ndjson")).name)
    out_path = ndjson_dir / out_name if not Path(out_name).is_absolute() else Path(out_name)

    # Decide extraction
    have_batches = any(batches_dir.glob("metadata_batch_*.json"))
    skip_extract = False
    if have_batches:
        skip_extract = ui.prompt_yes_no("Existing batch JSON detected. Use them and skip extraction?", True)

    # Find exiftool
    exe = None
    if not skip_extract:
        candidate = ui.strip_quotes(ui.prompt_input("Provide exiftool path or leave blank to auto-detect"))
        if candidate and Path(candidate).exists():
            exe = candidate
        if not exe:
            exe = pipeline.which("exiftool.exe") or pipeline.which("exiftool")
        if not exe and ui.prompt_yes_no("ExifTool not found. Try to install automatically?", True):
            exe = pipeline.try_install_exiftool()
        if not exe:
            print("ERROR: ExifTool is required for extraction and was not found.")
            return

    # Extraction
    if not skip_extract:
        exts = pipeline.parse_exts(getattr(pipeline, "DEFAULT_EXTS", "jpg,jpeg,png,tif,tiff,webp,heic"))
        files = pipeline.scan_images(in_dir, exts)
        print(f"Found {len(files):,} image files under {in_dir}")
        if not files:
            print("Nothing to do.")
            return
        pairs = pipeline.write_filelists(batches_dir, files, max(1, batch_size))
        pipeline.run_exiftool_batches(exe, pairs, getattr(pipeline, "DEFAULT_EXECUTOR", "thread"), max(1, getattr(pipeline, "DEFAULT_EXIF_WORKERS", 4)))
    else:
        print("Skipping extraction; will process existing metadata_batch_*.json")

    # Normalization
    batch_files = sorted(batches_dir.glob("metadata_batch_*.json"))
    if not batch_files:
        print(f"WARNING: No batch JSON found in {work_dir}. Pattern: metadata_batch_*.json")
        return

    per_file = ui.prompt_yes_no("Also write per-entry JSON files?", False)
    per_file_dir = perfile_dir
    if per_file:
        per_file_dir.mkdir(parents=True, exist_ok=True)

    # Use pipeline normalization path (already parallelized and robust)
    # Build a minimal Args-like object for reuse
    class Args:
        pass

    args = Args()
    args.pretty = pretty
    args.per_file = per_file
    args.parse_workers = getattr(pipeline, "DEFAULT_PARSE_WORKERS", 4)
    args.wrap = wrap

    out_path.unlink(missing_ok=True)
    total = 0
    tick = 0
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"Normalizing {len(batch_files)} batch files with {max(1, args.parse_workers)} workers...")
    with ProcessPoolExecutor(max_workers=max(1, args.parse_workers)) as ex:
        futures = []
        WAVE = 4000

        def flush_wave(futs):
            nonlocal total, tick
            for fut in as_completed(futs):
                obj = fut.result()
                with open(out_path, "ab") as f:
                    f.write(pipeline.fast_dump_bytes(obj, pretty=args.pretty))
                    f.write(b"\n")
                if per_file:
                    base = Path(obj.get("SourceFile") or "unknown").stem or "unknown"
                    out_p = per_file_dir / f"{base}_metadata.json"
                    with open(out_p, "wb") as pf:
                        pf.write(pipeline.fast_dump_bytes(obj, pretty=args.pretty))
                total += 1
                tick += 1
                if tick >= getattr(pipeline, "PROGRESS_EVERY", 2000):
                    print(f"normalized {total:,} entries...")
                    tick = 0

        for bf in batch_files:
            for entry in pipeline.stream_entries(bf):
                futures.append(ex.submit(pipeline.process_entry, entry, args.wrap))
                if len(futures) >= WAVE:
                    flush_wave(futures)
                    futures = []
        if futures:
            flush_wave(futures)

    print(f"Done. NDJSON: {out_path}  |  Total entries: {total:,}")


def run_exports_interactive() -> None:
    print("Export structured outputs: CSV, Rename Map, or fixed NDJSON.")
    source_mode = ui.menu_select("Select input type", [
        "NDJSON (faster for large datasets) - Recommended",
        "Per-file JSON folder",
    ])
    nd_path: Optional[Path] = None
    folder: Optional[Path] = None
    if source_mode == 1:
        from scripts import pipeline as pipeline
        default_nd = Path(getattr(pipeline, "DEFAULT_OUT_PATH", "structured_metadata.ndjson"))
        while True:
            nd_path = Path(ui.strip_quotes(ui.prompt_input("NDJSON path", str(default_nd))))
            if nd_path.exists():
                break
            print(f"NDJSON not found: {nd_path}")
            if not ui.prompt_yes_no("Try entering NDJSON path again?", True):
                return
    else:
        from scripts import pipeline as pipeline
        default_pf = Path(getattr(pipeline, "DEFAULT_PERFILE_DIR", "structured_metadata"))
        while True:
            folder = Path(ui.strip_quotes(ui.prompt_input("Per-file JSON folder", str(default_pf))))
            if folder.exists():
                break
            print(f"Folder not found: {folder}")
            if not ui.prompt_yes_no("Try entering folder again?", True):
                return

    # What to export
    exp = ui.menu_select("Choose export", [
        "Full CSV — include all fields",
        "Lean CSV — only file name, Model, Scheduler, Sampler",
        "Rename Map — NDJSON (default) or CSV",
        "Fixed NDJSON — apply rules and write corrected file",
    ])

    # Load rules for exports
    rules = None
    if ui.prompt_yes_no("Apply saved rules during export?", True):
        rules = load_rules(Path(RULES_DEFAULT_PATH))

    if exp in (1, 2):
        lean = exp == 2
        default_name = "ai_metadata_lean.csv" if lean else "ai_metadata_master.csv"
        from scripts import pipeline as pipeline
        default_csv_path = Path(getattr(pipeline, "DEFAULT_CSV_DIR", Path.cwd())) / default_name
        out_csv = ui.normalize_output_path(ui.prompt_input("Output CSV path", str(default_csv_path)), default_name)
        if nd_path:
            export_ndjson_to_csv(nd_path, out_csv, rules=rules, lean=lean)
        else:
            export_folder_to_csv(folder, out_csv, rules=rules, lean=lean)
        return

    if exp == 3:
        fmt_sel = ui.menu_select("Rename map format", ["NDJSON (default)", "CSV"])
        fmt = "csv" if fmt_sel == 2 else "ndjson"
        default_name = "image_model_map.ndjson" if fmt == "ndjson" else "image_model_map.csv"
        from scripts import pipeline as pipeline
        default_map_dir = Path(getattr(pipeline, "DEFAULT_RENAMEMAP_DIR", Path.cwd()))
        out_map = ui.normalize_output_path(ui.prompt_input("Rename map output path", str(default_map_dir / default_name)), default_name)
        pad = int(ui.prompt_input("Pad width (0 for none)", "3"))
        keep_ext = ui.prompt_yes_no("Keep image extension in first column?", True)
        sep = ui.prompt_input("Separator between model and number", " ")
        sanitize = ui.prompt_yes_no("Sanitize model text for filesystem use?", True)
        if nd_path:
            export_ndjson_rename_map(nd_path, out_map, pad=pad, separator=sep, keep_extension=keep_ext, sanitize=sanitize, rules=rules, fmt=fmt)
        else:
            export_folder_rename_map(folder, out_map, pad=pad, separator=sep, keep_extension=keep_ext, sanitize=sanitize, rules=rules, fmt=fmt)
        if ui.prompt_yes_no("Open and verify the rename map now?", False):
            print(f"Please open: {out_map}")
        if ui.prompt_yes_no("Apply this rename map to image files and metadata now?", False):
            images_root = Path(ui.strip_quotes(ui.prompt_input("Images root directory (will be searched recursively)", str(Path.cwd()))))
            data_nd = nd_path if nd_path else None
            data_folder = folder if (folder and folder.exists()) else None
            renamed, updated = apply_rename_map_and_update_data(out_map, images_root, data_ndjson=data_nd, data_folder=data_folder)
            print(f"Renamed files: {renamed} | Updated metadata entries: {updated}")
        else:
            print("Saved rename map for later use.")
        return

    if exp == 4:
        default_name = "structured_metadata_fixed.ndjson"
        from scripts import pipeline as pipeline
        default_ndjson_dir = Path(getattr(pipeline, "DEFAULT_NDJSON_DIR", Path.cwd()))
        out_nd = ui.normalize_output_path(ui.prompt_input("Fixed NDJSON output path", str(default_ndjson_dir / default_name)), default_name)
        if nd_path:
            lines = write_fixed_ndjson_from_ndjson(nd_path, out_nd, rules=rules or {})
        else:
            lines = write_fixed_ndjson_from_folder(folder, out_nd, rules=rules or {})
        print(f"Fixed NDJSON saved: {out_nd} | lines: {lines}")
        return


def run_edits_interactive() -> None:
    print("Edit metadata: replace, relocate, or delete values. Changes can be saved as rules for future runs.")
    source_mode = ui.menu_select("Select input type", [
        "NDJSON",
        "Per-file JSON folder",
    ])
    nd_path: Optional[Path] = None
    folder: Optional[Path] = None
    if source_mode == 1:
        while True:
            from scripts import pipeline as pipeline
            default_nd = Path(getattr(pipeline, "DEFAULT_OUT_PATH", "structured_metadata.ndjson"))
            nd_path = Path(ui.strip_quotes(ui.prompt_input("NDJSON path", str(default_nd))))
            if nd_path.exists():
                break
            print(f"NDJSON not found: {nd_path}")
            if not ui.prompt_yes_no("Try entering NDJSON path again?", True):
                return
    else:
        while True:
            from scripts import pipeline as pipeline
            default_pf = Path(getattr(pipeline, "DEFAULT_PERFILE_DIR", "structured_metadata"))
            folder = Path(ui.strip_quotes(ui.prompt_input("Per-file JSON folder", str(default_pf))))
            if folder.exists():
                break
            print(f"Folder not found: {folder}")
            if not ui.prompt_yes_no("Try entering folder again?", True):
                return

    action = ui.menu_select("Choose action", [
        "Replace — change a value (optionally across all files)",
        "Relocate — move a value from one field to another",
        "Delete — remove a value",
    ])

    # Optionally target a specific image for context
    if ui.prompt_yes_no("Provide an image path to inspect its metadata first?", False):
        img_path = Path(ui.strip_quotes(ui.prompt_input("Image path")))
        entry = find_entry_by_image(nd_path, folder, img_path)
        if not entry:
            print("No entry found for that image.")
        else:
            print("Entry preview (truncated):")
            from scripts.meta_export import flatten_entry
            flat = flatten_entry(entry)
            shown = 0
            for k in sorted(flat.keys()):
                print(f"  {k}: {flat[k]}")
                shown += 1
                if shown >= 25:
                    print("  ...")
                    break

    if action == 1:
        slot = ui.prompt_input("Field (e.g., AI:Model or General:Author)")
        old_val = ui.prompt_input("Exact value to replace")
        new_val = ui.prompt_input("New value")
        apply_all = ui.prompt_yes_no("Apply to all matching occurrences?", True)
        only_image_name = None
        if not apply_all:
            only_image_name = Path(ui.strip_quotes(ui.prompt_input("Limit to which image file? (enter path)")).strip()).name
        persist = ui.prompt_yes_no("Save as a replacement rule for future runs?", apply_all)
        affected, new_nd = replace_value(nd_path, folder, slot, old_val, new_val, apply_all, persist, only_image_name=only_image_name)
        print(f"Replaced {affected} occurrence(s)." + (f" Wrote: {new_nd}" if new_nd else ""))
        return

    if action == 2:
        src = ui.prompt_input("From field (e.g., AI:Model Version)")
        dst = ui.prompt_input("To field (e.g., AI:Model)")
        mv = ui.prompt_input("Match exact value")
        mode = "append" if ui.prompt_yes_no("Append to destination instead of replace?", False) else "replace"
        apply_all = ui.prompt_yes_no("Apply across all files?", True)
        only_image_name = None
        if not apply_all:
            only_image_name = Path(ui.strip_quotes(ui.prompt_input("Limit to which image file? (enter path)")).strip()).name
        persist = ui.prompt_yes_no("Save as a relocation rule?", apply_all)
        affected, new_nd = relocate_value(nd_path, folder, src, dst, mv, mode=mode, apply_all=apply_all, persist_rule=persist, only_image_name=only_image_name)
        print(f"Relocated {affected} occurrence(s)." + (f" Wrote: {new_nd}" if new_nd else ""))
        return

    if action == 3:
        slot = ui.prompt_input("Field (e.g., AI:Sampler or General:Notes)")
        mv_in = ui.prompt_input("Match exact value (leave blank to delete unconditionally)", "")
        mv = mv_in if mv_in != "" else None
        apply_all = ui.prompt_yes_no("Apply across all files?", True)
        only_image_name = None
        if not apply_all:
            only_image_name = Path(ui.strip_quotes(ui.prompt_input("Limit to which image file? (enter path)")).strip()).name
        persist = ui.prompt_yes_no("Save as a deletion rule (replacement to empty)?", apply_all)
        affected, new_nd = delete_value(nd_path, folder, slot, mv, apply_all=apply_all, persist_rule=persist, only_image_name=only_image_name)
        print(f"Deleted {affected} value(s)." + (f" Wrote: {new_nd}" if new_nd else ""))
        return


def interactive_main() -> None:
    print("AI Metadata Toolkit")
    print("This program processes images to generate metadata, then creates structured outputs (CSV/NDJSON). You can also rename files or edit metadata via replace/relocate/delete operations.")

    choice = ui.menu_select("Select an action", [
        "Process images — extract + normalize (Recommended for initial setup)",
        "Export — CSV / Rename Map / Fixed NDJSON",
        "Edit — Replace / Relocate / Delete",
        "Exit",
    ])
    if choice == 1:
        run_pipeline_interactive()
        return
    if choice == 2:
        run_exports_interactive()
        return
    if choice == 3:
        run_edits_interactive()
        return


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified AI Metadata Toolkit")
    ap.add_argument("--interactive", action="store_true", help="Run interactive menu")
    args = ap.parse_args()
    if len(vars(args)) == 1 or args.interactive:
        return interactive_main()
    print("For now, run with --interactive for guided usage.")


if __name__ == "__main__":
    main()
