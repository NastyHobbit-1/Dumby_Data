"""
meta2csv.py

Utilities to export AI metadata to CSV or rename maps, and to
optionally write a corrected NDJSON applying persistent rules.

Main capabilities:
- Read pipeline NDJSON or per-file JSONs
- Export a full flattened CSV (AI:* and General:* columns)
- Export a rename map: ImageName, ModelTag (model + incremental number)
- Apply and remember exact-match replacements and relocations
- Optionally emit a fixed NDJSON with applied rules
"""

import argparse
import sys
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import orjson  # type: ignore
    HAVE_ORJSON = True
except Exception:
    HAVE_ORJSON = False


# ----------------------- Persistent Rules -----------------------
# Default rules path inside Dumby/Boxxed_Data/Rules
RULES_DEFAULT_PATH = str(Path(__file__).resolve().parents[1] / "Boxxed_Data" / "Rules" / "export_rules.json")

RulesType = Dict[str, Any]


def _empty_rules() -> RulesType:
    return {"value_replacements": {}, "relocations": []}


def load_rules(path: Path) -> RulesType:
    if not path.exists():
        return _empty_rules()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty_rules()
        data.setdefault("value_replacements", {})
        data.setdefault("relocations", [])
        if not isinstance(data["value_replacements"], dict):
            data["value_replacements"] = {}
        if not isinstance(data["relocations"], list):
            data["relocations"] = []
        return data
    except Exception:
        return _empty_rules()


def save_rules(path: Path, rules: RulesType) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)


def _apply_value_replacements_inplace(obj: Any, repl: Dict[str, str]) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str):
        return repl.get(obj, obj)
    if isinstance(obj, list):
        return [_apply_value_replacements_inplace(x, repl) for x in obj]
    if isinstance(obj, dict):
        return {k: _apply_value_replacements_inplace(v, repl) for k, v in obj.items()}
    return obj


def _resolve_slot(entry: Dict[str, Any], slot: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    # Supports: AI:Field, General:Field, SourceFile
    s = slot.strip()
    if s.lower() in ("sourcefile", "root:sourcefile"):
        return entry, "SourceFile"
    if ":" in s:
        prefix, key = s.split(":", 1)
        prefix = prefix.strip().lower()
        key = key.strip()
        if prefix in ("ai", "ai_metadata"):
            d = entry.get("AI_Metadata")
            if isinstance(d, dict):
                return d, key
            entry["AI_Metadata"] = {}
            return entry["AI_Metadata"], key
        if prefix in ("general", "generalmetadata"):
            d = entry.get("GeneralMetadata")
            if isinstance(d, dict):
                return d, key
            entry["GeneralMetadata"] = {}
            return entry["GeneralMetadata"], key
    return None, None


def _apply_relocations_inplace(entry: Dict[str, Any], relocations: List[Dict[str, Any]]) -> None:
    for rule in relocations:
        try:
            src = str(rule.get("from", "")).strip()
            dst = str(rule.get("to", "")).strip()
            match = rule.get("match", None)
            if not src or not dst:
                continue
            src_dict, src_key = _resolve_slot(entry, src)
            dst_dict, dst_key = _resolve_slot(entry, dst)
            if not src_dict or not dst_dict or not src_key or not dst_key:
                continue
            if src_key not in src_dict:
                continue
            val = src_dict.get(src_key)
            # Only identical matches; string equivalence
            if match is not None and str(val) != str(match):
                continue
            # Move: set destination if empty or equal; then delete source
            cur = dst_dict.get(dst_key)
            if cur in (None, "") or cur == val:
                dst_dict[dst_key] = val
            # Delete source
            try:
                del src_dict[src_key]
            except Exception:
                pass
        except Exception:
            continue


def apply_rules_to_entry(entry: Dict[str, Any], rules: Optional[RulesType]) -> Dict[str, Any]:
    if not rules:
        return entry
    # Value replacements across AI and General
    repl = rules.get("value_replacements") or {}
    if isinstance(repl, dict) and (repl):
        if isinstance(entry.get("AI_Metadata"), dict):
            entry["AI_Metadata"] = _apply_value_replacements_inplace(entry["AI_Metadata"], repl)
        if isinstance(entry.get("GeneralMetadata"), dict):
            entry["GeneralMetadata"] = _apply_value_replacements_inplace(entry["GeneralMetadata"], repl)
        # Also cover SourceFile if needed
        if isinstance(entry.get("SourceFile"), str):
            sf = entry["SourceFile"]
            entry["SourceFile"] = repl.get(sf, sf)
    # Relocations
    rel = rules.get("relocations") or []
    if isinstance(rel, list) and rel:
        _apply_relocations_inplace(entry, rel)
    return entry


def parse_line_json(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    if not s:
        return None
    try:
        if HAVE_ORJSON:
            return orjson.loads(s)
        return json.loads(s)
    except Exception:
        return None


def to_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def dumps_obj(obj: Any) -> str:
    try:
        if HAVE_ORJSON:
            return orjson.dumps(obj).decode("utf-8")
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        # Fallback best-effort
        return json.dumps(obj, ensure_ascii=False, default=str)


def flatten_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    flat: Dict[str, str] = {"SourceFile": to_cell(entry.get("SourceFile", "unknown"))}

    general = entry.get("GeneralMetadata", {})
    if isinstance(general, dict):
        for k, v in general.items():
            flat[f"General:{k}"] = to_cell(v)

    ai = entry.get("AI_Metadata", {})
    if isinstance(ai, dict):
        for k, v in ai.items():
            flat[f"AI:{k}"] = to_cell(v)

    return flat


def ndjson_iter(path: Path, rules: Optional[RulesType] = None) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = parse_line_json(line)
            if isinstance(obj, dict):
                yield apply_rules_to_entry(obj, rules)


def folder_iter(folder: Path, rules: Optional[RulesType] = None) -> Iterable[Dict[str, Any]]:
    for file in sorted(folder.glob("*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                yield apply_rules_to_entry(obj, rules)
        except Exception:
            continue


def collect_headers(it: Iterable[Dict[str, Any]]) -> Tuple[List[str], int]:
    keys: Set[str] = set(["SourceFile"])  # ensure SourceFile included
    count = 0
    for obj in it:
        row = flatten_entry(obj)
        keys.update(row.keys())
        count += 1
        if count % 20000 == 0:
            print(f"scanned {count:,} rows for headers...")
    headers = sorted(keys)
    if "SourceFile" in headers:
        headers.remove("SourceFile")
        headers = ["SourceFile"] + headers
    return headers, count


def write_rows(it: Iterable[Dict[str, Any]], headers: List[str], out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for obj in it:
            row = flatten_entry(obj)
            w.writerow(row)
            count += 1
            if count % 20000 == 0:
                print(f"wrote {count:,} rows...")
    return count


def export_ndjson_to_csv(ndjson_path: Path, out_csv: Path, rules: Optional[RulesType] = None) -> None:
    headers, _ = collect_headers(ndjson_iter(ndjson_path, rules))
    rows = write_rows(ndjson_iter(ndjson_path, rules), headers, out_csv)
    print(f"CSV saved: {out_csv} | rows: {rows}")


def export_folder_to_csv(folder: Path, out_csv: Path, rules: Optional[RulesType] = None) -> None:
    headers, _ = collect_headers(folder_iter(folder, rules))
    rows = write_rows(folder_iter(folder, rules), headers, out_csv)
    print(f"CSV saved: {out_csv} | rows: {rows}")


# ----------------------- Fixed outputs (NDJSON/JSON) -----------------------
def write_fixed_ndjson_from_ndjson(in_path: Path, out_path: Path, rules: RulesType) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for obj in ndjson_iter(in_path, rules):
            out_f.write(dumps_obj(obj))
            out_f.write("\n")
            count += 1
            if count % 20000 == 0:
                print(f"wrote fixed ndjson {count:,} lines...")
    return count


def write_fixed_ndjson_from_folder(in_folder: Path, out_path: Path, rules: RulesType) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for obj in folder_iter(in_folder, rules):
            out_f.write(dumps_obj(obj))
            out_f.write("\n")
            count += 1
            if count % 20000 == 0:
                print(f"wrote fixed ndjson {count:,} lines...")
    return count


# ----------------------- Rename-map generation -----------------------
def extract_model(entry: Dict[str, Any]) -> Optional[str]:
    ai = entry.get("AI_Metadata")
    if isinstance(ai, dict):
        model = ai.get("Model") or ai.get("Model Version") or ai.get("Model Hash")
        if model:
            return str(model)
    # Fallbacks
    m = entry.get("Model")
    return str(m) if m else None


INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*]+')


def sanitize_filename_component(s: str) -> str:
    # Remove invalid characters and collapse whitespace
    s = INVALID_FS_CHARS.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def export_ndjson_rename_map(
    ndjson_path: Path,
    out_csv: Path,
    pad: int = 0,
    separator: str = " ",
    keep_extension: bool = True,
    sanitize: bool = True,
    rules: Optional[RulesType] = None,
) -> None:
    counts: Dict[str, int] = {}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ImageName", "ModelTag"])  # image name first, then model+number
        for entry in ndjson_iter(ndjson_path, rules=rules):
            src = entry.get("SourceFile") or entry.get("Source") or "unknown"
            img_name = Path(str(src)).name if keep_extension else Path(str(src)).stem
            model_raw = extract_model(entry)
            model = (str(model_raw).strip() if model_raw is not None else "")
            if sanitize:
                model = sanitize_filename_component(model)
            if not model:
                model = "Unknown"
            # Apply output-level replacement if exact match rule exists
            if rules and isinstance(rules.get("value_replacements"), dict):
                model = rules["value_replacements"].get(model, model)
            counts[model] = counts.get(model, 0) + 1
            idx = counts[model]
            if pad and pad > 0:
                tag = f"{model}{separator}{idx:0{pad}d}"
            else:
                tag = f"{model}{separator}{idx}"
            w.writerow([img_name, tag])


def export_folder_rename_map(
    folder: Path,
    out_csv: Path,
    pad: int = 0,
    separator: str = " ",
    keep_extension: bool = True,
    sanitize: bool = True,
    rules: Optional[RulesType] = None,
) -> None:
    counts: Dict[str, int] = {}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ImageName", "ModelTag"])  # image name first, then model+number
        for entry in folder_iter(folder, rules=rules):
            src = entry.get("SourceFile") or entry.get("Source") or "unknown"
            img_name = Path(str(src)).name if keep_extension else Path(str(src)).stem
            model_raw = extract_model(entry)
            model = (str(model_raw).strip() if model_raw is not None else "")
            if sanitize:
                model = sanitize_filename_component(model)
            if not model:
                model = "Unknown"
            if rules and isinstance(rules.get("value_replacements"), dict):
                model = rules["value_replacements"].get(model, model)
            counts[model] = counts.get(model, 0) + 1
            idx = counts[model]
            if pad and pad > 0:
                tag = f"{model}{separator}{idx:0{pad}d}"
            else:
                tag = f"{model}{separator}{idx}"
            w.writerow([img_name, tag])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export structured AI metadata: either a full CSV or a rename-map (ImageName, Model+index)."
    )
    ap.add_argument("--ndjson", default=None, help="Path to structured_metadata.ndjson produced by the pipeline")
    ap.add_argument(
        "--input-dir",
        default=None,
        help="Folder of per-file JSONs (structured_metadata) if using --per-file",
    )
    ap.add_argument("--out", default="ai_metadata_master.csv", help="Output CSV path")
    # Rules: persistent transforms and relocations
    ap.add_argument("--rules", default=RULES_DEFAULT_PATH, help="Path to persistent rules JSON (created if missing)")
    ap.add_argument("--no-rules", action="store_true", help="Do not apply rules during this run")
    ap.add_argument(
        "--remember-replace", action="append", default=[],
        help='Remember a global exact value replacement in the form "old=new"; can be used multiple times.'
    )
    ap.add_argument(
        "--remember-move", action="append", default=[],
        help='Remember a relocation rule: "from=AI:Field;to=AI:Field;match=ExactValue"; can be used multiple times.'
    )
    # Fixed outputs (apply rules to write corrected data)
    ap.add_argument(
        "--fixed-ndjson-out",
        default=None,
        help=(
            "Write a corrected NDJSON applying rules. "
            "Uses --ndjson or --input-dir as source."
        ),
    )
    ap.add_argument(
        "--rename-map",
        action="store_true",
        help=(
            "Output two columns: ImageName, ModelTag (model name + "
            "incremental number)"
        ),
    )
    ap.add_argument(
        "--pad",
        type=int,
        default=0,
        help="Zero-pad the incremental number to N digits (for rename-map only)",
    )
    ap.add_argument(
        "--no-ext",
        action="store_true",
        help=(
            "Use image base name without extension in first column "
            "(rename-map only)"
        ),
    )
    ap.add_argument(
        "--separator",
        default=" ",
        help="Separator between model and number (rename-map only)",
    )
    ap.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Do not sanitize model text for filesystem use (rename-map only)",
    )
    ap.add_argument("--interactive", action="store_true", help="Run with guided prompts instead of flags")
    args = ap.parse_args()

    if len(sys.argv) == 1 or args.interactive:
        return interactive_main()

    out_csv = Path(args.out)

    # Load existing rules, apply requested updates, and persist
    rules_path = Path(args.rules)
    rules = load_rules(rules_path)

    # Parse remember-replace entries
    new_repl = 0
    for spec in args.remember_replace:
        if not isinstance(spec, str) or "=" not in spec:
            continue
        old, new = spec.split("=", 1)
        old = old.strip()
        new = new.strip()
        if old:
            rules.setdefault("value_replacements", {})[old] = new
            new_repl += 1

    # Parse remember-move entries
    def parse_move_spec(s: str) -> Optional[Dict[str, str]]:
        try:
            parts = [p.strip() for p in s.split(";") if p.strip()]
            d: Dict[str, str] = {}
            for p in parts:
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                d[k.strip().lower()] = v.strip()
            if "from" in d and "to" in d:
                return {"from": d["from"], "to": d["to"], "match": d.get("match")}
        except Exception:
            pass
        return None

    new_moves = 0
    for spec in args.remember_move:
        if not isinstance(spec, str):
            continue
        rule = parse_move_spec(spec)
        if rule:
            rules.setdefault("relocations", []).append(rule)
            new_moves += 1

    if new_repl or new_moves:
        save_rules(rules_path, rules)
        print(f"Saved rules to {rules_path} (added {new_repl} replacements, {new_moves} moves)")

    active_rules = None if args.no_rules else rules

    # Optional: write corrected NDJSON
    if args.fixed_ndjson_out:
        out_fixed = Path(args.fixed_ndjson_out)
        if args.ndjson:
            nd = Path(args.ndjson)
            if not nd.exists():
                raise SystemExit(f"NDJSON not found: {nd}")
            lines = write_fixed_ndjson_from_ndjson(nd, out_fixed, rules=active_rules or {})
            print(f"Fixed NDJSON saved: {out_fixed} | lines: {lines}")
        else:
            folder = Path(args.input_dir or "structured_metadata")
            if not folder.exists():
                raise SystemExit(f"Input folder not found: {folder}")
            lines = write_fixed_ndjson_from_folder(folder, out_fixed, rules=active_rules or {})
            print(f"Fixed NDJSON saved: {out_fixed} | lines: {lines}")

    if args.rename_map:
        keep_ext = not args.no_ext
        sanitize = not args.no_sanitize
        if args.ndjson:
            nd = Path(args.ndjson)
            if not nd.exists():
                raise SystemExit(f"NDJSON not found: {nd}")
            export_ndjson_rename_map(
                nd,
                out_csv,
                pad=args.pad,
                separator=args.separator,
                keep_extension=keep_ext,
                sanitize=sanitize,
                rules=active_rules,
            )
            return
        folder = Path(args.input_dir or "structured_metadata")
        if not folder.exists():
            raise SystemExit(f"Input folder not found: {folder}")
        export_folder_rename_map(
            folder,
            out_csv,
            pad=args.pad,
            separator=args.separator,
            keep_extension=keep_ext,
            sanitize=sanitize,
            rules=active_rules,
        )
        return

    # Default full CSV export
    if args.ndjson:
        nd = Path(args.ndjson)
        if not nd.exists():
            raise SystemExit(f"NDJSON not found: {nd}")
        export_ndjson_to_csv(nd, out_csv, rules=active_rules)
        return
    folder = Path(args.input_dir or "structured_metadata")
    if not folder.exists():
        raise SystemExit(f"Input folder not found: {folder}")
    export_folder_to_csv(folder, out_csv, rules=active_rules)


# ----------------------- Interactive Mode -----------------------

def _prompt_input(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or (default or "")


def _prompt_yes_no(question: str, default_yes: bool = True) -> bool:
    a = "Y" if default_yes else "N"
    b = "N" if default_yes else "Y"
    while True:
        ans = input(f"{question} Please select {a} for option A or {b} for option B: ").strip().lower()
        if not ans:
            return default_yes
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def _prompt_select(title: str, options: List[str]) -> int:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print(title)
    for i, name in enumerate(options, 1):
        label = f"Option {letters[i-1]} - {i}"
        print(f"  {label}: {name}")
    while True:
        sel = input(f"Please select {' '.join(str(i) for i in range(1, len(options)+1))}: ").strip()
        if sel.isdigit():
            n = int(sel)
            if 1 <= n <= len(options):
                return n


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        return s[1:-1]
    return s


def _normalize_output_path(user_input: str, default_filename: str) -> Path:
    """Resolve user output path; if it's a directory, append default filename.
    Handles quoted input and trailing separators.
    """
    cleaned = _strip_quotes(user_input)
    p = Path(cleaned)
    # If path exists and is directory, use default file inside it
    if p.exists() and p.is_dir():
        return p / default_filename
    # Treat explicit trailing separators as directory intent
    if cleaned.endswith(("/", "\\")):
        return Path(cleaned) / default_filename
    return p


def _ensure_orjson_interactive() -> None:
    global HAVE_ORJSON
    if not HAVE_ORJSON:
        if _prompt_yes_no("orjson is not installed. Install now for faster parsing?", True):
            try:
                import subprocess
                _py = sys.executable
                subprocess.check_call([_py, "-m", "pip", "install", "--upgrade", "orjson"])  # noqa: S603,S607
                HAVE_ORJSON = True
                globals()["orjson"] = __import__("orjson")
                print("Installed orjson.")
            except Exception as e:  # pragma: no cover
                print(f"WARNING: Failed to install orjson: {e}")


def interactive_main() -> None:
    print("Welcome to Meta2CSV interactive mode.")
    top = _prompt_select("Select mode", [
        "Scan (Export: CSV / RenameMap / Fixed NDJSON)",
        "Repl/Relo (Manage replacements/relocations)",
        "Exit",
    ])
    if top == 2:
        rules_path = Path(RULES_DEFAULT_PATH)
        # Reuse existing rule helpers
        # Simple loop using existing load/save
        rules = load_rules(rules_path)
        while True:
            sel = _prompt_select("Rules: choose an action", [
                "Add replacement (exact value)",
                "Add relocation (from/to/match)",
                "List current rules",
                "Save & return",
            ])
            if sel == 1:
                old = _prompt_input("Enter exact value to replace")
                new = _prompt_input("Enter replacement value")
                if old:
                    rules.setdefault("value_replacements", {})[old] = new
                    print("Added replacement.")
            elif sel == 2:
                src = _prompt_input("From slot (e.g., AI:Model Version)")
                dst = _prompt_input("To slot (e.g., AI:Model)")
                match = _prompt_input("Match exact value (required)")
                if src and dst and match:
                    rules.setdefault("relocations", []).append({"from": src, "to": dst, "match": match})
                    print("Added relocation.")
            elif sel == 3:
                print(json.dumps(rules, ensure_ascii=False, indent=2))
            else:
                save_rules(rules_path, rules)
                print(f"Saved rules to {rules_path}")
                return
    if top == 3:
        return

    # Scan/Export
    _ensure_orjson_interactive()
    use_nd = _prompt_yes_no("Use NDJSON as input? (No = per-file JSON folder)", True)
    nd_path: Optional[Path] = None
    folder: Optional[Path] = None
    if use_nd:
        # Loop until a valid NDJSON path is provided or user aborts
        while True:
            nd_path = Path(_strip_quotes(_prompt_input("NDJSON path", "structured_metadata.ndjson")))
            if nd_path.exists():
                break
            print(f"NDJSON not found: {nd_path}")
            if not _prompt_yes_no("Try entering NDJSON path again?", True):
                print("Cancelled.")
                return
    else:
        # Loop until a valid folder is provided or user aborts
        while True:
            folder = Path(_strip_quotes(_prompt_input("Per-file JSON folder", "structured_metadata")))
            if folder.exists():
                break
            print(f"Input folder not found: {folder}")
            if not _prompt_yes_no("Try entering folder again?", True):
                print("Cancelled.")
                return
    apply_rules = _prompt_yes_no("Apply saved rules during export?", True)
    rules = None if not apply_rules else load_rules(Path(RULES_DEFAULT_PATH))

    # Export type
    export_sel = _prompt_select("Choose export", ["Full CSV", "Rename Map", "Fixed NDJSON"])
    if export_sel == 1:
        out_csv = _normalize_output_path(_prompt_input("Output CSV path", "ai_metadata_master.csv"), "ai_metadata_master.csv")
        if nd_path:
            export_ndjson_to_csv(nd_path, out_csv, rules=rules)
        else:
            if not folder or not folder.exists():
                print("Input folder not found.")
                return
            export_folder_to_csv(folder, out_csv, rules=rules)
        return
    if export_sel == 2:
        out_csv = _normalize_output_path(_prompt_input("Rename map CSV path", "image_model_map.csv"), "image_model_map.csv")
        pad = int(_prompt_input("Pad width (0 for none)", "3"))
        keep_ext = _prompt_yes_no("Keep image extension in first column?", True)
        sep = _prompt_input("Separator between model and number", " ")
        sanitize = _prompt_yes_no("Sanitize model text for filesystem use?", True)
        if nd_path:
            export_ndjson_rename_map(
                nd_path,
                out_csv,
                pad=pad,
                separator=sep,
                keep_extension=keep_ext,
                sanitize=sanitize,
                rules=rules,
            )
        else:
            if not folder or not folder.exists():
                print("Input folder not found.")
                return
            export_folder_rename_map(
                folder,
                out_csv,
                pad=pad,
                separator=sep,
                keep_extension=keep_ext,
                sanitize=sanitize,
                rules=rules,
            )
        return
    if export_sel == 3:
        out_nd = _normalize_output_path(_prompt_input("Fixed NDJSON output path", "structured_metadata_fixed.ndjson"), "structured_metadata_fixed.ndjson")
        if nd_path:
            # nd_path existence was validated earlier, but double-check for safety
            if not nd_path.exists():
                print(f"NDJSON not found: {nd_path}")
                return
            lines = write_fixed_ndjson_from_ndjson(nd_path, out_nd, rules=rules or {})
        else:
            if not folder or not folder.exists():
                print("Input folder not found.")
                return
            lines = write_fixed_ndjson_from_folder(folder, out_nd, rules=rules or {})
        print(f"Fixed NDJSON saved: {out_nd} | lines: {lines}")
        return


if __name__ == "__main__":
    main()
