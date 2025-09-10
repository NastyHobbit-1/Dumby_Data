from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import csv
import json
import re

from .rules import RulesType, apply_rules_to_entry

try:
    import orjson  # type: ignore
    HAVE_ORJSON = True
except Exception:
    HAVE_ORJSON = False

INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*]+')


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
            s = line.strip()
            if not s:
                continue
            try:
                obj = orjson.loads(s) if HAVE_ORJSON else json.loads(s)
            except Exception:
                continue
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
    keys: Set[str] = set(["SourceFile"])
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


def export_ndjson_to_csv(ndjson_path: Path, out_csv: Path, rules: Optional[RulesType] = None, lean: bool = False) -> None:
    if lean:
        # Lean headers preset
        headers = [
            "SourceFile",
            "AI:Model",
            "AI:Scheduler",
            "AI:Sampler",
        ]
    else:
        headers, _ = collect_headers(ndjson_iter(ndjson_path, rules))
    rows = write_rows(ndjson_iter(ndjson_path, rules), headers, out_csv)
    print(f"CSV saved: {out_csv} | rows: {rows}")


def export_folder_to_csv(folder: Path, out_csv: Path, rules: Optional[RulesType] = None, lean: bool = False) -> None:
    if lean:
        headers = [
            "SourceFile",
            "AI:Model",
            "AI:Scheduler",
            "AI:Sampler",
        ]
    else:
        headers, _ = collect_headers(folder_iter(folder, rules))
    rows = write_rows(folder_iter(folder, rules), headers, out_csv)
    print(f"CSV saved: {out_csv} | rows: {rows}")


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


def extract_model(entry: Dict[str, Any]) -> Optional[str]:
    ai = entry.get("AI_Metadata")
    if isinstance(ai, dict):
        model = ai.get("Model") or ai.get("Model Version") or ai.get("Model Hash")
        if model:
            return str(model)
    m = entry.get("Model")
    return str(m) if m else None


def sanitize_filename_component(s: str) -> str:
    s = INVALID_FS_CHARS.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def export_ndjson_rename_map(
    ndjson_path: Path,
    out_path: Path,
    pad: int = 0,
    separator: str = " ",
    keep_extension: bool = True,
    sanitize: bool = True,
    rules: Optional[RulesType] = None,
    fmt: str = "ndjson",
) -> None:
    counts: Dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt.lower() == "csv":
        with open(out_path, "w", newline="", encoding="utf-8") as f:
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
                if rules and isinstance(rules.get("value_replacements"), dict):
                    model = rules["value_replacements"].get(model, model)
                counts[model] = counts.get(model, 0) + 1
                idx = counts[model]
                tag = f"{model}{separator}{idx:0{pad}d}" if pad and pad > 0 else f"{model}{separator}{idx}"
                w.writerow([img_name, tag])
        return
    # NDJSON map (default)
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in ndjson_iter(ndjson_path, rules=rules):
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
            tag = f"{model}{separator}{idx:0{pad}d}" if pad and pad > 0 else f"{model}{separator}{idx}"
            f.write(dumps_obj({"ImageName": img_name, "ModelTag": tag}))
            f.write("\n")


def export_folder_rename_map(
    folder: Path,
    out_path: Path,
    pad: int = 0,
    separator: str = " ",
    keep_extension: bool = True,
    sanitize: bool = True,
    rules: Optional[RulesType] = None,
    fmt: str = "ndjson",
) -> None:
    counts: Dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt.lower() == "csv":
        with open(out_path, "w", newline="", encoding="utf-8") as f:
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
                tag = f"{model}{separator}{idx:0{pad}d}" if pad and pad > 0 else f"{model}{separator}{idx}"
                w.writerow([img_name, tag])
        return
    with open(out_path, "w", encoding="utf-8") as f:
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
            tag = f"{model}{separator}{idx:0{pad}d}" if pad and pad > 0 else f"{model}{separator}{idx}"
            f.write(dumps_obj({"ImageName": img_name, "ModelTag": tag}))
            f.write("\n")
