from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json

from .rules import load_rules, save_rules, RULES_DEFAULT_PATH
from .meta_export import ndjson_iter, folder_iter, dumps_obj


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                pass
    return out


def _write_ndjson(path: Path, items: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(dumps_obj(obj))
            f.write("\n")
            count += 1
    return count


def _get_slot(entry: Dict[str, Any], slot: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    s = slot.strip()
    if s.lower() == "sourcefile":
        return entry, "SourceFile"
    if ":" in s:
        prefix, key = s.split(":", 1)
        prefix = prefix.strip().lower()
        key = key.strip()
        if prefix in ("ai", "ai_metadata"):
            d = entry.setdefault("AI_Metadata", {})
            return d, key
        if prefix in ("general", "generalmetadata"):
            d = entry.setdefault("GeneralMetadata", {})
            return d, key
    return None, None


def find_entry_by_image(ndjson_path: Optional[Path], folder: Optional[Path], image_path: Path) -> Optional[Dict[str, Any]]:
    target_name = image_path.name.lower()
    if ndjson_path and ndjson_path.exists():
        for obj in ndjson_iter(ndjson_path, rules=None):
            sf = str(obj.get("SourceFile") or obj.get("Source") or "").lower()
            if sf.endswith(target_name):
                return obj
    if folder and folder.exists():
        for obj in folder_iter(folder, rules=None):
            sf = str(obj.get("SourceFile") or obj.get("Source") or "").lower()
            if sf.endswith(target_name):
                return obj
    return None


def replace_value(ndjson_path: Optional[Path], folder: Optional[Path], slot: str, old_value: str, new_value: str,
                  apply_all: bool, persist_rule: bool, rules_path: Path = Path(RULES_DEFAULT_PATH),
                  only_image_name: Optional[str] = None) -> Tuple[int, Optional[Path]]:
    """Replace exact value in selected slot across data. Returns (affected_count, new_ndjson_path)."""
    affected = 0
    new_nd = None
    if ndjson_path and ndjson_path.exists():
        items = _read_ndjson(ndjson_path)
        for obj in items:
            if only_image_name:
                sfv = str(obj.get("SourceFile") or "")
                if Path(sfv).name.lower() != only_image_name.lower():
                    continue
            target, key = _get_slot(obj, slot)
            if not target or not key:
                continue
            v = target.get(key)
            if v is None:
                continue
            if str(v) == old_value:
                target[key] = new_value
                affected += 1
        if affected:
            new_nd = ndjson_path.with_name(ndjson_path.stem + "_edited.ndjson")
            _write_ndjson(new_nd, items)
    elif folder and folder.exists():
        for f in folder.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                if only_image_name:
                    sfv = str(obj.get("SourceFile") or "")
                    if Path(sfv).name.lower() != only_image_name.lower():
                        continue
                target, key = _get_slot(obj, slot)
                if not target or not key:
                    continue
                v = target.get(key)
                if v is None:
                    continue
                if str(v) == old_value:
                    target[key] = new_value
                    affected += 1
                    with open(f, "w", encoding="utf-8") as fh:
                        json.dump(obj, fh, ensure_ascii=False, indent=2)
            except Exception:
                continue
    if persist_rule and apply_all and old_value:
        rules = load_rules(rules_path)
        rules.setdefault("value_replacements", {})[old_value] = new_value
        save_rules(rules_path, rules)
    return affected, new_nd


def relocate_value(ndjson_path: Optional[Path], folder: Optional[Path], src_slot: str, dst_slot: str, match_value: str,
                   mode: str = "replace", apply_all: bool = True, persist_rule: bool = True,
                   rules_path: Path = Path(RULES_DEFAULT_PATH), only_image_name: Optional[str] = None) -> Tuple[int, Optional[Path]]:
    """Relocate values matching `match_value` from src_slot to dst_slot. mode: replace|append"""
    affected = 0
    new_nd = None

    def do_move(entry: Dict[str, Any]) -> bool:
        if only_image_name:
            sfv = str(entry.get("SourceFile") or "")
            if Path(sfv).name.lower() != only_image_name.lower():
                return False
        sdict, skey = _get_slot(entry, src_slot)
        ddict, dkey = _get_slot(entry, dst_slot)
        if not sdict or not skey or not ddict or not dkey:
            return False
        if skey not in sdict:
            return False
        val = sdict.get(skey)
        if str(val) != match_value:
            return False
        if mode == "append" and dkey in ddict and ddict[dkey]:
            try:
                if isinstance(ddict[dkey], list):
                    ddict[dkey].append(val)
                else:
                    ddict[dkey] = f"{ddict[dkey]} {val}"
            except Exception:
                ddict[dkey] = val
        else:
            ddict[dkey] = val
        try:
            del sdict[skey]
        except Exception:
            pass
        return True

    if ndjson_path and ndjson_path.exists():
        items = _read_ndjson(ndjson_path)
        for obj in items:
            if do_move(obj):
                affected += 1
        if affected:
            new_nd = ndjson_path.with_name(ndjson_path.stem + "_edited.ndjson")
            _write_ndjson(new_nd, items)
    elif folder and folder.exists():
        for f in folder.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                if do_move(obj):
                    affected += 1
                    with open(f, "w", encoding="utf-8") as fh:
                        json.dump(obj, fh, ensure_ascii=False, indent=2)
            except Exception:
                continue

    if persist_rule and apply_all and match_value:
        rules = load_rules(rules_path)
        rules.setdefault("relocations", []).append({"from": src_slot, "to": dst_slot, "match": match_value})
        save_rules(rules_path, rules)
    return affected, new_nd


def delete_value(ndjson_path: Optional[Path], folder: Optional[Path], slot: str, match_value: Optional[str],
                 apply_all: bool = True, persist_rule: bool = True,
                 rules_path: Path = Path(RULES_DEFAULT_PATH), only_image_name: Optional[str] = None) -> Tuple[int, Optional[Path]]:
    """Delete a value at slot. If match_value is provided, only delete when equal."""
    affected = 0
    new_nd = None

    def do_delete(entry: Dict[str, Any]) -> bool:
        if only_image_name:
            sfv = str(entry.get("SourceFile") or "")
            if Path(sfv).name.lower() != only_image_name.lower():
                return False
        d, k = _get_slot(entry, slot)
        if not d or not k:
            return False
        if k not in d:
            return False
        v = d.get(k)
        if match_value is not None and str(v) != match_value:
            return False
        try:
            del d[k]
        except Exception:
            return False
        return True

    if ndjson_path and ndjson_path.exists():
        items = _read_ndjson(ndjson_path)
        for obj in items:
            if do_delete(obj):
                affected += 1
        if affected:
            new_nd = ndjson_path.with_name(ndjson_path.stem + "_edited.ndjson")
            _write_ndjson(new_nd, items)
    elif folder and folder.exists():
        for f in folder.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                if do_delete(obj):
                    affected += 1
                    with open(f, "w", encoding="utf-8") as fh:
                        json.dump(obj, fh, ensure_ascii=False, indent=2)
            except Exception:
                continue

    # Persist as replacement to empty string to emulate deletion in future transforms
    if persist_rule and apply_all and match_value is not None:
        rules = load_rules(rules_path)
        rules.setdefault("value_replacements", {})[match_value] = ""
        save_rules(rules_path, rules)
    return affected, new_nd


def apply_rename_map_and_update_data(
    map_path: Path,
    images_root: Path,
    data_ndjson: Optional[Path] = None,
    data_folder: Optional[Path] = None,
) -> Tuple[int, int]:
    """Apply rename map to files and update SourceFile in metadata. Returns (files_renamed, metadata_updates)."""
    mapping: List[Tuple[str, str]] = []
    # Map reader supports CSV or NDJSON lines of {ImageName, ModelTag}
    if map_path.suffix.lower() == ".csv":
        import csv
        with open(map_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                old = str(row.get("ImageName") or "").strip()
                new = str(row.get("ModelTag") or "").strip()
                if old and new:
                    mapping.append((old, new))
    else:
        with open(map_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    old = str(obj.get("ImageName") or "").strip()
                    new = str(obj.get("ModelTag") or "").strip()
                    if old and new:
                        mapping.append((old, new))
                except Exception:
                    continue

    # Rename files under root
    renamed = 0
    for old_name, new_base in mapping:
        # Find file by exact basename
        matches = list(images_root.rglob(old_name))
        for m in matches:
            new_path = m.with_name(new_base + m.suffix)
            try:
                m.rename(new_path)
                renamed += 1
            except Exception:
                pass

    # Update metadata files
    updated = 0
    base_map = {old: new for old, new in mapping}
    def update_sourcefile(sf: str) -> str:
        p = Path(sf)
        if p.name in base_map:
            return str(p.with_name(base_map[p.name] + p.suffix))
        return sf

    if data_ndjson and data_ndjson.exists():
        items = _read_ndjson(data_ndjson)
        for obj in items:
            sf = obj.get("SourceFile")
            if isinstance(sf, str):
                new_sf = update_sourcefile(sf)
                if new_sf != sf:
                    obj["SourceFile"] = new_sf
                    updated += 1
        out = data_ndjson.with_name(data_ndjson.stem + "_renamed.ndjson")
        _write_ndjson(out, items)
    if data_folder and data_folder.exists():
        for f in data_folder.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                sf = obj.get("SourceFile")
                if isinstance(sf, str):
                    new_sf = update_sourcefile(sf)
                    if new_sf != sf:
                        obj["SourceFile"] = new_sf
                        with open(f, "w", encoding="utf-8") as fh:
                            json.dump(obj, fh, ensure_ascii=False, indent=2)
                        updated += 1
            except Exception:
                continue
    return renamed, updated
