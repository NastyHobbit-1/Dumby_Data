from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

# Default rules path inside Dumby/Boxxed_Data/Rules
_DUMBY_ROOT = Path(__file__).resolve().parents[2]
RULES_DEFAULT_PATH = str(_DUMBY_ROOT / "Boxxed_Data" / "Rules" / "export_rules.json")

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
            if match is not None and str(val) != str(match):
                continue
            cur = dst_dict.get(dst_key)
            if cur in (None, "") or cur == val:
                dst_dict[dst_key] = val
            try:
                del src_dict[src_key]
            except Exception:
                pass
        except Exception:
            continue


def apply_rules_to_entry(entry: Dict[str, Any], rules: Optional[RulesType]) -> Dict[str, Any]:
    if not rules:
        return entry
    repl = rules.get("value_replacements") or {}
    if isinstance(repl, dict) and repl:
        if isinstance(entry.get("AI_Metadata"), dict):
            entry["AI_Metadata"] = _apply_value_replacements_inplace(entry["AI_Metadata"], repl)
        if isinstance(entry.get("GeneralMetadata"), dict):
            entry["GeneralMetadata"] = _apply_value_replacements_inplace(entry["GeneralMetadata"], repl)
        if isinstance(entry.get("SourceFile"), str):
            sf = entry["SourceFile"]
            entry["SourceFile"] = repl.get(sf, sf)
    rel = rules.get("relocations") or []
    if isinstance(rel, list) and rel:
        _apply_relocations_inplace(entry, rel)
    return entry


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
