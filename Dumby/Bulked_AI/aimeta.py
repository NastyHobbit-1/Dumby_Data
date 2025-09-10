"""
aimeta.py

All-in-one high-throughput pipeline for image metadata:
- Recursively scan an input directory for images (configurable extensions)
- Batch into file lists and run ExifTool in parallel (robust auto-install / discovery)
- Stream + normalize metadata into canonical AI fields
- Output NDJSON (one JSON per line) and optional per-entry JSON files

Features merged from prior variants:
- Configurable image extensions (--exts)
- Skip stages: --skip-extract and --skip-parse
- Custom NDJSON output path (--out) with --ndjson as a backward-compatible alias
- Multi-core: separate --exif-workers and --parse-workers
- Choice of exif executor: --exif-executor thread|process (default thread; process available)
- Auto-install Python deps (orjson, ijson) with fallback to stdlib json
- ExifTool auto-install via winget/choco or SourceForge/official zip fallback
- Rich ExifTool flags for deeper metadata; clean NDJSON overwrite
- Regex enhancements & fuzzy sampler fallback; LoRA/lyco merging and coalescing
"""

from __future__ import annotations
import argparse
import os
import sys
import re
import json
import shutil
import subprocess
import textwrap
import zipfile
import tempfile
import decimal
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------- Defaults -----------------------
# Core configuration for input/output locations, batching, and parallelism.
DEFAULT_INPUT_DIR = r"D:\AI"
DEFAULT_WORK_DIR = r"D:\AI_JSON"
DEFAULT_BATCH_SIZE = 1000
DEFAULT_EXIF_WORKERS = max(1, (os.cpu_count() or 4))     # parallel exiftool batches
DEFAULT_PARSE_WORKERS = max(1, (os.cpu_count() or 4))     # parallel normalization
DEFAULT_OUT_PATH = "structured_metadata.ndjson"
DEFAULT_WRAP_WIDTH = 0
DEFAULT_EXTS = "jpg,jpeg,png,tif,tiff,webp,heic"
DEFAULT_EXECUTOR = "thread"  # exif executor: thread | process
PROGRESS_EVERY = 2000      # normalization progress tick

# ----------------------- Dependency bootstrap ----------------------------
# Lazily ensure optional JSON speedups (orjson, ijson) are available.
# Falls back to stdlib json if not installed.


def ensure_pip_package(pkg_name: str) -> bool:
    """Ensure a package is importable; auto-install with pip if missing. Returns True if importable."""
    try:
        __import__(pkg_name)
        return True
    except Exception:
        try:
            print(f"Installing missing package: {pkg_name} ...", file=sys.stderr)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade",
                                  pkg_name], stdout=sys.stdout, stderr=sys.stderr)
            __import__(pkg_name)
            return True
        except Exception as e:
            print(f"WARNING: Failed to install {pkg_name}: {e}", file=sys.stderr)
            return False


HAVE_ORJSON = ensure_pip_package("orjson")
HAVE_IJSON = ensure_pip_package("ijson")

if HAVE_ORJSON:
    import orjson  # type: ignore
if HAVE_IJSON:
    import ijson   # type: ignore


def fast_loads(buf: str | bytes) -> Any:
    """Load JSON quickly using orjson if present, else stdlib json."""
    if HAVE_ORJSON:
        return orjson.loads(buf if isinstance(buf, (bytes, bytearray)) else buf.encode("utf-8"))
    return json.loads(buf)


def _default_json(o: Any) -> Any:
    # Normalize types not directly serializable by orjson/json
    if isinstance(o, decimal.Decimal):
        try:
            # Prefer int if integral to avoid float artifacts
            if o == o.to_integral_value():
                return int(o)
        except Exception:
            pass
        return float(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return list(o)
    return str(o)


def fast_dump_bytes(obj: Any, pretty: bool = False) -> bytes:
    """Serialize an object to UTF-8 bytes, optionally pretty-printed."""
    if HAVE_ORJSON:
        opt = 0
        if pretty:
            opt |= orjson.OPT_INDENT_2
        return orjson.dumps(obj, option=opt | orjson.OPT_NON_STR_KEYS, default=_default_json)
    s = json.dumps(obj, ensure_ascii=False, indent=(2 if pretty else None))
    return s.encode("utf-8")


def fast_dumps_str(obj: Any, pretty: bool = False) -> str:
    """Serialize an object to a Unicode string, using the fast byte path."""
    if HAVE_ORJSON:
        return fast_dump_bytes(obj, pretty).decode("utf-8")
    return json.dumps(obj, ensure_ascii=False, indent=(2 if pretty else None), default=_default_json)

# ----------------------- ExifTool discovery/installation -----------------
# Find ExifTool or install it to enable robust metadata extraction.


def which(program: str) -> Optional[str]:
    return shutil.which(program)


def try_install_exiftool() -> Optional[str]:
    """
    Try to install exiftool using winget or choco. If both fail, download official zip.
    Returns path to exiftool.exe or None on failure.
    """
    # 1) winget
    try:
        print("Trying winget install for exiftool...", file=sys.stderr)
        subprocess.run(["winget", "install", "--id", "PhilHarvey.ExifTool", "-e", "-h",
                        "--accept-package-agreements", "--accept-source-agreements"], check=False)
    except Exception:
        pass
    p = which("exiftool.exe") or which("exiftool")
    if p:
        return p

    # 2) choco
    try:
        print("Trying choco install for exiftool...", file=sys.stderr)
        subprocess.run(["choco", "install", "exiftool", "-y"], check=False)
    except Exception:
        pass
    p = which("exiftool.exe") or which("exiftool")
    if p:
        return p

    # 3) Direct download from SourceForge (preferred) then official site as fallback
    zip_urls = [
        "https://sourceforge.net/projects/exiftool/files/exiftool-13.36_64.zip/download",
        "https://exiftool.org/exiftool-13.36_64.zip",
    ]
    for zip_url in zip_urls:
        try:
            import urllib.request
            tmpdir = Path(tempfile.mkdtemp(prefix="exiftool_dl_"))
            zip_path = tmpdir / "exiftool.zip"
            print(f"Downloading exiftool from {zip_url} ...", file=sys.stderr)
            # handle redirect pages by following URL opener
            opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(zip_url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)
            exe = None
            # common exe names in the zip
            for cand in tmpdir.rglob("exiftool(-k).exe"):
                exe = cand
                break
            if not exe:
                for cand in tmpdir.rglob("exiftool.exe"):
                    exe = cand
                    break
            if exe:
                target_dir = Path(os.environ.get("LOCALAPPDATA", str(tmpdir))) / "exiftool"
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / "exiftool.exe"
                shutil.copy2(exe, target)
                print(f"Installed exiftool to {target}", file=sys.stderr)
                return str(target)
        except Exception as e:
            print(f"WARNING: Failed to fetch exiftool zip from {zip_url}: {e}", file=sys.stderr)

    return None


def find_or_install_exiftool(user_path: Optional[str]) -> Optional[str]:
    # Explicit path
    if user_path:
        p = Path(user_path)
        if p.exists():
            return str(p)

    # Common locations
    common = [
        which("exiftool.exe"),
        which("exiftool"),
        r"C:\Program Files\ExifTool\exiftool.exe",
        r"C:\Program Files (x86)\ExifTool\exiftool.exe",
        r"D:\exiftool\exiftool.exe",
        r"D:\exiftool-13.36_64\exiftool.exe",
    ]
    for c in common:
        if c and Path(c).exists():
            return str(Path(c))

    # Try installing
    return try_install_exiftool()

# ----------------------- Scanning, batching, exiftool --------------------
# Scan images, create batches, and run ExifTool in parallel to JSON files.


def parse_exts(csv: str) -> set[str]:
    out = set()
    for part in csv.split(","):
        s = part.strip().lower()
        if not s:
            continue
        if s.startswith("."):
            s = s[1:]
        out.add(s)
    return out


def scan_images(input_dir: Path, exts: set[str]) -> List[Path]:
    files = []
    for p in input_dir.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower()
            if ext.startswith("."):
                ext = ext[1:]
            if ext in exts:
                files.append(p.resolve())
    files.sort(key=lambda x: str(x).lower())
    return files


def chunk(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def write_filelists(work_dir: Path, files: List[Path], batch_size: int) -> List[Tuple[Path, Path]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    pairs: List[Tuple[Path, Path]] = []
    for idx, sub in enumerate(chunk(files, batch_size), start=1):
        num = f"{idx:04d}"
        list_file = work_dir / f"filelist_{num}.txt"
        out_file = work_dir / f"metadata_batch_{num}.json"
        with open(list_file, "w", encoding="utf-8") as f:
            for sp in sub:
                f.write(str(sp) + "\n")
        pairs.append((list_file, out_file))
    return pairs


def exiftool_cmd(exe: str, list_file: Path) -> List[str]:
    return [
        exe, "-json", "-fast2", "-n", "-m",
        "-charset", "filename=UTF8",
        "-api", "LargeFileSupport=1",
        "-api", "RequestAll=3",
        "-api", "QuickTimeUTC",
        "-ee3", "-G:all", "-struct", "-j",
        "-@", str(list_file),
    ]


def run_exiftool_one(exe: str, list_file: Path, out_file: Path) -> Tuple[Path, bool, str]:
    try:
        proc = subprocess.run(exiftool_cmd(exe, list_file), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        if proc.returncode != 0:
            return (out_file, False, proc.stderr.decode("utf-8", "ignore"))
        with open(out_file, "wb") as f:
            f.write(proc.stdout)
        return (out_file, True, "")
    except Exception as e:
        return (out_file, False, str(e))


def run_exiftool_batches(exe: str, pairs: List[Tuple[Path, Path]], mode: str, workers: int) -> None:
    mode = (mode or DEFAULT_EXECUTOR).lower()
    print(f"Running exiftool on {len(pairs)} batches with {workers} {mode}s...", file=sys.stderr)
    ok = 0
    fail = 0
    Exec = ThreadPoolExecutor if mode == "thread" else ProcessPoolExecutor
    with Exec(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(run_exiftool_one, exe, lf, of) for lf, of in pairs]
        for fut in as_completed(futs):
            out_file, success, err = fut.result()
            if success:
                ok += 1
                if ok % 10 == 0:
                    print(f"  exiftool OK: {ok}/{len(pairs)}", file=sys.stderr)
            else:
                fail += 1
                print(f"  exiftool FAILED: {out_file.name} :: {err}", file=sys.stderr)
    print(f"ExifTool done. ok={ok}, failed={fail}", file=sys.stderr)


# ----------------------- Normalization -----------------------------------
# Parse and normalize raw metadata into canonical AI fields.
FLAGS = re.IGNORECASE | re.DOTALL
PAT: Dict[str, re.Pattern[str]] = {
    "Prompt": re.compile(r"^(.*?)(?=\s*Negative\s+prompt\s*:)", FLAGS),
    "Negative Prompt": re.compile(
        r"Negative\s+prompt\s*:\s*(.*?)(?=\b("
        r"Steps?|Sampler|Seed|CFG|Size|Model|Version|Lora|LoRA|Civitai|Clip|VAE|Scheduler|Hires"
        r")\b|$)",
        FLAGS,
    ),
    "Steps": re.compile(r"\bSteps?[:=]?\s*(\d+)\b", FLAGS),
    "Sampler": re.compile(r"\bSampler[:=]?\s*([^\|,\n]+)", FLAGS),
    "CFG Scale": re.compile(r"\bCFG\s*scale[:=]?\s*([\d\.]+)\b", FLAGS),
    "Seed": re.compile(r"\bSeed[:=]?\s*(\d+)\b", FLAGS),
    "Size": re.compile(r"\bSize[:=]?\s*(\d+\s*x\s*\d+)\b", FLAGS),
    "Scheduler": re.compile(r"\bScheduler[:=]?\s*([\w\-]+)", FLAGS),
    "Model": re.compile(r"\bModel[:=]?\s*([^\|,\n]+)", FLAGS),
    "Model Hash": re.compile(r"\bModel\s+hash[:=]?\s*([0-9a-f]{6,})\b", FLAGS),
    "Clip Skip": re.compile(r"\bClip\s*skip[:=]?\s*(\d+)\b", FLAGS),
    "Denoising Strength": re.compile(r"\bDenoising\s+strength[:=]?\s*([\d\.]+)\b", FLAGS),
    "VAE": re.compile(r"\bvae[\"']?\s*[:=]\s*\"?([0-9a-f]{6,})\"?\b", FLAGS),
    "Generation Time": re.compile(r"\b(Created|Generated|generation\s*time)\s*(Date)?[:=]?\s*([\d\-T:\.Z]+)", FLAGS),
    "Hires Upscale": re.compile(r"\bHires\s+upscale[:=]?\s*([0-9\.]+)", FLAGS),
    "Hires Steps": re.compile(r"\bHires\s+steps[:=]?\s*(\d+)", FLAGS),
    "Upscaler": re.compile(r"\bHires\s+upscaler[:=]?\s*([^\|,\n]+)", FLAGS),
}

KNOWN_SAMPLERS = [
    "Euler", "Euler a", "DDIM", "PLMS", "DPM++ 2M Karras", "DPM++ SDE", "Heun", "UniPC",
    "dpmpp_2m_sde_gpu", "DPM++ 2M SDE Karras", "DPM++ SDE Karras"
]
# Precompile a fuzzy sampler regex for fallback
def _escape(s: str) -> str: return re.escape(s)


SAMPLER_FUZZY = re.compile(r"\b(" + "|".join(_escape(s)
                           for s in sorted(KNOWN_SAMPLERS, key=len, reverse=True)) + r")\b", re.IGNORECASE)

LORA_TAG = re.compile(r"<lora:([^:>]+)(?::([^>]+))?>", FLAGS)
LYCO_TAG = re.compile(r"<lyco:([^:>]+)(?::([^>]+))?>", FLAGS)
LORA_HASH = re.compile(r"Lora\s+hashes?\s*:\s*(.+)$", FLAGS)
LORA_PAIR = re.compile(r"([A-Za-z0-9_\-\.]+)\s*:\s*([0-9a-f]{6,})", FLAGS)
CIVITAI_RES = re.compile(r"Civitai\s+resources\s*:\s*(\[.*\])", FLAGS)
MODEL_ALT = re.compile(r"\b(model|checkpoint)\s*[:=]\s*([^\|,\n]+)", FLAGS)
# Generic fallback: capture up to pipe/comma/newline; allow spaces
MODEL_ANY = re.compile(r"(?is)\bModel\s*[:=]\s*([^\|,\r\n]+)")

AI_FIELDS: List[str] = [
    "Prompt", "Negative Prompt", "Model", "Model Version", "Sampler", "Scheduler",
    "Steps", "Seed", "CFG Scale", "Clip Skip", "Denoising Strength",
    "Hires Upscale", "Hires Steps", "Size", "Face Restoration", "Batch Size",
    "Width", "Height", "Upscaler", "VAE", "Refiner", "ControlNet", "LoRA",
    "Tiling", "Restore Faces", "Eta", "Guidance", "Strength", "Noise Level",
    "Style", "Source", "Generation Time", "Device", "Backend", "Model Hash"
]

# Common non-AI metadata groups where 'Model' means camera model, not AI model
NON_AI_GROUPS = {
    "EXIF", "EXIFIFD", "IFD0", "IFD1", "GPS", "DNG", "JFIF", "ICC_PROFILE", "COMPOSITE", "XMP", "IPTC",
    "PHOTOSHOP", "PNG", "FILE", "QUICKTIME", "MAKERNOTES", "APP14", "JUMD", "TIFF", "ADOBE", "NIKON",
    "CANON", "SONY", "PENTAX", "LEICA", "FUJIFILM", "OLYMPUS", "PANASONIC", "SIGMA", "KODAK", "RICOH"
}


def to_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    return x if isinstance(x, str) else str(x)


def wrap_lines(s: str, width: int) -> str:
    if not isinstance(s, str) or width <= 0 or len(s) <= width:
        return s
    return "\n".join(textwrap.wrap(s, width=width))


def wrap_recursive(obj: Any, width: int) -> Any:
    if isinstance(obj, dict):
        return {k: wrap_recursive(v, width) for k, v in obj.items()}
    if isinstance(obj, list):
        return [wrap_recursive(v, width) for v in obj]
    if isinstance(obj, str):
        return wrap_lines(obj, width)
    return obj


def split_size(ai: Dict[str, Any]) -> None:
    sz = ai.get("Size")
    if isinstance(sz, str) and "x" in sz:
        m = re.match(r"\s*(\d+)\s*x\s*(\d+)\s*$", sz)
        if m:
            ai["Width"] = ai.get("Width") or int(m.group(1))
            ai["Height"] = ai.get("Height") or int(m.group(2))


def coerce_numeric_strings(ai: Dict[str, Any]) -> None:
    for k in list(ai.keys()):
        v = ai[k]
        if isinstance(v, str):
            if v.isdigit():
                ai[k] = int(v)
            else:
                try:
                    fv = float(v)
                    if re.fullmatch(r"[+-]?\d*\.\d+([eE][+-]?\d+)?|[+-]?\d+[eE][+-]?\d+", v):
                        ai[k] = fv
                except Exception:
                    pass


def normalize_lora_list(loras: Any) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    if isinstance(loras, list):
        for item in loras:
            d: Dict[str, Any] = {}
            if isinstance(item, dict):
                d["name"] = to_str(item.get("name") or item.get("modelName"))
                d["version"] = to_str(item.get("version") or item.get("modelVersionName"))
                d["weight"] = to_str(item.get("weight"))
                d["hash"] = to_str(item.get("hash"))
            elif isinstance(item, str):
                d["name"] = item
            if d.get("name"):
                flat.append(d)
    # dedupe by (name, version); coalesce attributes
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for it in flat:
        key = (it.get("name") or "", it.get("version") or "")
        cur = seen.get(key)
        if not cur:
            seen[key] = dict(it)
        else:
            if it.get("weight"):
                cur["weight"] = it["weight"]
            if it.get("hash"):
                cur["hash"] = it["hash"]
    out = list(seen.values())
    out.sort(key=lambda d: (d.get("name") or "", d.get("version") or "", d.get("weight") or "", d.get("hash") or ""))
    return out


def merge_unique_loras(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # set-style merge using tuples, then back to dicts
    def tupify(d: Dict[str, Any]) -> Tuple:
        return tuple(sorted((k, v) for k, v in d.items() if v not in (None, "")))
    s = {tupify(x) for x in a} | {tupify(x) for x in b}
    merged = [dict(t) for t in s]
    return normalize_lora_list(merged)


def parse_lora_tags(s: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in LORA_TAG.finditer(s):
        d = {"name": m.group(1).strip()}
        if m.group(2):
            d["weight"] = m.group(2).strip()
        out.append(d)
    for m in LYCO_TAG.finditer(s):
        d = {"name": m.group(1).strip()}
        if m.group(2):
            d["weight"] = m.group(2).strip()
        out.append(d)
    return out


def parse_lora_hashes_block(s: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    m = LORA_HASH.search(s)
    if not m:
        return out
    for name, hsh in LORA_PAIR.findall(m.group(1)):
        out.append({"name": name, "hash": hsh})
    return out


def parse_civitai_resources_array(s: str) -> List[Dict[str, Any]]:
    try:
        arr = fast_loads(s)
        if isinstance(arr, list):
            out: List[Dict[str, Any]] = []
            for it in arr:
                if not isinstance(it, dict):
                    continue
                res = {}
                if str(it.get("type", "")).lower() == "lora":
                    res["name"] = it.get("modelName")
                    res["version"] = it.get("modelVersionName")
                    hashes = it.get("hashes")
                    if isinstance(hashes, dict):
                        res["hash"] = hashes.get("SHA256") or hashes.get("AutoV2") or hashes.get("Hash")
                if res:
                    out.append(res)
            return out
    except Exception:
        pass
    return []


def parse_text_block(text: str) -> Dict[str, Any]:
    """Extract AI fields from a multi-line Parameters-like text block."""
    res: Dict[str, Any] = {k: None for k in AI_FIELDS}
    txt = re.sub(r'\r\n?', '\n', text).strip()

    # LoRA sources
    lo = parse_lora_tags(txt)
    lh = parse_lora_hashes_block(txt)
    civ: List[Dict[str, Any]] = []
    cm = CIVITAI_RES.search(txt)
    if cm:
        civ = parse_civitai_resources_array(cm.group(1))
    lora_all = merge_unique_loras(normalize_lora_list(lo), merge_unique_loras(
        normalize_lora_list(lh), normalize_lora_list(civ)))
    if lora_all:
        res["LoRA"] = lora_all

    # Standard fields via precompiled patterns
    for field, pat in PAT.items():
        m = pat.search(txt)
        if not m:
            continue
        if field == "Generation Time":
            res[field] = m.group(3).strip()
        else:
            res[field] = m.group(1).strip()

    # Fuzzy sampler fallback
    if not res.get("Sampler"):
        ms = SAMPLER_FUZZY.search(txt)
        if ms:
            res["Sampler"] = ms.group(1)

    # Derive Model (alt)
    if not res.get("Model"):
        m2 = MODEL_ALT.search(txt)
        if m2:
            res["Model"] = m2.group(2).strip()

    split_size(res)
    coerce_numeric_strings(res)
    return res


def parse_json_block(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract AI fields from a structured JSON object from various UIs."""
    res: Dict[str, Any] = {k: None for k in AI_FIELDS}

    def assign(keys: List[str], target: str) -> None:
        for k in keys:
            if k in obj and obj[k] not in (None, "") and res.get(target) in (None, ""):
                res[target] = to_str(obj[k])

    assign(["prompt", "Prompt"], "Prompt")
    assign(["negative_prompt", "Negative Prompt"], "Negative Prompt")
    # Broaden model key aliases from various UIs
    assign([
        "model", "Model", "sd_model_name", "sd_checkpoint", "sd_model_checkpoint",
        "modelName", "base_model", "baseModel", "checkpoint", "main_model"
    ], "Model")
    assign(["model_version", "version", "Model Version", "modelVersion", "model_version_name"], "Model Version")
    assign(["sampler", "Sampler"], "Sampler")
    assign(["scheduler", "Scheduler"], "Scheduler")
    assign(["steps", "Steps"], "Steps")
    assign(["seed", "Seed"], "Seed")
    assign(["cfg_scale", "CFG Scale"], "CFG Scale")
    assign(["clip_skip", "Clip Skip"], "Clip Skip")
    assign(["denoising_strength", "Denoising Strength"], "Denoising Strength")
    assign(["vae", "VAE"], "VAE")
    assign(["model_hash", "Model Hash", "sd_model_hash", "modelHash"], "Model Hash")
    assign(["device", "Device"], "Device")
    assign(["backend", "Backend"], "Backend")
    assign(["style"], "Style")
    assign(["source"], "Source")
    assign(["tiling", "tile"], "Tiling")
    assign(["restore_faces", "Face Restoration", "Restore Faces"], "Restore Faces")
    assign(["eta", "Eta"], "Eta")
    assign(["guidance"], "Guidance")
    assign(["strength"], "Strength")
    assign(["noise_level"], "Noise Level")
    assign(["batch_size"], "Batch Size")
    assign(["hires_upscale"], "Hires Upscale")
    assign(["hires_steps"], "Hires Steps")

    if not res.get("Size") and obj.get("width") and obj.get("height"):
        res["Size"] = f"{obj['width']}x{obj['height']}"

    if "loras" in obj and isinstance(obj["loras"], list):
        parsed = []
        for lora in obj["loras"]:
            if isinstance(lora, dict):
                parsed.append({
                    "name": lora.get("modelName") or lora.get("name"),
                    "version": lora.get("modelVersionName"),
                    "weight": lora.get("weight"),
                    "hash": lora.get("hash"),
                })
            elif isinstance(lora, str):
                parsed.append({"name": lora})
        if parsed:
            res["LoRA"] = normalize_lora_list(parsed)

    split_size(res)
    coerce_numeric_strings(res)
    return res


def collect_candidates(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Sweep a raw ExifTool entry for likely AI metadata and coalesce it.

    - Lifts structured AI fields if present
    - Parses embedded JSON strings
    - Parses free-form parameter text
    - Gathers LoRA model references
    """
    ai: Dict[str, Any] = {k: None for k in AI_FIELDS}
    for key, val in entry.items():
        if key in AI_FIELDS and val not in (None, ""):
            ai[key] = to_str(val)
            continue
        if isinstance(key, str) and ":" in key:
            prefix, suffix = key.split(":", 1)
            prefix = str(prefix or "").strip()
            suffix = suffix.strip()
            if suffix in AI_FIELDS and entry[key] not in (None, ""):
                # Avoid pulling camera 'EXIF:Model' into AI Model
                if suffix == "Model" and prefix.upper() in NON_AI_GROUPS:
                    pass
                else:
                    ai[suffix] = to_str(entry[key])
                    continue
        if isinstance(val, str):
            s = val.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    d = fast_loads(s)
                    if isinstance(d, dict):
                        parsed = parse_json_block(d)
                        for k, v in parsed.items():
                            if v and not ai.get(k):
                                ai[k] = v
                        continue
                except Exception:
                    pass
            parsed_ai = parse_text_block(s)
            for k, v in parsed_ai.items():
                if v and not ai.get(k):
                    ai[k] = v
            continue
        if isinstance(val, dict):
            parsed_ai = parse_json_block(val)
            for k, v in parsed_ai.items():
                if v and not ai.get(k):
                    ai[k] = v
            continue
        if isinstance(val, list):
            loras: List[Dict[str, Any]] = []
            for item in val:
                if isinstance(item, dict) and any(
                    t in item for t in ("modelName", "modelVersionName", "weight", "name", "hash")
                ):
                    loras.append({
                        "name": item.get("modelName") or item.get("name"),
                        "version": item.get("modelVersionName"),
                        "weight": item.get("weight"),
                        "hash": item.get("hash"),
                    })
            if loras:
                ai["LoRA"] = merge_unique_loras(normalize_lora_list(ai.get("LoRA") or []), normalize_lora_list(loras))

    split_size(ai)
    coerce_numeric_strings(ai)
    if ai.get("LoRA"):
        ai["LoRA"] = normalize_lora_list(ai["LoRA"])
    return ai


def audit_and_fix(structured: Dict[str, Any]) -> Dict[str, Any]:
    """Final fixups and safeguards for the normalized record.

    Ensures Model vs Model Hash separation, prevents camera EXIF 'Model'
    bleed-through, and coerces numeric-like strings.
    """
    ai = structured["AI_Metadata"]
    gen = structured["GeneralMetadata"]
    for gk, gv in list(gen.items()):
        moved = False
        if isinstance(gv, str):
            s = gv.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    d = fast_loads(s)
                    if isinstance(d, dict):
                        parsed = parse_json_block(d)
                        for k, v in parsed.items():
                            if v and not ai.get(k):
                                ai[k] = v
                                moved = True
                        if moved:
                            del gen[gk]
                            continue
                except Exception:
                    pass
            parsed_ai = parse_text_block(s)
            added = False
            for k, v in parsed_ai.items():
                if v and not ai.get(k):
                    ai[k] = v
                    added = True
            if added:
                del gen[gk]
                continue
        elif isinstance(gv, dict):
            parsed_ai = parse_json_block(gv)
            added = False
            for k, v in parsed_ai.items():
                if v and not ai.get(k):
                    ai[k] = v
                    added = True
            if added:
                del gen[gk]
    if ai.get("Model"):
        model_val = str(ai["Model"]).strip()
        if re.fullmatch(r"[0-9a-f]{6,}", model_val, re.IGNORECASE):
            if not ai.get("Model Hash"):
                ai["Model Hash"] = model_val
            ai["Model"] = None
        else:
            mhash = re.fullmatch(r"(?i)\s*hash\s*:\s*([0-9a-f]{6,})\s*", model_val)
            if mhash:
                if not ai.get("Model Hash"):
                    ai["Model Hash"] = mhash.group(1)
                ai["Model"] = None
    full_txt = fast_dumps_str(structured["GeneralMetadata"]) + "\n" + fast_dumps_str(ai)
    m = MODEL_ANY.search(full_txt)
    if m:
        cand = m.group(1).strip()
        if not re.fullmatch(r"[0-9a-f]{6,}", cand, re.IGNORECASE):
            if not ai.get("Model"):
                ai["Model"] = cand
    split_size(ai)
    coerce_numeric_strings(ai)
    if ai.get("LoRA"):
        ai["LoRA"] = normalize_lora_list(ai["LoRA"])
    structured["AI_Metadata"] = ai
    structured["GeneralMetadata"] = gen
    return structured


def sort_structure(structured: Dict[str, Any]) -> Dict[str, Any]:
    """Stable, readable key ordering for output (AI fields first)."""
    ai = structured.get("AI_Metadata", {})
    gen = structured.get("GeneralMetadata", {})
    if isinstance(ai.get("LoRA"), list):
        ai["LoRA"] = normalize_lora_list(ai["LoRA"])
    ordered_ai: Dict[str, Any] = {}
    seen = set()
    for k in AI_FIELDS:
        if k in ai and ai[k] not in (None, "", []):
            ordered_ai[k] = ai[k]
            seen.add(k)
    for k in sorted(set(ai.keys()) - seen):
        if ai[k] not in (None, "", []):
            ordered_ai[k] = ai[k]
    ordered_gen: Dict[str, Any] = {}
    for k in sorted(gen.keys(), key=lambda x: str(x).lower()):
        v = gen[k]
        if v not in (None, "", []):
            ordered_gen[k] = v
    return {
        "SourceFile": structured.get("SourceFile", "unknown"),
        "AI_Metadata": ordered_ai,
        "GeneralMetadata": ordered_gen,
    }


def process_entry(entry: Dict[str, Any], wrap_width: int) -> Dict[str, Any]:
    """Normalize one raw entry into the final structured shape."""
    ai = collect_candidates(entry)
    general = {k: v for k, v in entry.items() if k not in AI_FIELDS}
    structured = {"SourceFile": entry.get("SourceFile", entry.get("Source", "unknown")),
                  "GeneralMetadata": general, "AI_Metadata": ai}
    structured = audit_and_fix(structured)
    if wrap_width and wrap_width > 0:
        structured["AI_Metadata"] = wrap_recursive(structured["AI_Metadata"], wrap_width)
        structured["GeneralMetadata"] = wrap_recursive(structured["GeneralMetadata"], wrap_width)
    return sort_structure(structured)


def stream_entries(path: Path) -> Iterable[Dict[str, Any]]:
    """Iterate entries from a batch JSON file, using ijson if available."""
    if HAVE_IJSON:
        with open(path, "rb") as f:
            for item in ijson.items(f, "item"):
                if isinstance(item, dict):
                    yield item
    else:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    yield it

# ----------------------- Orchestration -----------------------------------


def main():
    """CLI entry point: scan/extract, then normalize to NDJSON output."""
    ap = argparse.ArgumentParser(description="Unified AI metadata pipeline (scan + exiftool + normalize).")
    ap.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Input directory of images (recursive)")
    ap.add_argument("--work", default=DEFAULT_WORK_DIR, help="Work directory for filelists and batch JSON")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Files per ExifTool batch")
    ap.add_argument("--exts", default=DEFAULT_EXTS, help=f"Comma-separated image extensions (default: {DEFAULT_EXTS})")
    ap.add_argument("--exiftool", default=None, help="Path to exiftool.exe (optional)")
    ap.add_argument("--exif-workers", type=int, default=DEFAULT_EXIF_WORKERS,
                    help="Parallel workers for ExifTool batches")
    ap.add_argument("--exif-executor", choices=["thread", "process"],
                    default=DEFAULT_EXECUTOR, help="Executor type for ExifTool parallelism")
    ap.add_argument("--parse-workers", type=int, default=DEFAULT_PARSE_WORKERS,
                    help="Parallel workers for normalization")
    ap.add_argument("--wrap", type=int, default=DEFAULT_WRAP_WIDTH, help="Wrap long string values at N chars (0 = off)")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print NDJSON lines (slower/larger)")
    ap.add_argument("--out", default=DEFAULT_OUT_PATH,
                    help="NDJSON output path (absolute or relative). If relative, resolves inside --work")
    ap.add_argument("--ndjson", default=None, help="(Deprecated alias) NDJSON output filename; sets --out if provided")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Skip ExifTool extraction stage; process existing metadata_batch_*.json")
    ap.add_argument("--skip-parse", action="store_true", help="Skip normalization stage; only run ExifTool extraction")
    ap.add_argument("--per-file", action="store_true",
                    help="Also write per-entry JSON under work/structured_metadata/ (slower)")
    ap.add_argument("--interactive", action="store_true", help="Run with guided prompts instead of flags")
    args = ap.parse_args()
    if len(sys.argv) == 1 or args.interactive:
        return interactive_main()

    work_dir = Path(args.work)
    work_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(args.input)
    if not in_dir.exists():
        print(f"ERROR: Input directory does not exist: {in_dir}", file=sys.stderr)
        sys.exit(2)

    # out path logic
    out_arg = args.out
    if args.ndjson:  # alias support
        out_arg = args.ndjson
    out_path = Path(out_arg)
    if not out_path.is_absolute():
        out_path = work_dir / out_path

    # 1) Extraction
    if not args.skip_extract:
        exe = find_or_install_exiftool(args.exiftool)
        if not exe or not Path(exe).exists():
            print(
                "ERROR: exiftool not found and installation failed. "
                "Re-run with --skip-extract to process existing batches.",
                file=sys.stderr,
            )
            sys.exit(3)
        exts = parse_exts(args.exts)
        files = scan_images(in_dir, exts)
        print(f"Found {len(files):,} image files under {in_dir}", file=sys.stderr)
        if not files:
            print("Nothing to do.", file=sys.stderr)
            return
        pairs = write_filelists(work_dir, files, max(1, args.batch_size))
        run_exiftool_batches(exe, pairs, args.exif_executor, max(1, args.exif_workers))
    else:
        print("Skipping extraction; will process existing metadata_batch_*.json", file=sys.stderr)

    if args.skip_parse:
        print("Extraction complete (skip-parse active). Exiting before normalization.", file=sys.stderr)
        return

    # 2) Normalization
    batch_files = sorted(work_dir.glob("metadata_batch_*.json"))
    if not batch_files:
        print(f"WARNING: No batch JSON found in {work_dir}. Pattern: metadata_batch_*.json", file=sys.stderr)
        return

    per_file_dir = work_dir / "structured_metadata"
    if args.per_file:
        per_file_dir.mkdir(parents=True, exist_ok=True)

    out_path.unlink(missing_ok=True)
    total = 0
    tick = 0

    print(f"Normalizing {len(batch_files)} batch files with {max(1, args.parse_workers)} workers...", file=sys.stderr)
    with ProcessPoolExecutor(max_workers=max(1, args.parse_workers)) as ex:
        futures = []
        WAVE = 4000

        def flush_wave(futs: List):
            nonlocal total, tick
            for fut in as_completed(futs):
                obj = fut.result()
                with open(out_path, "ab") as f:
                    f.write(fast_dump_bytes(obj, pretty=args.pretty))
                    f.write(b"\n")
                if args.per_file:
                    base = Path(obj.get("SourceFile") or "unknown").stem or "unknown"
                    out_p = per_file_dir / f"{base}_metadata.json"
                    with open(out_p, "wb") as pf:
                        pf.write(fast_dump_bytes(obj, pretty=args.pretty))
                total += 1
                tick += 1
                if tick >= PROGRESS_EVERY:
                    print(f"normalized {total:,} entries...", file=sys.stderr)
                    tick = 0

        for bf in batch_files:
            for entry in stream_entries(bf):
                futures.append(ex.submit(process_entry, entry, args.wrap))
                if len(futures) >= WAVE:
                    flush_wave(futures)
                    futures = []
        if futures:
            flush_wave(futures)

    print(f"Done. NDJSON: {out_path}  |  Total entries: {total:,}", file=sys.stderr)


# ----------------------- Interactive Mode ---------------------------------

RULES_DEFAULT_PATH = "export_rules.json"


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


def _load_rules(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"value_replacements": {}, "relocations": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return {"value_replacements": {}, "relocations": []}
        obj.setdefault("value_replacements", {})
        obj.setdefault("relocations", [])
        return obj
    except Exception:
        return {"value_replacements": {}, "relocations": []}


def _save_rules(path: Path, rules: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)


def _manage_rules_interactive(rules_path: Path) -> None:
    rules = _load_rules(rules_path)
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
                print("Added relocation rule.")
        elif sel == 3:
            print(json.dumps(rules, ensure_ascii=False, indent=2))
        else:
            _save_rules(rules_path, rules)
            print(f"Saved rules to {rules_path}")
            return


def interactive_main() -> None:
    print("Welcome to AIMeta interactive mode.")
    top = _prompt_select(
        "Select mode",
        [
            "Scan (Extract + Normalize)",
            "Repl/Relo (Manage replacements/relocations)",
            "Exit",
        ],
    )
    if top == 2:
        rp = Path(RULES_DEFAULT_PATH)
        _manage_rules_interactive(rp)
        return
    if top == 3:
        return

    # Scan mode
    in_dir = Path(_prompt_input("Enter source/input directory", DEFAULT_INPUT_DIR))
    work_dir = Path(_prompt_input("Enter work/output directory", DEFAULT_WORK_DIR))
    work_dir.mkdir(parents=True, exist_ok=True)

    want_install = _prompt_yes_no("If required, install missing programs (ExifTool, Python packages)?", True)

    # Optional: ensure fast libs
    global HAVE_ORJSON, HAVE_IJSON
    if not HAVE_ORJSON and want_install:
        HAVE_ORJSON = ensure_pip_package("orjson")
        if HAVE_ORJSON:
            import importlib
            importlib.invalidate_caches()
            globals()["orjson"] = __import__("orjson")
    if not HAVE_IJSON and want_install:
        HAVE_IJSON = ensure_pip_package("ijson")
        if HAVE_IJSON:
            import importlib
            importlib.invalidate_caches()
            globals()["ijson"] = __import__("ijson")

    # Decide extraction
    have_batches = any(work_dir.glob("metadata_batch_*.json"))
    skip_extract = False
    if have_batches:
        skip_extract = _prompt_yes_no("Existing batch JSON detected. Use them and skip extraction?", True)

    exe = None
    if not skip_extract:
        candidate = _prompt_input("Provide exiftool path or leave blank to auto-detect")
        if candidate and Path(candidate).exists():
            exe = candidate
        if not exe:
            exe = which("exiftool.exe") or which("exiftool")
        if not exe and want_install:
            if _prompt_yes_no("ExifTool not found. Install now?", True):
                exe = try_install_exiftool()
        if not exe:
            print("ERROR: ExifTool is required for extraction and was not found.", file=sys.stderr)
            return

    per_file = _prompt_yes_no("Also write per-entry JSON files?", False)
    pretty = _prompt_yes_no("Pretty-print NDJSON lines?", False)
    wrap_on = _prompt_yes_no("Wrap long text fields?", False)
    wrap = int(_prompt_input("Wrap width", "100")) if wrap_on else 0
    pw_sel = _prompt_select("Parse workers", ["All cores", "Specify number"])
    parse_workers = max(1, (os.cpu_count() or 4)) if pw_sel == 1 else max(1, int(_prompt_input("Workers count", "4")))
    out_path = Path(_prompt_input("NDJSON output filename/path", DEFAULT_OUT_PATH))
    if not out_path.is_absolute():
        out_path = work_dir / out_path

    # Extraction
    if not skip_extract:
        exts = parse_exts(DEFAULT_EXTS)
        files = scan_images(in_dir, exts)
        print(f"Found {len(files):,} image files under {in_dir}", file=sys.stderr)
        if not files:
            print("Nothing to do.", file=sys.stderr)
            return
        pairs = write_filelists(work_dir, files, max(1, DEFAULT_BATCH_SIZE))
        run_exiftool_batches(exe, pairs, DEFAULT_EXECUTOR, max(1, DEFAULT_EXIF_WORKERS))
    else:
        print("Skipping extraction; will process existing metadata_batch_*.json", file=sys.stderr)

    # Normalization
    batch_files = sorted(work_dir.glob("metadata_batch_*.json"))
    if not batch_files:
        print(f"WARNING: No batch JSON found in {work_dir}. Pattern: metadata_batch_*.json", file=sys.stderr)
        return
    per_file_dir = work_dir / "structured_metadata"
    if per_file:
        per_file_dir.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    total = 0
    tick = 0
    print(f"Normalizing {len(batch_files)} batch files with {parse_workers} workers...", file=sys.stderr)
    with ProcessPoolExecutor(max_workers=max(1, parse_workers)) as ex:
        futures: List[Any] = []
        WAVE = 4000

        def flush_wave(futs: List[Any]) -> None:
            nonlocal total, tick
            for fut in as_completed(futs):
                obj = fut.result()
                with open(out_path, "ab") as f:
                    f.write(fast_dump_bytes(obj, pretty=pretty))
                    f.write(b"\n")
                if per_file:
                    base = Path(obj.get("SourceFile") or "unknown").stem or "unknown"
                    out_p = per_file_dir / f"{base}_metadata.json"
                    with open(out_p, "wb") as pf:
                        pf.write(fast_dump_bytes(obj, pretty=pretty))
                total += 1
                tick += 1
                if tick >= PROGRESS_EVERY:
                    print(f"normalized {total:,} entries...", file=sys.stderr)
                    tick = 0

        for bf in batch_files:
            for entry in stream_entries(bf):
                futures.append(ex.submit(process_entry, entry, wrap))
                if len(futures) >= WAVE:
                    flush_wave(futures)
                    futures = []
        if futures:
            flush_wave(futures)

    print(f"Done. NDJSON: {out_path}  |  Total entries: {total:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
