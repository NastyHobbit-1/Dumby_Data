import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "Dumby" / "Bulked_AI"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.meta_export import (  # noqa: E402
    export_ndjson_rename_map,
    sanitize_filename_component,
)


def test_sanitize_filename_component_normalizes_invalid_characters():
    dirty = '  Fancy<>:"Model*Name??  '
    assert sanitize_filename_component(dirty) == "Fancy Model Name"


def test_export_ndjson_rename_map_sanitizes_model_tags(tmp_path):
    ndjson_path = tmp_path / "structured_metadata.ndjson"
    entries = [
        {
            "SourceFile": "C:/images/fancy_one.png",
            "AI_Metadata": {"Model": 'Fancy<>:"Model*Name??  '},
        },
        {
            "SourceFile": "C:/images/fancy_two.png",
            "AI_Metadata": {"Model": 'Fancy<>:"Model*Name??  '},
        },
    ]
    with open(ndjson_path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry))
            handle.write("\n")

    csv_path = tmp_path / "rename_map.csv"
    export_ndjson_rename_map(ndjson_path, csv_path, sanitize=True, fmt="csv")
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["ImageName", "ModelTag"]
    assert rows[1][1] == "Fancy Model Name 1"
    assert rows[2][1] == "Fancy Model Name 2"

    ndjson_map = tmp_path / "rename_map.ndjson"
    export_ndjson_rename_map(ndjson_path, ndjson_map, sanitize=True, fmt="ndjson")
    with open(ndjson_map, "r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]

    assert lines[0]["ModelTag"] == "Fancy Model Name 1"
    assert lines[1]["ModelTag"] == "Fancy Model Name 2"
