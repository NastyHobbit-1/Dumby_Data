from __future__ import annotations
from pathlib import Path
from typing import List, Optional


def prompt_input(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or (default or "")


def prompt_yes_no(question: str, default_yes: bool = True) -> bool:
    y = "Y" if default_yes else "y"
    n = "n" if default_yes else "N"
    while True:
        ans = input(f"{question} ({y}/{n}): ").strip().lower()
        if not ans:
            return default_yes
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def menu_select(title: str, options: List[str]) -> int:
    print(title)
    for i, name in enumerate(options, 1):
        print(f"{i} - {name}")
    while True:
        sel = input(f"Select 1..{len(options)}: ").strip()
        if sel.isdigit():
            n = int(sel)
            if 1 <= n <= len(options):
                return n


def strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def normalize_output_path(user_input: str, default_filename: str) -> Path:
    cleaned = strip_quotes(user_input)
    p = Path(cleaned)
    if p.exists() and p.is_dir():
        return p / default_filename
    if cleaned.endswith(("/", "\\")):
        return Path(cleaned) / default_filename
    return p
