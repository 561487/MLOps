import os
from pathlib import Path
from typing import List

from .constants import SUPPORTED_AUDIO_EXTS


def ensure_dir(path: str) -> None:
    if not path:
        raise ValueError("directory path is empty")
    os.makedirs(path, exist_ok=True)


def ensure_parent_dir(file_path: str) -> None:
    if not file_path:
        raise ValueError("file path is empty")
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def assert_file_exists(path: str, name: str = "file") -> None:
    if not path:
        raise ValueError(f"{name} is empty")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def assert_dir_exists(path: str, name: str = "directory") -> None:
    if not path:
        raise ValueError(f"{name} is empty")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def scan_audio_files(input_audio_dir: str, recursive: bool = True) -> List[str]:
    assert_dir_exists(input_audio_dir, "input_audio_dir")
    root = Path(input_audio_dir)
    pattern = "**/*" if recursive else "*"
    files = [
        str(p)
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS
    ]
    return sorted(files)


def safe_stem(path: str) -> str:
    stem = Path(path).stem
    safe_chars = []
    for ch in stem:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars) or "audio"
