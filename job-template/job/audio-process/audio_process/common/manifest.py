from typing import Optional

import pandas as pd

from .constants import DEFAULT_AUDIO_PATH_COL
from .paths import assert_file_exists, scan_audio_files, ensure_parent_dir


def load_manifest(input_manifest: str, audio_path_col: str = DEFAULT_AUDIO_PATH_COL) -> pd.DataFrame:
    assert_file_exists(input_manifest, "input_manifest")
    df = pd.read_csv(input_manifest)
    if audio_path_col not in df.columns:
        raise ValueError(
            f"manifest missing audio path column: {audio_path_col}; columns={list(df.columns)}"
        )
    return df


def build_manifest_from_audio_dir(
    input_audio_dir: str,
    audio_path_col: str = DEFAULT_AUDIO_PATH_COL,
    recursive: bool = True,
) -> pd.DataFrame:
    audio_files = scan_audio_files(input_audio_dir, recursive=recursive)
    return pd.DataFrame({audio_path_col: audio_files})


def load_input_as_manifest(
    input_manifest: Optional[str] = "",
    input_audio_dir: Optional[str] = "",
    audio_path_col: str = DEFAULT_AUDIO_PATH_COL,
    recursive: bool = True,
) -> pd.DataFrame:
    input_manifest = input_manifest or ""
    input_audio_dir = input_audio_dir or ""

    if input_manifest.strip():
        return load_manifest(input_manifest, audio_path_col=audio_path_col)
    if input_audio_dir.strip():
        return build_manifest_from_audio_dir(
            input_audio_dir, audio_path_col=audio_path_col, recursive=recursive
        )
    raise ValueError("input_manifest and input_audio_dir cannot both be empty")


def save_manifest(df: pd.DataFrame, output_manifest_path: str) -> None:
    ensure_parent_dir(output_manifest_path)
    df.to_csv(output_manifest_path, index=False)
