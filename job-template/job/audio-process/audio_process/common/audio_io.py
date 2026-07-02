import numpy as np
import soundfile as sf
import librosa

from .paths import ensure_parent_dir


def load_audio(audio_path: str, sr=None, mono: bool = False):
    """
    Load audio with librosa.
    sr=None keeps original sample rate.
    mono=False keeps original channels when possible.
    """
    y, sample_rate = librosa.load(audio_path, sr=sr, mono=mono)
    return y, sample_rate


def get_channel_count(y) -> int:
    arr = np.asarray(y)
    if arr.ndim == 1:
        return 1
    return int(min(arr.shape))


def to_mono(y):
    arr = np.asarray(y)
    if arr.ndim == 1:
        return arr
    # librosa returns channel-first when mono=False: shape=(channels, samples)
    if arr.shape[0] <= arr.shape[-1]:
        return np.mean(arr, axis=0)
    return np.mean(arr, axis=1)


def resample_audio(y, orig_sr: int, target_sr: int):
    if not target_sr or orig_sr == target_sr:
        return y
    return librosa.resample(np.asarray(y), orig_sr=orig_sr, target_sr=target_sr)


def save_audio(audio_path: str, y, sample_rate: int) -> None:
    ensure_parent_dir(audio_path)
    sf.write(audio_path, y, sample_rate)
