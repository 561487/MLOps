import random

import librosa
import numpy as np


def normalize_volume(y, target_dbfs: float = -20.0):
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    eps = 1e-10
    rms = np.sqrt(np.mean(np.square(y)) + eps)
    current_db = 20.0 * np.log10(rms + eps)
    gain_db = target_dbfs - current_db
    gain = 10.0 ** (gain_db / 20.0)
    return y * gain


def trim_silence(y, top_db: float = 40.0):
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed


def limit_amplitude(y):
    return np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0)


def add_noise(y, noise_level: float = 0.005):
    y = np.asarray(y, dtype=np.float32)
    noise = np.random.randn(y.size).astype(np.float32)
    return y + noise_level * noise


def change_volume(y, gain_db: float):
    gain = 10.0 ** (gain_db / 20.0)
    return np.asarray(y, dtype=np.float32) * gain


def change_speed(y, speed_rate: float):
    y = np.asarray(y, dtype=np.float32)
    if speed_rate <= 0:
        raise ValueError(f"speed_rate must be positive, got {speed_rate}")
    return librosa.effects.time_stretch(y, rate=speed_rate)


def random_speed(min_speed: float, max_speed: float) -> float:
    return random.uniform(min_speed, max_speed)


def random_volume_gain(min_gain: float, max_gain: float) -> float:
    return random.uniform(min_gain, max_gain)
