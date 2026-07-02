import numpy as np


def calc_duration(y, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(len(y) / sample_rate)


def calc_rms_db(y) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return -120.0
    eps = 1e-10
    rms = np.sqrt(np.mean(np.square(y)) + eps)
    return float(20.0 * np.log10(rms + eps))


def calc_clipping_ratio(y, threshold: float = 0.99) -> float:
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    return float(np.mean(np.abs(y) >= threshold))


def calc_silence_ratio(y, silence_threshold_db: float = -40.0, frame_size: int = 1024) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return 1.0
    if y.size < frame_size:
        return 1.0 if calc_rms_db(y) < silence_threshold_db else 0.0

    silence_count = 0
    frame_count = 0
    for start in range(0, y.size, frame_size):
        frame = y[start : start + frame_size]
        if frame.size == 0:
            continue
        frame_count += 1
        if calc_rms_db(frame) < silence_threshold_db:
            silence_count += 1
    return float(silence_count / frame_count) if frame_count else 1.0
