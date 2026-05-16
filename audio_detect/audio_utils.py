from __future__ import annotations

import wave
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.signal import resample_poly


def read_wav_mono_float32(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width != 2:
        raise ValueError("only 16-bit PCM WAV files are supported")

    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return (samples.astype(np.float32) / 32768.0, sample_rate)


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32)
    return resample_poly(samples, target_rate, source_rate).astype(np.float32)
