from __future__ import annotations

import argparse
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_detect.classifier import Classifier
from audio_detect.collector import AudioInput
from audio_detect.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a short clip and classify it")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML configuration file",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=4.0,
        help="Recording duration in seconds",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    samples_dir = ROOT / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(timezone.utc).strftime("sample-%Y%m%d-%H%M%S.wav")
    target = samples_dir / filename
    audio_input = AudioInput(settings.audio)
    target_samples = max(1, int(settings.audio.sample_rate * args.seconds))
    collected = []
    samples_collected = 0
    try:
        while samples_collected < target_samples:
            chunk = audio_input.read_chunk()
            collected.append(np.asarray(chunk, dtype=np.int16).copy())
            samples_collected += int(np.asarray(chunk).shape[0])
    finally:
        audio_input.close()

    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(settings.audio.sample_rate)
        handle.writeframes(np.concatenate(collected).astype(np.int16).tobytes())

    classifier = Classifier(settings.classification)
    classifier.ensure_available()
    result = classifier.classify(target)
    print(result)


if __name__ == "__main__":
    main()
