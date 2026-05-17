from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from audio_detect.audio_utils import read_wav_mono_float32, resample_audio
from audio_detect.config import ClassificationConfig


@dataclass
class ClassificationResult:
    label: str
    score: float
    top_classes: List[str]


class Classifier:
    def __init__(self, config: ClassificationConfig) -> None:
        self.config = config
        self._labels = []
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self.unavailable_reason: Optional[str] = None

        if self.config.enabled:
            self._labels = self._load_labels(config.class_map_path)
            if not self._labels:
                self.unavailable_reason = (
                    "classification class map is missing or empty: {path}".format(
                        path=config.class_map_path
                    )
                )
            interpreter = self._load_interpreter(config.model_path)
            if interpreter is not None and self._labels:
                self._interpreter = interpreter
                self._interpreter.allocate_tensors()
                self._input_details = self._interpreter.get_input_details()[0]
                self._output_details = self._interpreter.get_output_details()[0]
            elif self.unavailable_reason is None:
                self.unavailable_reason = (
                    "classification model is unavailable: {path}".format(
                        path=config.model_path
                    )
                )
        self._retained_labels = set(self.config.retained_labels)

    @property
    def is_available(self) -> bool:
        return (not self.config.enabled) or self.unavailable_reason is None

    def ensure_available(self) -> None:
        if self.config.enabled and self.unavailable_reason is not None:
            raise RuntimeError(self.unavailable_reason)

    def should_retain(self, classification: ClassificationResult) -> bool:
        if not self._retained_labels:
            return True
        for item in classification.top_classes:
            label = item.split(":", 1)[0].strip()
            if label in self._retained_labels:
                return True
        return classification.label in self._retained_labels

    def classify(self, recording_path: Path) -> ClassificationResult:
        samples, sample_rate = read_wav_mono_float32(recording_path)
        return self.classify_samples(samples, sample_rate)

    def classify_samples(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> ClassificationResult:
        if not self.config.enabled:
            return ClassificationResult(
                label=self.config.label,
                score=0.0,
                top_classes=[self.config.label],
            )

        if self._interpreter is None:
            raise RuntimeError(self.unavailable_reason or "classification model is unavailable")

        waveform = resample_audio(samples, sample_rate, 16000)
        waveform = waveform.astype(np.float32)
        if waveform.ndim != 1:
            waveform = np.squeeze(waveform)

        self._interpreter.resize_tensor_input(
            self._input_details["index"],
            [waveform.shape[0]],
            strict=False,
        )
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()[0]
        self._output_details = self._interpreter.get_output_details()[0]
        self._interpreter.set_tensor(self._input_details["index"], waveform)
        self._interpreter.invoke()

        scores = self._interpreter.get_tensor(self._output_details["index"])
        mean_scores = np.mean(scores, axis=0)
        top_indices = np.argsort(mean_scores)[-3:][::-1]
        top_classes = [
            "{label}:{score:.3f}".format(
                label=self._labels[index] if index < len(self._labels) else str(index),
                score=float(mean_scores[index]),
            )
            for index in top_indices
        ]
        best_index = int(top_indices[0])
        best_score = float(mean_scores[best_index])
        best_label = (
            self._labels[best_index] if best_index < len(self._labels) else self.config.label
        )
        if best_score < self.config.score_threshold:
            best_label = self.config.label

        return ClassificationResult(
            label=best_label,
            score=best_score,
            top_classes=top_classes,
        )

    @staticmethod
    def _load_labels(path: Path) -> List[str]:
        if not path.exists():
            return []

        labels = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                labels.append(row["display_name"])
        return labels

    @staticmethod
    def _load_interpreter(model_path: Path):
        if not model_path.exists():
            return None
        try:
            import tflite_runtime.interpreter as tflite

            return tflite.Interpreter(model_path=str(model_path))
        except Exception:
            return None
