from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass
class AppConfig:
    host: str
    port: int


@dataclass
class AudioConfig:
    device_name: Optional[str]
    sample_rate: int
    chunk_size: int
    sample_format: str


@dataclass
class DetectionConfig:
    window_seconds: float
    capture_seconds: float
    threshold_stddev_multiplier: float
    threshold_rms_offset: float


@dataclass
class StorageConfig:
    records_dir: Path
    database_path: Path


@dataclass
class ClassificationConfig:
    enabled: bool
    label: str
    score_threshold: float
    model_path: Path
    class_map_path: Path
    retained_labels: list[str]


@dataclass
class RetentionConfig:
    max_record_days: int
    max_records: int


@dataclass
class LoggingConfig:
    level: str
    file_path: Path


@dataclass
class Settings:
    app: AppConfig
    audio: AudioConfig
    detection: DetectionConfig
    storage: StorageConfig
    classification: ClassificationConfig
    retention: RetentionConfig
    logging: LoggingConfig


def load_settings(config_path: Optional[Union[Path, str]] = None) -> Settings:
    root = Path(__file__).resolve().parent.parent
    if config_path is None:
        path = root / "config" / "default.yaml"
        base_dir = root
    else:
        path = Path(config_path).expanduser().resolve()
        base_dir = path.parent
    payload = _read_yaml(path)

    return Settings(
        app=AppConfig(**payload["app"]),
        audio=AudioConfig(
            device_name=payload["audio"].get("device_name"),
            sample_rate=payload["audio"]["sample_rate"],
            chunk_size=payload["audio"]["chunk_size"],
            sample_format=payload["audio"]["format"],
        ),
        detection=_resolve_detection_config(payload["detection"]),
        storage=StorageConfig(
            records_dir=_resolve_path(base_dir, payload["storage"]["records_dir"]),
            database_path=_resolve_path(base_dir, payload["storage"]["database_path"]),
        ),
        classification=_resolve_classification_config(
            base_dir,
            payload["classification"],
        ),
        retention=RetentionConfig(**payload["retention"]),
        logging=LoggingConfig(
            level=payload["logging"]["level"],
            file_path=_resolve_path(base_dir, payload["logging"]["file_path"]),
        ),
    )


def _resolve_classification_config(
    base_dir: Path,
    payload: dict[str, Any],
) -> ClassificationConfig:
    return ClassificationConfig(
        enabled=payload["enabled"],
        label=payload["label"],
        score_threshold=payload.get("score_threshold", 0.2),
        model_path=_resolve_path(base_dir, payload["model_path"]),
        class_map_path=_resolve_path(base_dir, payload["class_map_path"]),
        retained_labels=list(payload.get("retained_labels", [])),
    )


def _resolve_detection_config(payload: dict[str, Any]) -> DetectionConfig:
    return DetectionConfig(
        window_seconds=payload["window_seconds"],
        capture_seconds=payload["capture_seconds"],
        threshold_stddev_multiplier=payload.get("threshold_stddev_multiplier", 1.0),
        threshold_rms_offset=payload.get("threshold_rms_offset", 100.0),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(base_dir: Path, value: Union[Path, str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
