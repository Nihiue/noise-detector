from __future__ import annotations

import logging
import math
import threading
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from audio_detect.audio_devices import AudioDeviceInfo, find_input_device
from audio_detect.classifier import Classifier
from audio_detect.config import AudioConfig, DetectionConfig
from audio_detect.storage import EventStore, NoiseEvent

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - backend import is environment-specific
    sd = None


@dataclass
class CollectorRuntimeStatus:
    selected_device_index: Optional[int] = None
    selected_device_name: str = ""
    is_running: bool = False
    last_peak_rms: float = 0.0
    last_peak_dbfs: float = -90.0
    detection_window_seconds: float = 0.0
    capture_seconds: float = 0.0
    noise_floor_rms: float = 0.0
    noise_floor_std_rms: float = 0.0
    trigger_threshold_rms: float = 0.0
    noise_floor_dbfs: float = -90.0
    trigger_threshold_dbfs: float = -90.0
    gate_ready: bool = False
    is_calibrating: bool = False
    last_gate_open: bool = False
    recent_detection_windows: list[dict[str, object]] = field(default_factory=list)
    last_error: str = ""


@dataclass(frozen=True)
class _NoiseGateDecision:
    should_classify: bool
    current_rms: float
    current_dbfs: float
    noise_floor_rms: float
    noise_floor_std_rms: float
    trigger_threshold_rms: float
    noise_floor_dbfs: float
    trigger_threshold_dbfs: float
    is_ready: bool


class _NoiseFloorGate:
    _SILENCE_FLOOR_DBFS = -90.0
    _CALIBRATION_BUCKETS = 20

    def __init__(
        self,
        detection_window_seconds: float,
        threshold_stddev_multiplier: float = 1.0,
    ) -> None:
        self.threshold_stddev_multiplier = threshold_stddev_multiplier
        self._noise_floor_rms: Optional[float] = None
        self._noise_floor_std_rms: float = 0.0
        self._trigger_threshold_rms: Optional[float] = None

    def evaluate(self, rms: float) -> _NoiseGateDecision:
        current_dbfs = self._rms_to_dbfs(rms)
        is_ready = self.is_ready

        return _NoiseGateDecision(
            should_classify=is_ready and rms > self.trigger_threshold_rms,
            current_rms=rms,
            current_dbfs=current_dbfs,
            noise_floor_rms=self.noise_floor_rms,
            noise_floor_std_rms=self.noise_floor_std_rms,
            trigger_threshold_rms=self.trigger_threshold_rms,
            noise_floor_dbfs=self.noise_floor_dbfs,
            trigger_threshold_dbfs=self.trigger_threshold_dbfs,
            is_ready=is_ready,
        )

    def calibrate(self, samples: np.ndarray) -> _NoiseGateDecision:
        rms_values = self._bucket_rms(samples)
        self._noise_floor_rms = float(np.mean(rms_values))
        self._noise_floor_std_rms = float(np.std(rms_values))
        self._trigger_threshold_rms = (
            self._noise_floor_rms
            + self._noise_floor_std_rms * self.threshold_stddev_multiplier
        )
        current_rms = float(self._compute_rms(samples))

        return _NoiseGateDecision(
            should_classify=False,
            current_rms=current_rms,
            current_dbfs=self._rms_to_dbfs(current_rms),
            noise_floor_rms=self.noise_floor_rms,
            noise_floor_std_rms=self.noise_floor_std_rms,
            trigger_threshold_rms=self.trigger_threshold_rms,
            noise_floor_dbfs=self.noise_floor_dbfs,
            trigger_threshold_dbfs=self.trigger_threshold_dbfs,
            is_ready=True,
        )

    @property
    def is_ready(self) -> bool:
        return self._noise_floor_rms is not None and self._trigger_threshold_rms is not None

    @property
    def noise_floor_rms(self) -> float:
        return self._noise_floor_rms or 0.0

    @property
    def noise_floor_std_rms(self) -> float:
        return self._noise_floor_std_rms

    @property
    def trigger_threshold_rms(self) -> float:
        return self._trigger_threshold_rms or 0.0

    @property
    def noise_floor_dbfs(self) -> float:
        return self._rms_to_dbfs(self.noise_floor_rms)

    @property
    def trigger_threshold_dbfs(self) -> float:
        return self._rms_to_dbfs(self.trigger_threshold_rms)

    @classmethod
    def _bucket_rms(cls, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return np.asarray([0.0], dtype=np.float32)
        bucket_count = min(cls._CALIBRATION_BUCKETS, samples.size)
        buckets = np.array_split(samples, bucket_count)
        return np.asarray(
            [cls._compute_rms(bucket) for bucket in buckets if bucket.size > 0],
            dtype=np.float32,
        )

    @staticmethod
    def _compute_rms(samples: np.ndarray) -> float:
        normalized = samples.astype(np.float32)
        return float(np.sqrt(np.mean(normalized * normalized)))

    @classmethod
    def _rms_to_dbfs(cls, rms: float) -> float:
        if rms <= 0.0:
            return cls._SILENCE_FLOOR_DBFS
        dbfs = 20.0 * math.log10(rms / 32768.0)
        return max(cls._SILENCE_FLOOR_DBFS, float(dbfs))


class AudioInputLike(Protocol):
    selected_device: Optional[AudioDeviceInfo]
    last_error: str

    def read_chunk(self) -> np.ndarray: ...

    def close(self) -> None: ...


class AudioInput:
    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_lock = threading.Lock()
        self._buffer_ready = threading.Condition(self._buffer_lock)
        self._stream = None
        self.last_error = ""
        self.selected_device = find_input_device(
            device_name=config.device_name,
        )
        if sd is None:
            raise RuntimeError("sounddevice backend is unavailable")
        if self.selected_device is None:
            raise RuntimeError("no matching input device was found")
        self._open_stream()

    def read_chunk(self) -> np.ndarray:
        try:
            return self._read_stream_chunk()
        except Exception as exc:
            self.last_error = str(exc)
            self.close()
            raise RuntimeError(
                "failed to read from input device: {error}".format(error=exc)
            ) from exc

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        with self._buffer_ready:
            self._buffer.clear()
            self._buffer_ready.notify_all()

    def _open_stream(self) -> None:
        if sd is None or self.selected_device is None:
            return

        def callback(indata, frames, time_info, status) -> None:
            if status:
                self.last_error = str(status)
            chunk = np.squeeze(np.array(indata, copy=True))
            if chunk.ndim == 0:
                chunk = np.asarray([chunk], dtype=np.int16)
            chunk = np.asarray(chunk, dtype=np.int16)
            with self._buffer_ready:
                self._buffer.append(chunk)
                self._buffer_ready.notify()

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.chunk_size,
            channels=1,
            dtype=self.config.sample_format,
            device=self.selected_device.index,
            callback=callback,
        )
        self._stream.start()

    def _read_stream_chunk(self) -> np.ndarray:
        with self._buffer_ready:
            while not self._buffer:
                self._buffer_ready.wait(timeout=1.0)
                if not self._buffer:
                    raise RuntimeError("timed out waiting for audio input stream data")
            chunk = self._buffer.popleft()
        return np.squeeze(chunk)


class CollectorService:
    _CLASSIFICATION_TARGET_RMS = 0.1
    _CLASSIFICATION_MAX_PEAK = 0.95

    def __init__(
        self,
        audio_config: AudioConfig,
        detection_config: DetectionConfig,
        event_store: EventStore,
        classifier: Classifier,
        audio_input: Optional[AudioInputLike] = None,
    ) -> None:
        self.audio_config = audio_config
        self.detection_config = detection_config
        self.event_store = event_store
        self.classifier = classifier
        self._audio_input_factory = None if audio_input is not None else lambda: AudioInput(audio_config)
        self.audio_input = audio_input or self._create_audio_input()
        self.logger = logging.getLogger("audio_detect.collector")
        self._noise_gate = _NoiseFloorGate(
            detection_config.window_seconds,
            threshold_stddev_multiplier=detection_config.threshold_stddev_multiplier,
        )
        self._recent_detection_windows: deque[dict[str, object]] = deque(maxlen=10)
        self._status_lock = threading.Lock()
        self.status = CollectorRuntimeStatus(
            selected_device_index=(
                self.audio_input.selected_device.index
                if self.audio_input.selected_device is not None
                else None
            ),
            selected_device_name=(
                self.audio_input.selected_device.name
                if self.audio_input.selected_device is not None
                else ""
            ),
            is_running=False,
            detection_window_seconds=self.detection_config.window_seconds,
            capture_seconds=self.detection_config.capture_seconds,
            noise_floor_rms=0.0,
            noise_floor_std_rms=0.0,
            trigger_threshold_rms=0.0,
            noise_floor_dbfs=_NoiseFloorGate._SILENCE_FLOOR_DBFS,
            trigger_threshold_dbfs=_NoiseFloorGate._SILENCE_FLOOR_DBFS,
            last_error=self.audio_input.last_error,
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._recovery_wait_seconds = 1.0
        self._calibration_requested = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.logger.info(
            "collector starting device_index=%s device_name=%s",
            self.status.selected_device_index,
            self.status.selected_device_name,
        )
        self._thread = threading.Thread(target=self._run, name="collector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.audio_input.close()
        if self._thread:
            self._thread.join(timeout=2.0)

    def request_calibration(self) -> CollectorRuntimeStatus:
        self._calibration_requested.set()
        self._update_status(is_calibrating=True)
        return self.get_status()

    def _run(self) -> None:
        self._update_status(
            is_running=True,
            last_error=self.audio_input.last_error,
        )
        while not self._stop_event.is_set():
            try:
                self._run_loop()
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._update_status(
                    last_error="collector loop failed: {error}".format(error=exc),
                )
                self.logger.exception("collector loop failed; retrying")
                try:
                    self._reset_audio_input()
                except Exception:
                    self.logger.exception("collector recovery failed")
                self._stop_event.wait(self._recovery_wait_seconds)
        self._update_status(
            is_running=False,
        )

    def _run_loop(self) -> None:
        self._update_status(
            last_error=self.audio_input.last_error,
        )
        self.logger.info(
            "collector initialized detection_window_seconds=%s capture_seconds=%s",
            self.detection_config.window_seconds,
            self.detection_config.capture_seconds,
        )
        detection_chunks = max(
            1,
            math.ceil(
                self.detection_config.window_seconds
                * self.audio_config.sample_rate
                / self.audio_config.chunk_size
            ),
        )
        capture_chunks = max(
            1,
            math.ceil(
                self.detection_config.capture_seconds
                * self.audio_config.sample_rate
                / self.audio_config.chunk_size
            ),
        )

        while not self._stop_event.is_set():
            detection_frames = self._collect_chunk_batch(detection_chunks)
            detection_audio = np.concatenate(detection_frames).astype(np.int16)
            rms = self._compute_rms(detection_audio)
            if self._calibration_requested.is_set() or not self._noise_gate.is_ready:
                self._calibration_requested.clear()
                gate_decision = self._noise_gate.calibrate(detection_audio)
                self._remember_detection_window(detection_audio, rms, gate_decision)
                self._mark_latest_detection_window("calibrated")
                self._update_status(
                    last_peak_rms=round(rms, 2),
                    last_peak_dbfs=round(gate_decision.current_dbfs, 2),
                    noise_floor_rms=round(gate_decision.noise_floor_rms, 2),
                    noise_floor_std_rms=round(gate_decision.noise_floor_std_rms, 2),
                    trigger_threshold_rms=round(gate_decision.trigger_threshold_rms, 2),
                    noise_floor_dbfs=round(gate_decision.noise_floor_dbfs, 2),
                    trigger_threshold_dbfs=round(gate_decision.trigger_threshold_dbfs, 2),
                    gate_ready=gate_decision.is_ready,
                    is_calibrating=False,
                    last_gate_open=False,
                    recent_detection_windows=self._get_detection_window_previews(),
                    last_error=self.audio_input.last_error,
                )
                self.logger.info(
                    "noise floor calibrated noise_floor_rms=%s noise_floor_std_rms=%s trigger_threshold_rms=%s",
                    round(gate_decision.noise_floor_rms, 2),
                    round(gate_decision.noise_floor_std_rms, 2),
                    round(gate_decision.trigger_threshold_rms, 2),
                )
                self.logger.debug(
                    "noise floor calibrated dbfs=%s window_rms=%s",
                    round(gate_decision.noise_floor_dbfs, 2),
                    round(rms, 2),
                )
                continue

            gate_decision = self._noise_gate.evaluate(rms)
            self._remember_detection_window(detection_audio, rms, gate_decision)
            self._update_status(
                last_peak_rms=round(rms, 2),
                last_peak_dbfs=round(gate_decision.current_dbfs, 2),
                noise_floor_rms=round(gate_decision.noise_floor_rms, 2),
                noise_floor_std_rms=round(gate_decision.noise_floor_std_rms, 2),
                trigger_threshold_rms=round(gate_decision.trigger_threshold_rms, 2),
                noise_floor_dbfs=round(gate_decision.noise_floor_dbfs, 2),
                trigger_threshold_dbfs=round(gate_decision.trigger_threshold_dbfs, 2),
                gate_ready=gate_decision.is_ready,
                is_calibrating=False,
                last_gate_open=gate_decision.should_classify,
                recent_detection_windows=self._get_detection_window_previews(),
                last_error=self.audio_input.last_error,
            )
            if not gate_decision.should_classify:
                self.logger.debug(
                    "detection window skipped by gate current_rms=%s trigger_threshold_rms=%s gate_ready=%s",
                    round(gate_decision.current_rms, 2),
                    round(gate_decision.trigger_threshold_rms, 2),
                    gate_decision.is_ready,
                )
                self.logger.debug(
                    "detection window skipped by gate current_dbfs=%s trigger_threshold_dbfs=%s",
                    round(gate_decision.current_dbfs, 2),
                    round(gate_decision.trigger_threshold_dbfs, 2),
                )
                continue
            classification_audio = self._normalize_for_classification(detection_audio)
            classification = self.classifier.classify_samples(
                classification_audio,
                self.audio_config.sample_rate,
            )
            if not self.classifier.should_retain(classification):
                self._mark_latest_detection_window("detected")
                self._update_status(
                    recent_detection_windows=self._get_detection_window_previews(),
                )
                self.logger.info(
                    "detection window discarded class=%s score=%s",
                    classification.label,
                    round(classification.score, 4),
                )
                continue
            self._mark_latest_detection_window("recorded")
            self._update_status(
                recent_detection_windows=self._get_detection_window_previews(),
            )
            capture_frames = []
            remaining_capture_chunks = capture_chunks
            while remaining_capture_chunks > 0 and not self._stop_event.is_set():
                batch_chunks = min(detection_chunks, remaining_capture_chunks)
                capture_batch = self._collect_chunk_batch(batch_chunks)
                capture_frames.extend(capture_batch)
                remaining_capture_chunks -= len(capture_batch)
                capture_audio = np.concatenate(capture_batch).astype(np.int16)
                capture_rms = self._compute_rms(capture_audio)
                capture_gate_decision = self._noise_gate.evaluate(capture_rms)
                self._remember_detection_window(
                    capture_audio,
                    capture_rms,
                    capture_gate_decision,
                )
                self._mark_latest_detection_window("recorded")
                self._update_status(
                    last_peak_rms=round(capture_rms, 2),
                    last_peak_dbfs=round(capture_gate_decision.current_dbfs, 2),
                    noise_floor_rms=round(capture_gate_decision.noise_floor_rms, 2),
                    noise_floor_std_rms=round(capture_gate_decision.noise_floor_std_rms, 2),
                    trigger_threshold_rms=round(capture_gate_decision.trigger_threshold_rms, 2),
                    noise_floor_dbfs=round(capture_gate_decision.noise_floor_dbfs, 2),
                    trigger_threshold_dbfs=round(capture_gate_decision.trigger_threshold_dbfs, 2),
                    gate_ready=capture_gate_decision.is_ready,
                    is_calibrating=False,
                    last_gate_open=True,
                    recent_detection_windows=self._get_detection_window_previews(),
                    last_error=self.audio_input.last_error,
                )
            recording_frames = [*detection_frames, *capture_frames]
            recording_audio = np.concatenate(recording_frames).astype(np.int16)
            peak_rms = self._compute_rms(recording_audio)
            recording_path = self._write_recording(recording_frames)
            event_time = datetime.now(timezone.utc).isoformat()
            self.event_store.append(
                NoiseEvent(
                    timestamp=event_time,
                    filename=recording_path.name,
                    peak_rms=round(peak_rms, 2),
                    duration_seconds=self._compute_duration_seconds(recording_audio),
                    classification=classification.label,
                    classification_score=round(classification.score, 4),
                    top_classes=classification.top_classes,
                )
            )
            self.logger.info(
                "event captured file=%s class=%s score=%s peak_rms=%s",
                recording_path.name,
                classification.label,
                round(classification.score, 4),
                round(peak_rms, 2),
            )

    def _write_recording(self, frames: list[np.ndarray]) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        suffix = uuid.uuid4().hex[:8]
        path = self.event_store.records_dir / f"{timestamp}-{suffix}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.audio_config.sample_rate)
            payload = np.concatenate(frames).astype(np.int16).tobytes()
            handle.writeframes(payload)
        return path

    def _create_audio_input(self) -> AudioInputLike:
        if self._audio_input_factory is None:
            raise RuntimeError("audio input cannot be recreated")
        return self._audio_input_factory()

    def _reset_audio_input(self) -> None:
        try:
            self.audio_input.close()
        except Exception:
            self.logger.exception("failed to close audio input during reset")

        if self._audio_input_factory is None:
            self.logger.warning("collector recovery unavailable for injected audio input")
            return

        self.audio_input = self._create_audio_input()
        self._update_status(
            selected_device_index=(
                self.audio_input.selected_device.index
                if self.audio_input.selected_device is not None
                else None
            ),
            selected_device_name=(
                self.audio_input.selected_device.name
                if self.audio_input.selected_device is not None
                else ""
            ),
            last_error=self.audio_input.last_error,
        )

    def _collect_chunk_batch(self, chunk_count: int) -> list[np.ndarray]:
        frames = []
        for _ in range(chunk_count):
            if self._stop_event.is_set():
                break
            frames.append(self.audio_input.read_chunk().copy())
        if not frames:
            raise RuntimeError("collector stopped before audio batch was captured")
        return frames

    @staticmethod
    def _compute_rms(samples: np.ndarray) -> float:
        normalized = samples.astype(np.float32)
        return float(np.sqrt(np.mean(normalized * normalized)))

    @classmethod
    def _normalize_for_classification(cls, samples: np.ndarray) -> np.ndarray:
        waveform = samples.astype(np.float32) / 32768.0
        if waveform.size == 0:
            return waveform

        rms = float(np.sqrt(np.mean(waveform * waveform)))
        peak = float(np.max(np.abs(waveform)))
        if rms <= 0.0 or peak <= 0.0:
            return waveform

        target_gain = cls._CLASSIFICATION_TARGET_RMS / rms
        peak_limited_gain = cls._CLASSIFICATION_MAX_PEAK / peak
        gain = max(1.0, min(target_gain, peak_limited_gain))
        return np.clip(
            waveform * gain,
            -cls._CLASSIFICATION_MAX_PEAK,
            cls._CLASSIFICATION_MAX_PEAK,
        ).astype(np.float32)

    def _compute_duration_seconds(self, samples: np.ndarray) -> float:
        return round(float(samples.shape[0]) / float(self.audio_config.sample_rate), 6)

    def _remember_detection_window(
        self,
        samples: np.ndarray,
        rms: float,
        gate_decision: _NoiseGateDecision,
    ) -> None:
        self._recent_detection_windows.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "window_duration_seconds": self.detection_config.window_seconds,
                "points": self._build_waveform_preview(samples),
                "rms": round(rms, 2),
                "dbfs": round(gate_decision.current_dbfs, 2),
                "noise_floor_rms": round(gate_decision.noise_floor_rms, 2),
                "noise_floor_std_rms": round(gate_decision.noise_floor_std_rms, 2),
                "trigger_threshold_rms": round(gate_decision.trigger_threshold_rms, 2),
                "noise_floor_dbfs": round(gate_decision.noise_floor_dbfs, 2),
                "trigger_threshold_dbfs": round(gate_decision.trigger_threshold_dbfs, 2),
                "gate_open": gate_decision.should_classify,
                "outcome": "detected" if gate_decision.should_classify else "ignored",
            }
        )

    def _mark_latest_detection_window(self, outcome: str) -> None:
        if not self._recent_detection_windows:
            return
        self._recent_detection_windows[-1]["outcome"] = outcome

    @staticmethod
    def _build_waveform_preview(
        samples: np.ndarray,
        *,
        point_count: int = 96,
    ) -> list[float]:
        if samples.size == 0:
            return []

        normalized = np.clip(samples.astype(np.float32) / 32768.0, -1.0, 1.0)
        bucket_count = min(point_count, normalized.size)
        buckets = np.array_split(normalized, bucket_count)
        # Use peak magnitude per bucket so short transients remain visible in the UI
        # after compressing a full detection window into a compact sparkline.
        preview = [
            float(bucket[np.argmax(np.abs(bucket))])
            for bucket in buckets
            if bucket.size > 0
        ]
        return [round(item, 4) for item in preview]

    def _get_detection_window_previews(self) -> list[dict[str, object]]:
        return [dict(item) for item in self._recent_detection_windows]

    def get_status(self) -> CollectorRuntimeStatus:
        with self._status_lock:
            return CollectorRuntimeStatus(
                selected_device_index=self.status.selected_device_index,
                selected_device_name=self.status.selected_device_name,
                is_running=self.status.is_running,
                last_peak_rms=self.status.last_peak_rms,
                last_peak_dbfs=self.status.last_peak_dbfs,
                detection_window_seconds=self.status.detection_window_seconds,
                capture_seconds=self.status.capture_seconds,
                noise_floor_rms=self.status.noise_floor_rms,
                noise_floor_std_rms=self.status.noise_floor_std_rms,
                trigger_threshold_rms=self.status.trigger_threshold_rms,
                noise_floor_dbfs=self.status.noise_floor_dbfs,
                trigger_threshold_dbfs=self.status.trigger_threshold_dbfs,
                gate_ready=self.status.gate_ready,
                is_calibrating=self.status.is_calibrating,
                last_gate_open=self.status.last_gate_open,
                recent_detection_windows=[dict(item) for item in self.status.recent_detection_windows],
                last_error=self.status.last_error,
            )

    def _update_status(self, **kwargs: object) -> None:
        with self._status_lock:
            for key, value in kwargs.items():
                setattr(self.status, key, value)
