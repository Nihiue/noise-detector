from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_detect.classifier import ClassificationResult
from audio_detect.audio_devices import AudioDeviceInfo
from audio_detect.collector import CollectorService, _NoiseFloorGate
from audio_detect.config import AudioConfig, ClassificationConfig, DetectionConfig
from audio_detect.main import main, serve_app
from audio_detect.storage import EventStore


class StubClassifier:
    def __init__(
        self,
        label: str = "Engine",
        *,
        top_classes: list[str] | None = None,
        results: list[ClassificationResult] | None = None,
    ) -> None:
        self.label = label
        self.top_classes = top_classes or [f"{self.label}:1.000"]
        self.results = list(results or [])
        self.calls = 0

    def classify(self, recording_path: Path) -> ClassificationResult:
        return self.classify_samples(np.array([], dtype=np.float32), 16000)

    def classify_samples(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> ClassificationResult:
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return ClassificationResult(
            label=self.label,
            score=1.0,
            top_classes=self.top_classes,
        )

    def should_retain(self, classification: ClassificationResult) -> bool:
        retained = {
            "Vehicle",
            "Motor vehicle (road)",
            "Car",
            "Car passing by",
            "Vehicle horn, car horn, honking",
            "Truck",
            "Bus",
            "Engine",
            "Engine starting",
            "Engine knocking",
            "Light engine (high frequency)",
            "Medium engine (mid frequency)",
            "Heavy engine (low frequency)",
            "Motorcycle",
            "Traffic noise, roadway noise",
        }
        for item in classification.top_classes:
            label = item.split(":", 1)[0].strip()
            if label in retained:
                return True
        return classification.label in retained


class StubAudioInput:
    def __init__(self) -> None:
        self.selected_device = AudioDeviceInfo(
            index=0,
            name="Test input",
            max_input_channels=1,
            default_samplerate=16000.0,
        )
        self.last_error = ""
        self._chunk = np.zeros(1024, dtype=np.int16)

    def read_chunk(self) -> np.ndarray:
        return self._chunk.copy()

    def close(self) -> None:
        return None


class CollectorAndMainTests(unittest.TestCase):
    def _make_service(
        self,
        base: Path,
        *,
        classifier: StubClassifier | None = None,
        audio_input: StubAudioInput | None = None,
    ) -> CollectorService:
        return CollectorService(
            audio_config=AudioConfig(
                device_name=None,
                sample_rate=16000,
                chunk_size=1024,
                sample_format="int16",
            ),
            detection_config=DetectionConfig(
                window_seconds=0.1,
                capture_seconds=0.1,
                threshold_stddev_multiplier=1.0,
            ),
            event_store=EventStore(
                records_dir=base / "records",
                database_path=base / "events.db",
            ),
            classifier=classifier or StubClassifier(),
            audio_input=audio_input or StubAudioInput(),
        )

    def test_service_uses_injected_audio_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(Path(tmpdir))

            self.assertEqual(service.get_status().selected_device_name, "Test input")

    def test_write_recording_uses_unique_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(Path(tmpdir))
            frames = [np.zeros(64, dtype=np.int16)]

            first = service._write_recording(frames)
            second = service._write_recording(frames)

            self.assertNotEqual(first.name, second.name)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_collect_chunk_batch_reads_requested_chunk_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(Path(tmpdir))
            reads = []

            def read_chunk() -> np.ndarray:
                reads.append("chunk")
                return np.zeros(1024, dtype=np.int16)

            service.audio_input.read_chunk = read_chunk  # type: ignore[method-assign]
            frames = service._collect_chunk_batch(4)

            self.assertEqual(len(frames), 4)
            self.assertEqual(len(reads), 4)

    def test_run_retries_after_loop_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(Path(tmpdir))
            service._recovery_wait_seconds = 0.0
            calls = []

            def failing_then_stop() -> None:
                calls.append("call")
                if len(calls) == 1:
                    raise RuntimeError("boom")
                service._stop_event.set()

            service._run_loop = failing_then_stop  # type: ignore[method-assign]
            service._run()

            self.assertEqual(len(calls), 2)
            self.assertIn("collector loop failed", service.get_status().last_error)
            self.assertFalse(service.get_status().is_running)

    def test_run_recreates_audio_input_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            created = []

            class RecoverableAudioInput(StubAudioInput):
                def __init__(self, fail_first: bool) -> None:
                    super().__init__()
                    self.fail_first = fail_first
                    self.closed = False
                    if not fail_first:
                        self._chunk = np.full(1024, 4000, dtype=np.int16)

                def read_chunk(self) -> np.ndarray:
                    if self.fail_first:
                        self.fail_first = False
                        raise RuntimeError("stream dropped")
                    return self._chunk.copy()

                def close(self) -> None:
                    self.closed = True

            def factory() -> RecoverableAudioInput:
                item = RecoverableAudioInput(fail_first=not created)
                created.append(item)
                return item

            service = CollectorService(
                audio_config=AudioConfig(
                    device_name=None,
                    sample_rate=16000,
                    chunk_size=1024,
                    sample_format="int16",
                ),
                detection_config=DetectionConfig(
                    window_seconds=0.1,
                    capture_seconds=0.1,
                    threshold_stddev_multiplier=1.0,
                ),
                event_store=EventStore(
                    records_dir=base / "records",
                    database_path=base / "events.db",
                ),
                classifier=StubClassifier(),
                audio_input=factory(),
            )
            service._audio_input_factory = factory
            service._recovery_wait_seconds = 0.0
            service._noise_gate.calibrate(np.full(1024, 120, dtype=np.int16))
            def append_and_stop(event) -> None:
                service._stop_event.set()

            service.event_store.append = append_and_stop  # type: ignore[method-assign]

            service._run()

            self.assertGreaterEqual(len(created), 2)
            self.assertTrue(created[0].closed)

    def test_non_target_classification_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(
                Path(tmpdir),
                classifier=StubClassifier(label="Speech"),
            )
            service.audio_input.read_chunk = lambda: np.full(1024, 2000, dtype=np.int16)  # type: ignore[method-assign]
            service._stop_event.set()

            service._run_loop()

            self.assertEqual(service.event_store.query_events(), [])
            self.assertEqual(list(service.event_store.records_dir.glob("*.wav")), [])

    def test_retained_classification_captures_full_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(Path(tmpdir))
            service._noise_gate.calibrate(np.full(1024, 120, dtype=np.int16))
            detect_chunk = np.full(1024, 500, dtype=np.int16)
            capture_chunk = np.full(1024, 1500, dtype=np.int16)
            remaining = [detect_chunk, capture_chunk, capture_chunk]

            def read_chunk() -> np.ndarray:
                if remaining:
                    return remaining.pop(0)
                return capture_chunk

            service.audio_input.read_chunk = read_chunk  # type: ignore[method-assign]
            original_append = service.event_store.append

            def append_and_stop(event) -> None:
                original_append(event)
                service._stop_event.set()

            service.event_store.append = append_and_stop  # type: ignore[method-assign]
            service._run_loop()

            events = service.event_store.query_events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].classification, "Engine")
            self.assertEqual(events[0].duration_seconds, 0.256)
            self.assertEqual(
                service.get_status().recent_detection_windows[-1]["outcome"],
                "recorded",
            )
            self.assertGreaterEqual(
                sum(
                    1
                    for item in service.get_status().recent_detection_windows
                    if item["outcome"] == "recorded"
                ),
                2,
            )

    def test_event_classification_uses_trigger_window_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            classifier = StubClassifier(
                results=[
                    ClassificationResult(
                        label="Engine",
                        score=0.9,
                        top_classes=["Engine:0.900", "Noise:0.200"],
                    ),
                    ClassificationResult(
                        label="Speech",
                        score=0.8,
                        top_classes=["Speech:0.800", "Engine:0.400"],
                    ),
                ]
            )
            service = self._make_service(Path(tmpdir), classifier=classifier)
            service._noise_gate.calibrate(np.full(1024, 120, dtype=np.int16))
            detect_chunk = np.full(1024, 500, dtype=np.int16)
            capture_chunk = np.full(1024, 3000, dtype=np.int16)
            remaining = [detect_chunk, capture_chunk, capture_chunk]

            def read_chunk() -> np.ndarray:
                if remaining:
                    return remaining.pop(0)
                return capture_chunk

            service.audio_input.read_chunk = read_chunk  # type: ignore[method-assign]
            original_append = service.event_store.append

            def append_and_stop(event) -> None:
                original_append(event)
                service._stop_event.set()

            service.event_store.append = append_and_stop  # type: ignore[method-assign]
            service._run_loop()

            events = service.event_store.query_events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].classification, "Engine")
            self.assertEqual(events[0].classification_score, 0.9)
            self.assertEqual(events[0].top_classes, ["Engine:0.900", "Noise:0.200"])
            self.assertEqual(classifier.calls, 1)

    def test_top3_match_is_retained_even_when_top1_is_not_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(
                Path(tmpdir),
                classifier=StubClassifier(
                    label="Speech",
                    top_classes=[
                        "Speech:0.900",
                        "Engine:0.700",
                        "Noise:0.200",
                    ],
                ),
            )
            service._noise_gate.calibrate(np.full(1024, 120, dtype=np.int16))
            detect_chunk = np.full(1024, 500, dtype=np.int16)
            capture_chunk = np.full(1024, 1500, dtype=np.int16)
            remaining = [detect_chunk, capture_chunk, capture_chunk]

            def read_chunk() -> np.ndarray:
                if remaining:
                    return remaining.pop(0)
                return capture_chunk

            service.audio_input.read_chunk = read_chunk  # type: ignore[method-assign]
            original_append = service.event_store.append

            def append_and_stop(event) -> None:
                original_append(event)
                service._stop_event.set()

            service.event_store.append = append_and_stop  # type: ignore[method-assign]
            service._run_loop()

            events = service.event_store.query_events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].classification, "Speech")
            self.assertEqual(
                events[0].top_classes,
                ["Speech:0.900", "Engine:0.700", "Noise:0.200"],
            )

    def test_noise_gate_requires_calibration_before_running_classifier(self) -> None:
        gate = _NoiseFloorGate(detection_window_seconds=0.1)
        decision = gate.evaluate(200.0)

        self.assertFalse(decision.should_classify)
        self.assertFalse(decision.is_ready)

    def test_noise_gate_opens_for_signal_above_background(self) -> None:
        gate = _NoiseFloorGate(detection_window_seconds=0.1)
        calibration = gate.calibrate(np.full(1024, 150, dtype=np.int16))

        decision = gate.evaluate(2500.0)

        self.assertTrue(calibration.is_ready)
        self.assertTrue(decision.should_classify)
        self.assertGreater(decision.current_rms, decision.trigger_threshold_rms)

    def test_noise_gate_threshold_uses_calibration_rms_mean_and_stddev(self) -> None:
        gate = _NoiseFloorGate(detection_window_seconds=0.1)
        samples = np.concatenate(
            [
                np.full(500, 100, dtype=np.int16),
                np.full(500, 200, dtype=np.int16),
            ]
        )

        calibration = gate.calibrate(samples)
        below_threshold = gate.evaluate(140.0)
        above_threshold = gate.evaluate(180.0)

        self.assertAlmostEqual(calibration.noise_floor_rms, 150.0)
        self.assertAlmostEqual(calibration.noise_floor_std_rms, 50.0)
        self.assertAlmostEqual(calibration.trigger_threshold_rms, 200.0)
        self.assertFalse(below_threshold.should_classify)
        self.assertFalse(above_threshold.should_classify)
        self.assertTrue(gate.evaluate(220.0).should_classify)

    def test_noise_gate_uses_configured_stddev_multiplier(self) -> None:
        gate = _NoiseFloorGate(
            detection_window_seconds=0.1,
            threshold_stddev_multiplier=2.0,
        )
        samples = np.concatenate(
            [
                np.full(500, 100, dtype=np.int16),
                np.full(500, 200, dtype=np.int16),
            ]
        )

        calibration = gate.calibrate(samples)

        self.assertAlmostEqual(calibration.noise_floor_rms, 150.0)
        self.assertAlmostEqual(calibration.noise_floor_std_rms, 50.0)
        self.assertAlmostEqual(calibration.trigger_threshold_rms, 250.0)
        self.assertFalse(gate.evaluate(220.0).should_classify)
        self.assertTrue(gate.evaluate(260.0).should_classify)

    def test_noise_gate_keeps_floor_until_next_calibration(self) -> None:
        gate = _NoiseFloorGate(detection_window_seconds=0.1)
        initial_floor = gate.calibrate(np.full(1024, 120, dtype=np.int16)).noise_floor_rms

        for _ in range(50):
            gate.evaluate(6000.0)
        later_floor = gate.evaluate(120.0).noise_floor_rms

        self.assertLess(abs(later_floor - initial_floor), 1.0)

    def test_detection_window_preview_includes_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(Path(tmpdir))
            decision = service._noise_gate.calibrate(np.full(1024, 120, dtype=np.int16))

            service._remember_detection_window(
                np.full(1024, 120, dtype=np.int16),
                120.0,
                decision,
            )

            timestamp = service._get_detection_window_previews()[0]["timestamp"]
            self.assertIsNotNone(datetime.fromisoformat(str(timestamp)))
            self.assertEqual(
                service._get_detection_window_previews()[0]["window_duration_seconds"],
                0.1,
            )

    def test_run_loop_skips_classification_when_gate_stays_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            classifier = StubClassifier()
            service = self._make_service(Path(tmpdir), classifier=classifier)
            service.audio_input.read_chunk = lambda: np.full(1024, 120, dtype=np.int16)  # type: ignore[method-assign]
            service._stop_event.set()

            service._run_loop()

            self.assertEqual(classifier.calls, 0)
            self.assertEqual(service.event_store.query_events(), [])

    def test_stop_closes_audio_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(Path(tmpdir))
            close_calls = []

            def close() -> None:
                close_calls.append("closed")

            service.audio_input.close = close  # type: ignore[method-assign]
            service.stop()

            self.assertEqual(close_calls, ["closed"])

    def test_main_fails_fast_when_classifier_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            settings = type(
                "SettingsStub",
                (),
                {
                    "logging": type("LoggingStub", (), {"level": "INFO", "file_path": base / "app.log"})(),
                    "storage": type(
                        "StorageStub",
                        (),
                        {"records_dir": base / "records", "database_path": base / "events.db"},
                    )(),
                    "retention": type(
                        "RetentionStub",
                        (),
                        {"max_record_days": 7, "max_records": 100},
                    )(),
                    "classification": ClassificationConfig(
                        enabled=True,
                        label="Unknown",
                        score_threshold=0.2,
                        model_path=base / "missing.tflite",
                        class_map_path=base / "missing.csv",
                        retained_labels=["Engine"],
                    ),
                    "audio": AudioConfig(
                        device_name=None,
                        sample_rate=16000,
                        chunk_size=1024,
                        sample_format="int16",
                    ),
                    "detection": DetectionConfig(
                        window_seconds=0.1,
                        capture_seconds=0.1,
                        threshold_stddev_multiplier=1.0,
                    ),
                    "app": type("AppStub", (), {"host": "127.0.0.1", "port": 8000})(),
                },
            )()

            with patch("audio_detect.main.load_settings", return_value=settings), patch(
                "audio_detect.main.configure_logging"
            ), patch("audio_detect.main.CollectorService"), patch(
                "audio_detect.main.create_app"
            ), patch(
                "argparse.ArgumentParser.parse_args",
                return_value=type(
                    "Args",
                    (),
                    {"config": None, "list_devices": False},
                )(),
            ):
                with self.assertRaises(RuntimeError):
                    main()

    def test_serve_app_stops_collector_on_keyboard_interrupt(self) -> None:
        class DummyServer:
            def __init__(self) -> None:
                self.closed = False
                self.shutdown_called = False

            def serve_forever(self) -> None:
                return None

            def shutdown(self) -> None:
                self.shutdown_called = True

            def server_close(self) -> None:
                self.closed = True

        class DummyCollector:
            def __init__(self) -> None:
                self.stop_calls = 0

            def stop(self) -> None:
                self.stop_calls += 1

        server = DummyServer()
        collector = DummyCollector()

        serve_app(
            object(),
            host="127.0.0.1",
            port=8000,
            collector=collector,  # type: ignore[arg-type]
            server_factory=lambda *args, **kwargs: server,  # type: ignore[arg-type]
        )

        self.assertEqual(collector.stop_calls, 1)
        self.assertTrue(server.shutdown_called)
        self.assertTrue(server.closed)


if __name__ == "__main__":
    unittest.main()
