from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from audio_detect.collector import CollectorRuntimeStatus
from audio_detect.audio_devices import AudioDeviceInfo
from audio_detect.config import (
    AppConfig,
    AudioConfig,
    ClassificationConfig,
    DetectionConfig,
    LoggingConfig,
    RetentionConfig,
    Settings,
    StorageConfig,
)
from audio_detect.storage import EventStore, InvalidTimeFilterError, NoiseEvent
from audio_detect.web import create_app


class DummyCollector:
    def __init__(self) -> None:
        self.calibration_requests = 0
        self.switch_requests = []

    def get_status(self) -> CollectorRuntimeStatus:
        return CollectorRuntimeStatus(
            selected_device_index=1,
            selected_device_name="Mic A",
            is_running=True,
            is_switching_device=False,
            detection_window_seconds=3.0,
            noise_floor_rms=100.0,
            noise_floor_std_rms=12.0,
            trigger_threshold_rms=112.0,
            trigger_threshold_dbfs=-42.0,
            noise_floor_dbfs=-51.0,
            recent_detection_windows=[
                {
                    "timestamp": "2026-05-18T04:00:00+00:00",
                    "window_duration_seconds": 3.0,
                    "points": [0.0, 0.4, -0.2],
                    "rms": 100.0,
                    "dbfs": -38.0,
                    "noise_floor_rms": 100.0,
                    "noise_floor_std_rms": 12.0,
                    "trigger_threshold_rms": 112.0,
                    "noise_floor_dbfs": -51.0,
                    "trigger_threshold_dbfs": -42.0,
                    "gate_open": True,
                    "outcome": "recorded",
                }
            ],
        )

    def request_calibration(self) -> CollectorRuntimeStatus:
        self.calibration_requests += 1
        return CollectorRuntimeStatus(
            selected_device_index=1,
            selected_device_name="Mic A",
            is_running=True,
            detection_window_seconds=3.0,
            noise_floor_rms=100.0,
            noise_floor_std_rms=12.0,
            trigger_threshold_rms=112.0,
            trigger_threshold_dbfs=-42.0,
            noise_floor_dbfs=-42.0,
            gate_ready=True,
            is_calibrating=True,
        )

    @property
    def configured_device_name(self) -> str:
        return "Built-in Mic"

    def can_switch_input_device(self) -> bool:
        return True

    def list_input_devices(self) -> list[AudioDeviceInfo]:
        return [
            AudioDeviceInfo(index=1, name="Mic A", max_input_channels=1, default_samplerate=16000.0),
            AudioDeviceInfo(index=2, name="Mic B", max_input_channels=2, default_samplerate=48000.0),
        ]

    def switch_input_device(self, device_index: int) -> CollectorRuntimeStatus:
        self.switch_requests.append(device_index)
        if device_index == 9:
            raise RuntimeError("failed to switch input device: missing device")
        return CollectorRuntimeStatus(
            selected_device_index=device_index,
            selected_device_name="Mic B" if device_index == 2 else "Mic A",
            is_running=True,
            detection_window_seconds=3.0,
            gate_ready=False,
            is_switching_device=False,
        )


class StorageAndWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        base = Path(self.tempdir.name)
        self.records_dir = base / "records"
        self.database_path = base / "events.db"
        self.event_store = EventStore(
            self.records_dir,
            self.database_path,
            max_record_days=365,
            max_records=500,
        )
        self.settings = Settings(
            app=AppConfig(host="127.0.0.1", port=8000),
            audio=AudioConfig(
                device_name=None,
                sample_rate=16000,
                chunk_size=1024,
                sample_format="int16",
            ),
            detection=DetectionConfig(
                window_seconds=4.0,
                capture_seconds=12.0,
                threshold_stddev_multiplier=1.0,
                threshold_rms_offset=100.0,
            ),
            storage=StorageConfig(
                records_dir=self.records_dir,
                database_path=self.database_path,
            ),
            classification=ClassificationConfig(
                enabled=False,
                label="Unknown",
                score_threshold=0.2,
                model_path=base / "model.tflite",
                class_map_path=base / "class_map.csv",
                retained_labels=["Engine", "Motorcycle"],
            ),
            retention=RetentionConfig(max_record_days=7, max_records=100),
            logging=LoggingConfig(level="INFO", file_path=base / "logs" / "app.log"),
        )
        self.collector = DummyCollector()
        self.app = create_app(
            self.event_store,
            self.collector,
        )
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.event_store.close()
        self.tempdir.cleanup()

    def test_query_events_rejects_invalid_start_time(self) -> None:
        with self.assertRaises(InvalidTimeFilterError):
            self.event_store.query_events(start_at="2026-05-12")

    def test_query_events_rejects_inverted_time_range(self) -> None:
        with self.assertRaises(InvalidTimeFilterError):
            self.event_store.query_events(
                start_at="20260513",
                end_at="20260512",
            )

    def test_api_events_returns_400_for_invalid_time_filter(self) -> None:
        response = self.client.get("/api/events?start_at=2026-05-12")

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"start_at must use YYYYMMDD format", response.data)

    def test_index_returns_react_shell(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("\u566a\u97f3\u76d1\u6d4b\u9762\u677f".encode(), response.data)
        self.assertIn(b"dayjs.min.js", response.data)
        self.assertIn(b"echarts.min.js", response.data)
        self.assertIn(b"https://unpkg.byted-static.com/react/18.1.0/umd/react.production.min.js", response.data)
        self.assertIn(b"/static/app.js", response.data)
        self.assertIn(b"/static/styles.css", response.data)

    def test_frontend_static_files_include_tabs_and_fixed_grids(self) -> None:
        app_response = self.client.get("/static/app.js")
        css_response = self.client.get("/static/styles.css")

        self.assertEqual(app_response.status_code, 200)
        self.assertEqual(css_response.status_code, 200)
        self.assertIn(b"tab-button", app_response.data)
        self.assertIn(b'localStorage.getItem("audio-detect:auto-refresh") !== "0"', app_response.data)
        self.assertIn(b"setInterval", app_response.data)
        self.assertIn(b"/api/calibrate", app_response.data)
        self.assertIn("\u6821\u51c6\u9608\u503c".encode(), app_response.data)
        self.assertIn("\u5df2\u68c0\u6d4b".encode(), app_response.data)
        self.assertIn("\u5df2\u8bb0\u5f55".encode(), app_response.data)
        self.assertIn(b"combined-waveform", app_response.data)
        self.assertIn(b"echart-timeline", app_response.data)
        self.assertIn(b"animationDurationUpdate", app_response.data)
        self.assertIn(b"grid-template-columns: repeat(4", css_response.data)
        self.assertIn(b".echart-timeline", css_response.data)

    def test_api_status_returns_detection_window_monitor_data(self) -> None:
        response = self.client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["detection_window_seconds"], 3.0)
        self.assertEqual(payload["noise_floor_rms"], 100.0)
        self.assertEqual(payload["noise_floor_std_rms"], 12.0)
        self.assertEqual(payload["trigger_threshold_rms"], 112.0)
        self.assertEqual(payload["trigger_threshold_dbfs"], -42.0)
        self.assertFalse(payload["is_calibrating"])
        self.assertEqual(
            payload["recent_detection_windows"][0]["timestamp"],
            "2026-05-18T04:00:00+00:00",
        )
        self.assertEqual(payload["recent_detection_windows"][0]["window_duration_seconds"], 3.0)
        self.assertEqual(payload["recent_detection_windows"][0]["points"], [0.0, 0.4, -0.2])
        self.assertEqual(payload["recent_detection_windows"][0]["outcome"], "recorded")

    def test_api_calibrate_requests_threshold_calibration(self) -> None:
        response = self.client.post("/api/calibrate")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.collector.calibration_requests, 1)
        self.assertTrue(response.get_json()["is_calibrating"])

    def test_api_audio_devices_returns_available_inputs(self) -> None:
        response = self.client.get("/api/audio-devices")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["selected_device_index"], 1)
        self.assertEqual(payload["configured_device_name"], "Built-in Mic")
        self.assertTrue(payload["can_switch"])
        self.assertEqual(len(payload["devices"]), 2)
        self.assertEqual(payload["devices"][1]["name"], "Mic B")

    def test_api_audio_device_switches_input(self) -> None:
        response = self.client.post(
            "/api/audio-device",
            json={"device_index": 2},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.collector.switch_requests, [2])
        payload = response.get_json()
        self.assertEqual(payload["selected_device_index"], 2)
        self.assertEqual(payload["selected_device_name"], "Mic B")
        self.assertFalse(payload["gate_ready"])

    def test_api_audio_device_rejects_invalid_payload(self) -> None:
        response = self.client.post(
            "/api/audio-device",
            json={"device_index": -1},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"device_index must be zero or a positive integer", response.data)

    def test_api_audio_device_returns_conflict_on_switch_failure(self) -> None:
        response = self.client.post(
            "/api/audio-device",
            json={"device_index": 9},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn(b"failed to switch input device", response.data)

    def test_api_classifications_returns_distinct_values(self) -> None:
        self.event_store.append(
            NoiseEvent(
                timestamp="2026-05-12T08:00:00+00:00",
                filename="sample.wav",
                peak_rms=1.0,
                duration_seconds=12.0,
                classification="Engine",
                classification_score=1.0,
                top_classes=[],
            )
        )

        response = self.client.get("/api/classifications")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["classifications"], ["Engine"])

    def test_api_events_returns_event_record_urls(self) -> None:
        self.event_store.append(
            NoiseEvent(
                timestamp="2026-05-12T08:00:00+00:00",
                filename="sample.wav",
                peak_rms=1.0,
                duration_seconds=12.0,
                classification="Engine",
                classification_score=1.0,
                top_classes=["Engine:1.000"],
            )
        )

        response = self.client.get("/api/events?classification=Engine")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        events = payload["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["record_url"], "/records/sample.wav")
        self.assertEqual(events[0]["top_classes"], ["Engine:1.000"])
        self.assertEqual(payload["pagination"]["page"], 1)
        self.assertEqual(payload["pagination"]["page_size"], 20)
        self.assertEqual(payload["pagination"]["total"], 1)
        self.assertEqual(payload["pagination"]["total_pages"], 1)

    def test_api_events_supports_pagination(self) -> None:
        for index in range(3):
            self.event_store.append(
                NoiseEvent(
                    timestamp="2026-05-12T08:00:0{index}+00:00".format(index=index),
                    filename="sample-{index}.wav".format(index=index),
                    peak_rms=1.0,
                    duration_seconds=12.0,
                    classification="Engine",
                    classification_score=1.0,
                    top_classes=[],
                )
            )

        response = self.client.get("/api/events?page=2&page_size=1&classification=Engine")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["pagination"]["page"], 2)
        self.assertEqual(payload["pagination"]["page_size"], 1)
        self.assertEqual(payload["pagination"]["total"], 3)
        self.assertEqual(payload["pagination"]["total_pages"], 3)
        self.assertEqual(len(payload["events"]), 1)
        self.assertEqual(payload["events"][0]["filename"], "sample-1.wav")

    def test_api_events_rejects_invalid_pagination(self) -> None:
        response = self.client.get("/api/events?page=0")

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"page must be a positive integer", response.data)

    def test_export_returns_400_for_invalid_time_filter(self) -> None:
        response = self.client.get("/export?end_at=2026-05-12")

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"end_at must use YYYYMMDD format", response.data)

    def test_query_events_accepts_yyyymmdd_range(self) -> None:
        self.event_store.append(
            NoiseEvent(
                timestamp="2026-05-12T08:00:00+00:00",
                filename="sample.wav",
                peak_rms=1.0,
                duration_seconds=12.0,
                classification="Engine",
                classification_score=1.0,
                top_classes=[],
            )
        )
        events = self.event_store.query_events(
            start_at="20260512",
            end_at="20260512",
        )

        self.assertEqual(len(events), 1)

    def test_query_events_page_returns_total_and_current_page(self) -> None:
        for index in range(4):
            self.event_store.append(
                NoiseEvent(
                    timestamp="2026-05-12T08:00:0{index}+00:00".format(index=index),
                    filename="sample-{index}.wav".format(index=index),
                    peak_rms=1.0,
                    duration_seconds=12.0,
                    classification="Engine",
                    classification_score=1.0,
                    top_classes=[],
                )
            )

        events, total = self.event_store.query_events_page(
            classification="Engine",
            page=2,
            page_size=2,
        )

        self.assertEqual(total, 4)
        self.assertEqual([event.filename for event in events], ["sample-1.wav", "sample-0.wav"])

    def test_append_applies_retention_limits_without_cleanup_call(self) -> None:
        self.event_store.close()
        self.event_store = EventStore(
            self.records_dir,
            self.database_path,
            max_record_days=1,
            max_records=1,
        )

        expired_name = "expired.wav"
        newest_name = "fresh.wav"
        now = datetime.now(timezone.utc)
        expired_at = (now - timedelta(days=3)).isoformat()
        newest_at = now.isoformat()
        (self.records_dir / expired_name).write_bytes(b"expired")
        (self.records_dir / newest_name).write_bytes(b"fresh")

        self.event_store.append(
            NoiseEvent(
                timestamp=expired_at,
                filename=expired_name,
                peak_rms=1.0,
                duration_seconds=12.0,
                classification="Engine",
                classification_score=1.0,
                top_classes=[],
            )
        )
        self.event_store.append(
            NoiseEvent(
                timestamp=newest_at,
                filename=newest_name,
                peak_rms=1.0,
                duration_seconds=12.0,
                classification="Engine",
                classification_score=1.0,
                top_classes=[],
            )
        )

        events = self.event_store.query_events()
        self.assertEqual([event.filename for event in events], [newest_name])
        self.assertFalse((self.records_dir / expired_name).exists())
        self.assertTrue((self.records_dir / newest_name).exists())


if __name__ == "__main__":
    unittest.main()
