from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from audio_detect.collector import CollectorRuntimeStatus
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

    def get_status(self) -> CollectorRuntimeStatus:
        return CollectorRuntimeStatus(
            is_running=True,
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


class StorageAndWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        base = Path(self.tempdir.name)
        self.records_dir = base / "records"
        self.database_path = base / "events.db"
        self.event_store = EventStore(self.records_dir, self.database_path)
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
        events = response.get_json()["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["record_url"], "/records/sample.wav")
        self.assertEqual(events[0]["top_classes"], ["Engine:1.000"])

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
