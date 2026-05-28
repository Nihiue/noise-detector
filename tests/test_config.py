from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from audio_detect.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_custom_config_resolves_relative_paths_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config_dir = base / "conf"
            model_dir = base / "models"
            config_dir.mkdir()
            model_dir.mkdir()
            (model_dir / "yamnet.tflite").write_bytes(b"model")
            (model_dir / "yamnet_class_map.csv").write_text(
                "index,mid,display_name\n0,/m/test,Test\n",
                encoding="utf-8",
            )
            config_path = config_dir / "settings.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    app:
                      host: 127.0.0.1
                      port: 9000
                    audio:
                      device_name: null
                      sample_rate: 16000
                      chunk_size: 1024
                      format: int16
                    detection:
                      window_seconds: 4.0
                      capture_seconds: 12.0
                      threshold_stddev_multiplier: 1.5
                      threshold_rms_offset: 120.0
                    storage:
                      records_dir: ../runtime/data/records
                      database_path: ../runtime/data/audio.db
                    classification:
                      enabled: true
                      label: Unknown
                      score_threshold: 0.2
                      model_path: ../models/yamnet.tflite
                      class_map_path: ../models/yamnet_class_map.csv
                      retained_labels:
                        - Engine
                        - Motorcycle
                    retention:
                      max_record_days: 7
                      max_records: 100
                    logging:
                      level: INFO
                      file_path: ../runtime/data/logs/app.log
                    """
                ).strip(),
                encoding="utf-8",
            )

            settings = load_settings(config_path)

            self.assertEqual(
                settings.storage.database_path,
                (base / "runtime" / "data" / "audio.db").resolve(),
            )
            self.assertEqual(
                settings.classification.model_path,
                (base / "models" / "yamnet.tflite").resolve(),
            )
            self.assertEqual(
                settings.logging.file_path,
                (base / "runtime" / "data" / "logs" / "app.log").resolve(),
            )
            self.assertEqual(
                settings.classification.retained_labels,
                ["Engine", "Motorcycle"],
            )
            self.assertEqual(settings.detection.window_seconds, 4.0)
            self.assertEqual(settings.detection.capture_seconds, 12.0)
            self.assertEqual(settings.detection.threshold_stddev_multiplier, 1.5)
            self.assertEqual(settings.detection.threshold_rms_offset, 120.0)
            self.assertFalse(hasattr(settings.audio, "channels"))

    def test_detection_threshold_stddev_multiplier_defaults_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config_path = base / "settings.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    app:
                      host: 127.0.0.1
                      port: 9000
                    audio:
                      device_name: null
                      sample_rate: 16000
                      chunk_size: 1024
                      format: int16
                    detection:
                      window_seconds: 4.0
                      capture_seconds: 12.0
                    storage:
                      records_dir: data/records
                      database_path: data/audio.db
                    classification:
                      enabled: false
                      label: Unknown
                      model_path: model.tflite
                      class_map_path: class_map.csv
                      retained_labels: []
                    retention:
                      max_record_days: 7
                      max_records: 100
                    logging:
                      level: INFO
                      file_path: logs/app.log
                    """
                ).strip(),
                encoding="utf-8",
            )

            settings = load_settings(config_path)

            self.assertEqual(settings.detection.threshold_stddev_multiplier, 1.0)
            self.assertEqual(settings.detection.threshold_rms_offset, 100.0)


if __name__ == "__main__":
    unittest.main()
