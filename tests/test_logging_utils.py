from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from audio_detect.logging_utils import BufferedFileHandler


class LoggingUtilsTests(unittest.TestCase):
    def test_buffered_file_handler_flushes_after_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffered.log"
            handler = BufferedFileHandler(path, flush_threshold_bytes=32)
            handler.setFormatter(logging.Formatter("%(message)s"))

            logger = logging.getLogger("test.buffered.threshold")
            logger.handlers = []
            logger.propagate = False
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            logger.info("1234567890")
            self.assertFalse(path.exists())

            logger.info("abcdefghij")
            handler.flush()

            self.assertTrue(path.exists())
            self.assertIn("1234567890", path.read_text(encoding="utf-8"))
            self.assertIn("abcdefghij", path.read_text(encoding="utf-8"))

            handler.close()

    def test_buffered_file_handler_flushes_errors_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffered.log"
            handler = BufferedFileHandler(path, flush_threshold_bytes=1024)
            handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))

            logger = logging.getLogger("test.buffered.error")
            logger.handlers = []
            logger.propagate = False
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            logger.error("boom")

            self.assertTrue(path.exists())
            self.assertIn("ERROR:boom", path.read_text(encoding="utf-8"))

            handler.close()


if __name__ == "__main__":
    unittest.main()
