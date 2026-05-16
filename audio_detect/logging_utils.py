from __future__ import annotations

import io
import logging
from pathlib import Path

from audio_detect.config import LoggingConfig


class BufferedFileHandler(logging.Handler):
    def __init__(
        self,
        path: Path,
        *,
        flush_threshold_bytes: int = 16 * 1024,
        flush_on_level: int = logging.ERROR,
    ) -> None:
        super().__init__()
        self.path = path
        self.flush_threshold_bytes = flush_threshold_bytes
        self.flush_on_level = flush_on_level
        self._buffer = io.StringIO()
        self._buffer_bytes = 0
        self._stream = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = self.format(record) + "\n"
            encoded = payload.encode("utf-8")
            self.acquire()
            try:
                self._buffer.write(payload)
                self._buffer_bytes += len(encoded)
                if (
                    self._buffer_bytes >= self.flush_threshold_bytes
                    or record.levelno >= self.flush_on_level
                ):
                    self._flush_locked()
            finally:
                self.release()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        self.acquire()
        try:
            self._flush_locked()
        finally:
            self.release()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            super().close()

    def _flush_locked(self) -> None:
        if self._buffer_bytes == 0:
            return
        if self._stream is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("a", encoding="utf-8")
        self._stream.write(self._buffer.getvalue())
        self._stream.flush()
        self._buffer.seek(0)
        self._buffer.truncate(0)
        self._buffer_bytes = 0


def configure_logging(config: LoggingConfig) -> None:
    config.file_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    root_logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = BufferedFileHandler(config.file_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
