from __future__ import annotations

import json
import logging
import sqlite3
import threading
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class NoiseEvent:
    timestamp: str
    filename: str
    peak_rms: float
    duration_seconds: float
    classification: str
    classification_score: float = 0.0
    top_classes: Optional[List[str]] = None


class InvalidTimeFilterError(ValueError):
    def __init__(self, field_name: str, value: str) -> None:
        super().__init__(
            "{field_name} must use YYYYMMDD format: {value}".format(
                field_name=field_name,
                value=value,
            )
        )
        self.field_name = field_name
        self.value = value


@dataclass(frozen=True)
class EventQuery:
    where_sql: str
    params: Tuple[object, ...]


class EventStore:
    def __init__(
        self,
        records_dir: Path,
        database_path: Path,
        max_record_days: int = 7,
        max_records: int = 500,
    ) -> None:
        self.records_dir = records_dir
        self.database_path = database_path
        self.max_record_days = max_record_days
        self.max_records = max_records
        self.logger = logging.getLogger("audio_detect.storage")
        self._lock = threading.RLock()
        self._closed = False
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        self._initialize_database()

    def append(self, event: NoiseEvent) -> None:
        timestamp_unix = _to_unix_timestamp(event.timestamp)
        if timestamp_unix is None:
            raise ValueError(
                "event timestamp must be a valid ISO 8601 value: {value}".format(
                    value=event.timestamp
                )
            )
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO noise_events (
                    timestamp,
                    timestamp_unix,
                    filename,
                    peak_rms,
                    duration_seconds,
                    classification,
                    classification_score,
                    top_classes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.timestamp,
                    timestamp_unix,
                    event.filename,
                    event.peak_rms,
                    event.duration_seconds,
                    event.classification,
                    event.classification_score,
                    json.dumps(event.top_classes or [], ensure_ascii=True),
                ),
            )
            removed_rows = self._remove_old_rows(conn)
            removed_rows += self._remove_excess_rows(conn)
            if removed_rows:
                self.logger.info("retention completed removed_rows=%s", removed_rows)

    def list_events(self) -> List[NoiseEvent]:
        return self.query_events()

    def query_events(
        self,
        classification: Optional[str] = None,
        start_at: Optional[str] = None,
        end_at: Optional[str] = None,
    ) -> List[NoiseEvent]:
        query = self._build_event_query(
            classification=classification,
            start_at=start_at,
            end_at=end_at,
        )
        sql = """
            SELECT
                timestamp,
                filename,
                peak_rms,
                duration_seconds,
                classification,
                classification_score,
                top_classes_json
            FROM noise_events
        """
        sql += query.where_sql
        sql += " ORDER BY timestamp_unix DESC, id DESC"

        with self._lock:
            rows = self._conn.execute(sql, query.params).fetchall()
        return [self._row_to_event(row) for row in rows]

    def query_events_page(
        self,
        *,
        classification: Optional[str] = None,
        start_at: Optional[str] = None,
        end_at: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[NoiseEvent], int]:
        query = self._build_event_query(
            classification=classification,
            start_at=start_at,
            end_at=end_at,
        )
        sql = """
            SELECT
                timestamp,
                filename,
                peak_rms,
                duration_seconds,
                classification,
                classification_score,
                top_classes_json
            FROM noise_events
        """
        sql += query.where_sql
        sql += " ORDER BY timestamp_unix DESC, id DESC LIMIT ? OFFSET ?"
        params = query.params + (page_size, (page - 1) * page_size)

        count_sql = "SELECT COUNT(*) FROM noise_events" + query.where_sql
        with self._lock:
            total = int(self._conn.execute(count_sql, query.params).fetchone()[0])
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_event(row) for row in rows], total

    def distinct_classifications(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT classification
                FROM noise_events
                WHERE classification IS NOT NULL AND classification != ''
                ORDER BY classification ASC
                """
            ).fetchall()
        return [row[0] for row in rows]

    def build_export_zip(self, events: List[NoiseEvent]) -> BytesIO:
        archive_buffer = BytesIO()
        with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "noise_events.jsonl",
                "".join(json.dumps(asdict(item), ensure_ascii=True) + "\n" for item in events),
            )
            for event in events:
                record_path = self.records_dir / event.filename
                if record_path.exists():
                    archive.write(record_path, arcname="records/{name}".format(name=event.filename))
        archive_buffer.seek(0)
        return archive_buffer

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    def _remove_old_rows(self, conn: sqlite3.Connection) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_record_days)
        cutoff_unix = cutoff.timestamp()
        rows_to_delete = self._select_rows(
            conn,
            """
            SELECT id, filename
            FROM noise_events
            WHERE timestamp_unix < ?
            """,
            (cutoff_unix,),
        )
        self._delete_rows(conn, rows_to_delete)
        return len(rows_to_delete)

    def _remove_excess_rows(self, conn: sqlite3.Connection) -> int:
        rows_to_delete = self._select_rows(
            conn,
            """
            SELECT id, filename
            FROM noise_events
            ORDER BY timestamp_unix DESC, id DESC
            LIMIT -1 OFFSET ?
            """,
            (self.max_records,),
        )
        self._delete_rows(conn, rows_to_delete)
        return len(rows_to_delete)

    def _select_rows(
        self,
        conn: sqlite3.Connection,
        sql: str,
        params: Tuple[object, ...],
    ) -> List[Tuple[int, str]]:
        rows = conn.execute(sql, params).fetchall()
        return [(row["id"], row["filename"]) for row in rows]

    def _delete_rows(
        self,
        conn: sqlite3.Connection,
        rows: List[Tuple[int, str]],
    ) -> None:
        if not rows:
            return
        row_ids = []
        for row_id, filename in rows:
            record_path = self.records_dir / filename
            if record_path.exists():
                record_path.unlink()
            row_ids.append((row_id,))
        conn.executemany("DELETE FROM noise_events WHERE id = ?", row_ids)

    def _initialize_database(self) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS noise_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    timestamp_unix REAL NOT NULL,
                    filename TEXT NOT NULL,
                    peak_rms REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    classification TEXT NOT NULL,
                    classification_score REAL NOT NULL DEFAULT 0.0,
                    top_classes_json TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_noise_events_timestamp_unix
                ON noise_events(timestamp_unix DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_noise_events_classification
                ON noise_events(classification)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.database_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _build_event_query(
        *,
        classification: Optional[str],
        start_at: Optional[str],
        end_at: Optional[str],
    ) -> EventQuery:
        clauses = []
        params: List[object] = []

        if classification:
            clauses.append("classification = ?")
            params.append(classification.strip())

        start_unix = _parse_date_filter(
            start_at,
            field_name="start_at",
            range_edge="start",
            strict=True,
        )
        if start_unix is not None:
            clauses.append("timestamp_unix >= ?")
            params.append(start_unix)

        end_unix = _parse_date_filter(
            end_at,
            field_name="end_at",
            range_edge="end",
            strict=True,
        )
        if end_unix is not None:
            clauses.append("timestamp_unix <= ?")
            params.append(end_unix)

        if (
            start_unix is not None
            and end_unix is not None
            and start_unix > end_unix
        ):
            raise InvalidTimeFilterError(
                "time_range",
                "{start} > {end}".format(start=start_at, end=end_at),
            )

        where_sql = ""
        if clauses:
            where_sql = " WHERE " + " AND ".join(clauses)
        return EventQuery(where_sql=where_sql, params=tuple(params))

    def _transaction(self):
        return _LockedConnection(self._conn, self._lock)

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> NoiseEvent:
        return NoiseEvent(
            timestamp=row["timestamp"],
            filename=row["filename"],
            peak_rms=row["peak_rms"],
            duration_seconds=row["duration_seconds"],
            classification=row["classification"],
            classification_score=row["classification_score"],
            top_classes=json.loads(row["top_classes_json"] or "[]"),
        )


def _to_unix_timestamp(value: Optional[str]) -> Optional[float]:
    return _parse_unix_timestamp(value, field_name="timestamp", strict=False)


def _parse_date_filter(
    value: Optional[str],
    *,
    field_name: str,
    range_edge: str,
    strict: bool,
) -> Optional[float]:
    if not value:
        return None
    try:
        dt = datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
        if range_edge == "end":
            dt = dt + timedelta(days=1) - timedelta(microseconds=1)
        return dt.timestamp()
    except ValueError:
        if strict:
            raise InvalidTimeFilterError(field_name, value)
        return None


def _parse_unix_timestamp(
    value: Optional[str],
    *,
    field_name: str,
    strict: bool,
) -> Optional[float]:
    if not value:
        return None
    try:
        normalized = value
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).timestamp()
    except ValueError:
        if strict:
            raise InvalidTimeFilterError(field_name, value)
        return None


class _LockedConnection:
    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = conn
        self._lock = lock

    def __enter__(self) -> sqlite3.Connection:
        self._lock.acquire()
        self._conn.__enter__()
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            return bool(self._conn.__exit__(exc_type, exc, tb))
        finally:
            self._lock.release()
