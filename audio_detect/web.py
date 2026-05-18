from __future__ import annotations

from datetime import datetime, timezone

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory, url_for

from audio_detect.collector import CollectorService
from audio_detect.storage import EventStore, InvalidTimeFilterError, NoiseEvent


def create_app(
    event_store: EventStore,
    collector: CollectorService,
) -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/status")
    def api_status():
        return jsonify(_status_to_dict(collector.get_status()))

    @app.post("/api/calibrate")
    def api_calibrate():
        return jsonify(_status_to_dict(collector.request_calibration()))

    @app.get("/api/events")
    def api_events():
        filters = _read_event_filters()
        try:
            events = event_store.query_events(
                classification=filters["classification"] or None,
                start_at=filters["start_at"] or None,
                end_at=filters["end_at"] or None,
            )
        except InvalidTimeFilterError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"events": [_event_to_dict(event) for event in events]})

    @app.get("/api/classifications")
    def api_classifications():
        return jsonify({"classifications": event_store.distinct_classifications()})

    @app.get("/export")
    def export_events():
        filters = _read_event_filters()
        try:
            events = event_store.query_events(
                classification=filters["classification"] or None,
                start_at=filters["start_at"] or None,
                end_at=filters["end_at"] or None,
            )
        except InvalidTimeFilterError as exc:
            abort(400, description=str(exc))
        archive = event_store.build_export_zip(events)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return send_file(
            archive,
            mimetype="application/zip",
            as_attachment=True,
            download_name="noise-events-{timestamp}.zip".format(timestamp=timestamp),
        )

    @app.get("/records/<path:filename>")
    def record_file(filename: str):
        path = event_store.records_dir / filename
        if not path.exists():
            abort(404)
        return send_from_directory(event_store.records_dir, filename)

    return app


def _status_to_dict(status) -> dict[str, object]:
    return {
        "selected_device_index": status.selected_device_index,
        "selected_device_name": status.selected_device_name,
        "is_running": status.is_running,
        "last_peak_rms": status.last_peak_rms,
        "last_peak_dbfs": status.last_peak_dbfs,
        "detection_window_seconds": status.detection_window_seconds,
        "capture_seconds": status.capture_seconds,
        "noise_floor_rms": status.noise_floor_rms,
        "noise_floor_std_rms": status.noise_floor_std_rms,
        "trigger_threshold_rms": status.trigger_threshold_rms,
        "noise_floor_dbfs": status.noise_floor_dbfs,
        "trigger_threshold_dbfs": status.trigger_threshold_dbfs,
        "gate_ready": status.gate_ready,
        "is_calibrating": status.is_calibrating,
        "last_gate_open": status.last_gate_open,
        "recent_detection_windows": status.recent_detection_windows,
        "last_error": status.last_error,
    }


def _event_to_dict(event: NoiseEvent) -> dict[str, object]:
    return {
        "timestamp": event.timestamp,
        "filename": event.filename,
        "record_url": url_for("record_file", filename=event.filename),
        "peak_rms": event.peak_rms,
        "duration_seconds": event.duration_seconds,
        "classification": event.classification,
        "classification_score": event.classification_score,
        "top_classes": event.top_classes or [],
    }


def _read_event_filters() -> dict[str, str]:
    return {
        "classification": request.args.get("classification", "").strip(),
        "start_at": request.args.get("start_at", "").strip(),
        "end_at": request.args.get("end_at", "").strip(),
    }
