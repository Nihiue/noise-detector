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

    @app.get("/api/audio-devices")
    def api_audio_devices():
        return jsonify(
            {
                "devices": [_device_to_dict(item) for item in collector.list_input_devices()],
                "selected_device_index": collector.get_status().selected_device_index,
                "configured_device_name": collector.configured_device_name,
                "can_switch": collector.can_switch_input_device(),
            }
        )

    @app.post("/api/audio-device")
    def api_audio_device():
        payload = request.get_json(silent=True) or {}
        try:
            device_index = _parse_positive_int_like(
                payload.get("device_index"),
                field_name="device_index",
            )
            status = collector.switch_input_device(device_index)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(_status_to_dict(status))

    @app.get("/api/events")
    def api_events():
        filters = _read_event_filters()
        try:
            pagination = _read_pagination()
            events, total = event_store.query_events_page(
                classification=filters["classification"] or None,
                start_at=filters["start_at"] or None,
                end_at=filters["end_at"] or None,
                page=pagination["page"],
                page_size=pagination["page_size"],
            )
        except (InvalidTimeFilterError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        total_pages = max(1, (total + pagination["page_size"] - 1) // pagination["page_size"])
        return jsonify(
            {
                "events": [_event_to_dict(event) for event in events],
                "pagination": {
                    "page": pagination["page"],
                    "page_size": pagination["page_size"],
                    "total": total,
                    "total_pages": total_pages,
                },
            }
        )

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
        except (InvalidTimeFilterError, ValueError) as exc:
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
        "is_switching_device": status.is_switching_device,
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


def _device_to_dict(device) -> dict[str, object]:
    return {
        "index": device.index,
        "name": device.name,
        "max_input_channels": device.max_input_channels,
        "default_samplerate": device.default_samplerate,
    }


def _read_event_filters() -> dict[str, str]:
    return {
        "classification": request.args.get("classification", "").strip(),
        "start_at": request.args.get("start_at", "").strip(),
        "end_at": request.args.get("end_at", "").strip(),
    }


def _read_pagination() -> dict[str, int]:
    page = _parse_positive_int(request.args.get("page", "1"), field_name="page")
    page_size = _parse_positive_int(request.args.get("page_size", "20"), field_name="page_size")
    return {
        "page": page,
        "page_size": min(page_size, 100),
    }


def _parse_positive_int(value: str, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("{field_name} must be a positive integer".format(field_name=field_name))
    if parsed <= 0:
        raise ValueError("{field_name} must be a positive integer".format(field_name=field_name))
    return parsed


def _parse_positive_int_like(value: object, *, field_name: str) -> int:
    if value is None:
        raise ValueError("{field_name} is required".format(field_name=field_name))
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("{field_name} must be an integer".format(field_name=field_name))
    if parsed < 0:
        raise ValueError("{field_name} must be zero or a positive integer".format(field_name=field_name))
    return parsed
