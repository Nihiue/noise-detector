from __future__ import annotations

import argparse
import signal
import threading
from types import FrameType
from typing import Callable, Optional

from audio_detect.audio_devices import list_input_devices
from audio_detect.classifier import Classifier
from audio_detect.collector import CollectorService
from audio_detect.config import load_settings
from audio_detect.logging_utils import configure_logging
from audio_detect.storage import EventStore
from audio_detect.web import create_app
from werkzeug.serving import BaseWSGIServer, make_server


def serve_app(
    app,
    *,
    host: str,
    port: int,
    collector: CollectorService,
    server_factory: Callable[..., BaseWSGIServer] = make_server,
) -> None:
    server = server_factory(host, port, app, threaded=True)
    shutdown_event = threading.Event()
    restore_handlers = _install_signal_handlers(shutdown_event)
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="web-server",
        daemon=True,
    )
    server_thread.start()
    try:
        while server_thread.is_alive():
            if shutdown_event.wait(0.2):
                break
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        server.shutdown()
        server_thread.join(timeout=2.0)
        collector.stop()
        server.server_close()
        restore_handlers()


def _install_signal_handlers(shutdown_event: threading.Event) -> Callable[[], None]:
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _raise_keyboard_interrupt(
        signum: int,
        frame: Optional[FrameType],
    ) -> None:
        shutdown_event.set()
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    def _restore_handlers() -> None:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    return _restore_handlers


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise detection service")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML configuration file",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        devices = list_input_devices()
        if not devices:
            print("No input devices found or sounddevice backend unavailable.")
            return
        for item in devices:
            print(
                "{index}: {name} (inputs={inputs}, default_sr={sr})".format(
                    index=item.index,
                    name=item.name,
                    inputs=item.max_input_channels,
                    sr=int(item.default_samplerate),
                )
            )
        return

    settings = load_settings(args.config)
    configure_logging(settings.logging)
    event_store = EventStore(
        records_dir=settings.storage.records_dir,
        database_path=settings.storage.database_path,
        max_record_days=settings.retention.max_record_days,
        max_records=settings.retention.max_records,
    )
    classifier = Classifier(settings.classification)
    classifier.ensure_available()
    collector = CollectorService(
        audio_config=settings.audio,
        detection_config=settings.detection,
        event_store=event_store,
        classifier=classifier,
    )
    collector.start()

    app = create_app(event_store, collector)
    try:
        serve_app(
            app,
            host=settings.app.host,
            port=settings.app.port,
            collector=collector,
        )
    finally:
        event_store.close()
