from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - backend import is environment-specific
    sd = None


@dataclass
class AudioDeviceInfo:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float


def list_input_devices() -> List[AudioDeviceInfo]:
    if sd is None:
        return []

    devices = []
    for index, item in enumerate(sd.query_devices()):
        if int(item.get("max_input_channels", 0)) < 1:
            continue
        devices.append(
            AudioDeviceInfo(
                index=index,
                name=str(item.get("name", "Unknown device")),
                max_input_channels=int(item.get("max_input_channels", 0)),
                default_samplerate=float(item.get("default_samplerate", 0.0)),
            )
        )
    return devices


def find_input_device(
    device_name: Optional[str] = None,
) -> Optional[AudioDeviceInfo]:
    devices = list_input_devices()
    if not devices:
        return None

    if device_name:
        keyword = device_name.strip().lower()
        for item in devices:
            if keyword in item.name.lower():
                return item

    return devices[0]


def get_input_device_by_index(device_index: int) -> Optional[AudioDeviceInfo]:
    for item in list_input_devices():
        if item.index == device_index:
            return item
    return None
