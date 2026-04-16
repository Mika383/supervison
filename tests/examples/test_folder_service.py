from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "image_object_counter_ui"
    / "folder_service.py"
)
SPEC = importlib.util.spec_from_file_location("folder_service", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
folder_service = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(folder_service)


class FakeDetector:
    def __init__(self, values: dict[str, int]) -> None:
        self.values = values

    def count_people(self, image_path: Path) -> int:
        return self.values.get(image_path.name, 0)


class FakeThresholdProvider:
    def __init__(self, values: dict[str, int], default_required_count: int = 0) -> None:
        self.values = values
        self.default_required_count = default_required_count

    def refresh(self) -> None:
        return None

    def get_required_count(self, camera_id: str) -> int:
        return self.values.get(camera_id, self.default_required_count)


class FakeResultStore:
    def __init__(self) -> None:
        self.saved_payloads: list[dict[str, object]] = []
        self.alert_flags: list[bool] = []

    def save(self, payload: dict[str, object], is_shortage: bool) -> None:
        self.saved_payloads.append(payload)
        self.alert_flags.append(is_shortage)


def test_infer_camera_id_from_nested_path() -> None:
    incoming_root = Path("incoming")
    image_path = Path("incoming/camera-a/frame-001.jpg")

    camera_id = folder_service.infer_camera_id(incoming_root, image_path)

    assert camera_id == "camera-a"


def test_infer_camera_id_default_for_root_files() -> None:
    incoming_root = Path("incoming")
    image_path = Path("incoming/frame-001.jpg")

    camera_id = folder_service.infer_camera_id(incoming_root, image_path)

    assert camera_id == "default"


def test_process_pending_images_once_moves_and_persists_results(tmp_path: Path) -> None:
    incoming_root = tmp_path / "incoming"
    processed_root = tmp_path / "processed"

    camera_dir = incoming_root / "camera-a"
    camera_dir.mkdir(parents=True)
    image_path = camera_dir / "frame-001.jpg"
    image_path.write_bytes(b"dummy-jpg")

    detector = FakeDetector({"camera-a_20260416_093000.jpg": 2})
    threshold_provider = FakeThresholdProvider({"camera-a": 3})
    result_store = FakeResultStore()
    fixed_time = datetime(2026, 4, 16, 9, 30, 0)

    results = folder_service.process_pending_images_once(
        incoming_root=incoming_root,
        processed_root=processed_root,
        threshold_provider=threshold_provider,
        detector=detector,
        result_store=result_store,
        processed_at=fixed_time,
    )

    assert len(results) == 1
    assert results[0].is_shortage is True

    archived_dir = processed_root / "2026-04-16" / "camera-a"
    archived_files = list(archived_dir.glob("*.jpg"))
    assert len(archived_files) == 1
    archived_file = archived_files[0]
    assert archived_file.name.startswith("camera-a_20260416_093000")
    assert image_path.exists() is False

    assert len(result_store.saved_payloads) == 1
    saved = result_store.saved_payloads[0]
    assert saved["camera_id"] == "camera-a"
    assert str(saved["source_file_name"]).startswith("camera-a_20260416_093000")
    assert saved["person_count"] == 2
    assert saved["required_count"] == 3
    assert saved["is_shortage"] is True
    assert result_store.alert_flags == [True]


def test_process_pending_images_once_without_shortage_does_not_alert(
    tmp_path: Path,
) -> None:
    incoming_root = tmp_path / "incoming"
    processed_root = tmp_path / "processed"

    camera_dir = incoming_root / "camera-b"
    camera_dir.mkdir(parents=True)
    image_path = camera_dir / "frame-002.jpg"
    image_path.write_bytes(b"dummy-jpg")

    detector = FakeDetector({"camera-b_20260416_100000.jpg": 5})
    threshold_provider = FakeThresholdProvider({"camera-b": 3})
    result_store = FakeResultStore()

    results = folder_service.process_pending_images_once(
        incoming_root=incoming_root,
        processed_root=processed_root,
        threshold_provider=threshold_provider,
        detector=detector,
        result_store=result_store,
        processed_at=datetime(2026, 4, 16, 10, 0, 0),
    )

    assert len(results) == 1
    assert results[0].is_shortage is False
    assert results[0].source_file_name.startswith("camera-b_20260416_100000")
    assert result_store.alert_flags == [False]
