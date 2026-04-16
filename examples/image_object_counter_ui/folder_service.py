from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

# Tự động nạp file .env để lấy API key của Supabase
load_dotenv()

import numpy as np
import supervision as sv
import cv2
try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime
    YOLO = None  # type: ignore[assignment]

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover - handled at runtime
    Client = Any  # type: ignore[assignment]
    create_client = None  # type: ignore[assignment]

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ImageProcessingQueue:
    """In-memory FIFO queue for sequential image processing.

    Example:
        >>> q = ImageProcessingQueue()
        >>> q.size
        0
    """

    def __init__(self) -> None:
        self._queue: deque[Path] = deque()
        self._enqueued: set[str] = set()

    @property
    def size(self) -> int:
        """Return current queue size."""
        return len(self._queue)

    def enqueue_many(self, paths: list[Path]) -> int:
        """Enqueue many image paths while deduplicating by absolute path.

        Args:
            paths: Candidate image file paths.

        Returns:
            Number of newly enqueued paths.
        """
        added = 0
        for path in sorted(paths):
            key = str(path.resolve())
            if key in self._enqueued:
                continue
            self._queue.append(path)
            self._enqueued.add(key)
            added += 1
        return added

    def pop_next(self) -> Path | None:
        """Pop next path from queue.

        Returns:
            Next queued path if available, otherwise None.
        """
        if not self._queue:
            return None
        path = self._queue.popleft()
        self._enqueued.discard(str(path.resolve()))
        return path


def _load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into process environment.

    Existing environment variables are preserved.

    Args:
        env_path: Path to .env file.
    """
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _bootstrap_env() -> None:
    """Load .env from common locations before parsing CLI args."""
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    _load_env_file(cwd / ".env")
    if script_dir != cwd:
        _load_env_file(script_dir / ".env")


class PersonDetector(Protocol):
    """Protocol for a detector that counts people in an image.

    Example:
        >>> class DemoDetector:
        ...     def count_people(self, image_path: Path) -> int:
        ...         return 3
        >>> DemoDetector().count_people(Path("image.jpg"))
        3
    """

    def count_people(self, image_path: Path) -> int:
        """Return number of detected people in an image path."""

    def count_and_annotate(self, image_path: Path) -> tuple[int, np.ndarray | None]:
        """Return people count and an annotated BGR image (if supported)."""


class CameraThresholdProvider(Protocol):
    """Protocol for required staffing threshold by camera.

    Example:
        >>> class DemoProvider:
        ...     def refresh(self) -> None:
        ...         return None
        ...     def get_required_count(self, camera_id: str) -> int:
        ...         return 2
        >>> DemoProvider().get_required_count("cam-a")
        2
    """

    def refresh(self) -> None:
        """Refresh in-memory thresholds if data source changed."""

    def get_required_count(self, camera_id: str) -> int:
        """Return required headcount for a camera."""


class ResultStore(Protocol):
    """Protocol for persisting processing results.

    Example:
        >>> class DemoStore:
        ...     def save(self, payload: dict[str, object], is_shortage: bool) -> None:
        ...         return None
        >>> DemoStore().save({"camera_id": "cam-a"}, False)
    """

    def save(self, payload: dict[str, object], status: str) -> None:
        """Persist one counting result and optional shortage alert."""


@dataclass(frozen=True)
class ProcessingResult:
    """Represents one processed image result.

    Example:
        >>> ProcessingResult("cam-a", "a.jpg", 2, 3, True).is_shortage
        True
    """

    camera_id: str
    source_file_name: str
    person_count: int
    required_count: int
    status: str


class YOLOPersonDetector:
    """YOLO-based detector that counts only the `person` class.

    Example:
        >>> isinstance("yolo11n.pt", str)
        True
    """

    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> None:
        """Initialize YOLO model and resolve the `person` class ID.

        Args:
            weights_path: Path to YOLO weights file.
            confidence_threshold: Detection confidence threshold.
            iou_threshold: IOU threshold for NMS.

        Raises:
            ValueError: If the loaded model has no `person` class.
        """
        if YOLO is None:
            raise ImportError(
                "Missing dependency 'ultralytics'. Install with: pip install -r requirements.txt"
            )
        self.model = YOLO(weights_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = self._resolve_person_class_id(self.model.names)

    @staticmethod
    def _resolve_person_class_id(model_names: dict[int, str]) -> int:
        """Find class ID for `person` in YOLO class names.

        Args:
            model_names: Mapping from class ID to class name.

        Returns:
            Class ID corresponding to `person`.

        Raises:
            ValueError: If `person` class is not found.
        """
        for class_id, class_name in model_names.items():
            if str(class_name).strip().lower() == "person":
                return int(class_id)
        raise ValueError("The selected YOLO model does not contain a 'person' class.")

    @staticmethod
    def _empty_detections() -> sv.Detections:
        """Create an empty detections object."""
        return sv.Detections(xyxy=np.empty((0, 4), dtype=np.float32))

    def _detect_people(self, image_path: Path) -> tuple[sv.Detections, np.ndarray]:
        """Run YOLO and return detections + BGR frame.

        Args:
            image_path: Path to input image.

        Returns:
            Tuple of people detections and original BGR frame.
        """
        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            return self._empty_detections(), np.zeros((1, 1, 3), dtype=np.uint8)

        result = self.model(
            frame_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        if len(detections) == 0 or detections.class_id is None:
            return self._empty_detections(), frame_bgr
        people = detections[detections.class_id == self.person_class_id]
        return people, frame_bgr

    def count_people(self, image_path: Path) -> int:
        """Count people in one image.

        Args:
            image_path: Path to input image.

        Returns:
            Number of person detections.
        """
        people, _ = self._detect_people(image_path)
        return len(people)

    def count_and_annotate(self, image_path: Path) -> tuple[int, np.ndarray | None]:
        """Count people and return annotated image with person boxes.

        Args:
            image_path: Path to input image.

        Returns:
            Tuple of (person_count, annotated_bgr_image).
        """
        people, frame_bgr = self._detect_people(image_path)
        if frame_bgr.size == 0:
            return 0, None

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        confidences = (
            people.confidence
            if people.confidence is not None
            else np.empty((len(people),), dtype=np.float32)
        )
        labels = [f"person {float(conf):.2f}" for conf in confidences]
        annotated = box_annotator.annotate(scene=frame_bgr.copy(), detections=people)
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=people,
            labels=labels,
        )
        return len(people), annotated


class JsonCameraThresholdProvider:
    """Threshold provider backed by a local JSON file.

    Example:
        >>> provider = JsonCameraThresholdProvider(Path("a.json"), 0)
        >>> isinstance(provider.default_required_count, int)
        True
    """

    def __init__(self, config_path: Path, default_required_count: int) -> None:
        """Create provider using local JSON map `camera_id -> required_count`.

        Args:
            config_path: Path to JSON file.
            default_required_count: Fallback when camera is missing.
        """
        self.config_path = config_path
        self.default_required_count = default_required_count
        self.thresholds: dict[str, int] = {}
        self.refresh()

    def refresh(self) -> None:
        """Reload thresholds from JSON file.

        Raises:
            ValueError: If file format is invalid.
        """
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Threshold config must be a JSON object.")

        thresholds: dict[str, int] = {}
        for camera_id, required in raw.items():
            if not isinstance(camera_id, str) or not camera_id.strip():
                raise ValueError("Each camera ID must be a non-empty string.")
            if not isinstance(required, int) or required < 0:
                raise ValueError(
                    f"Required count for camera '{camera_id}' must be a non-negative int."
                )
            thresholds[camera_id.strip()] = required

        self.thresholds = thresholds

    def get_required_count(self, camera_id: str) -> int:
        """Return required count for one camera.

        Args:
            camera_id: Camera identifier.

        Returns:
            Required headcount.
        """
        return self.thresholds.get(camera_id, self.default_required_count)


class SupabaseCameraThresholdProvider:
    """Threshold provider backed by Supabase table.

    Expected table columns:
        - camera_id (text)
        - required_count (int)
        - is_active (bool, optional)

    Example:
        >>> isinstance(0, int)
        True
    """

    def __init__(
        self,
        client: Client,
        table_name: str,
        default_required_count: int,
    ) -> None:
        """Create provider loading camera thresholds from Supabase.

        Args:
            client: Supabase client.
            table_name: Source table name.
            default_required_count: Fallback when camera not found.
        """
        self.client = client
        self.table_name = table_name
        self.default_required_count = default_required_count
        self.thresholds: dict[str, int] = {}

    def refresh(self) -> None:
        """Fetch latest camera thresholds from Supabase."""
        response = (
            self.client.table(self.table_name)
            .select("camera_id,required_count,is_active")
            .execute()
        )
        rows = response.data or []

        thresholds: dict[str, int] = {}
        for row in rows:
            camera_id = row.get("camera_id")
            required_count = row.get("required_count")
            is_active = row.get("is_active", True)
            if not isinstance(camera_id, str) or not camera_id.strip():
                continue
            if isinstance(required_count, int) and required_count >= 0 and is_active:
                thresholds[camera_id.strip()] = required_count

        self.thresholds = thresholds

    def get_required_count(self, camera_id: str) -> int:
        """Return required count for one camera.

        Args:
            camera_id: Camera identifier.

        Returns:
            Required headcount.
        """
        return self.thresholds.get(camera_id, self.default_required_count)


class SupabaseResultStore:
    """Store counting results and shortage alerts to Supabase.

    Example:
        >>> isinstance("person_count_results", str)
        True
    """

    def __init__(
        self,
        client: Client,
        results_table: str,
        alerts_table: str,
    ) -> None:
        """Create result store for Supabase tables.

        Args:
            client: Supabase client.
            results_table: Table for all count results.
            alerts_table: Table for shortage alerts.
        """
        self.client = client
        self.results_table = results_table
        self.alerts_table = alerts_table

    def save(self, payload: dict[str, object], status: str) -> None:
        """Insert one result row and optional shortage alert.

        Args:
            payload: Row payload for results table.
            status: Tình trạng (SHORTAGE, SUFFICIENT, SURPLUS).
        """
        response = self.client.table(self.results_table).insert(payload).execute()
        
        # Bắt mã ID của Kết Quả Check từ DB để liên kết làm khóa ngoại cho bảng Alert
        if not response.data:
            return
        
        inserted_row = response.data[0]
        result_id = inserted_row.get("id")

        # Cảnh báo sẽ sinh ra nếu THIẾU NGƯỜI (SHORTAGE) hoặc THỪA NGƯỜI (SURPLUS)
        if status != "SUFFICIENT" and result_id:
            alert_payload = {
                "result_id": result_id,
                "created_at": payload["processed_at"],
                "camera_id": payload["camera_id"],
                "source_file_name": payload["source_file_name"],
                "person_count": payload["person_count"],
                "required_count": payload["required_count"],
                "status": "OPEN",
            }
            self.client.table(self.alerts_table).insert(alert_payload).execute()


def infer_camera_id(incoming_root: Path, image_path: Path) -> str:
    """Infer camera ID from image path relative to incoming root.

    The convention is: `incoming/<camera_id>/<image_name>`.
    If an image is directly under `incoming/`, returns `default`.

    Args:
        incoming_root: Input root directory.
        image_path: Image file path to process.

    Returns:
        Camera ID extracted from first relative path segment.

    Example:
        >>> infer_camera_id(Path("incoming"), Path("incoming/cam-1/a.jpg"))
        'cam-1'
    """
    relative_path = image_path.relative_to(incoming_root)
    if len(relative_path.parts) <= 1:
        return "default"
    return relative_path.parts[0]


def build_archive_path(
    processed_root: Path,
    camera_id: str,
    normalized_file_name: str,
    processed_at: datetime,
) -> Path:
    """Build target archive path grouped by date and camera.

    Args:
        processed_root: Base directory for processed images.
        camera_id: Camera identifier.
        normalized_file_name: Normalized file name (cameraid_time format).
        processed_at: Processing timestamp.

    Returns:
        Full target path under `processed_root/YYYY-MM-DD/<camera_id>/...`.

    Example:
        >>> dt = datetime(2026, 4, 16, 10, 0, 0)
        >>> str(build_archive_path(Path("done"), "cam-a", "cam-a_20260416_100000.jpg", dt))
        'done\\2026-04-16\\cam-a\\cam-a_20260416_100000.jpg'
    """
    date_folder = processed_at.date().isoformat()
    return processed_root / date_folder / camera_id / normalized_file_name


def sanitize_camera_id_for_filename(camera_id: str) -> str:
    """Sanitize camera ID for safe filename usage.

    Args:
        camera_id: Raw camera identifier.

    Returns:
        Filename-safe camera ID.

    Example:
        >>> sanitize_camera_id_for_filename("camera/a#1")
        'camera-a-1'
    """
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", camera_id.strip())
    sanitized = sanitized.strip("-_")
    return sanitized or "camera"


def build_normalized_file_name(
    camera_id: str,
    processed_at: datetime,
    extension: str,
) -> str:
    """Build normalized file name in `cameraid_time` format.

    Args:
        camera_id: Camera identifier.
        processed_at: Processing timestamp.
        extension: File extension including leading dot.

    Returns:
        Normalized file name.

    Example:
        >>> t = datetime(2026, 4, 16, 10, 0, 0)
        >>> build_normalized_file_name("cam-a", t, ".jpg")
        'cam-a_20260416_100000.jpg'
    """
    safe_camera_id = sanitize_camera_id_for_filename(camera_id)
    time_part = processed_at.strftime("%Y%m%d_%H%M%S")
    return f"{safe_camera_id}_{time_part}{extension.lower()}"


def _resolve_unique_destination(destination_path: Path) -> Path:
    """Return unique destination path if file name already exists.

    Args:
        destination_path: Desired target file path.

    Returns:
        Existing path if free, or suffixed path like `_1`, `_2`, ...
    """
    if not destination_path.exists():
        return destination_path

    stem = destination_path.stem
    suffix = destination_path.suffix
    parent = destination_path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _build_annotated_destination(original_destination: Path) -> Path:
    """Build destination path for annotated image copy.

    Args:
        original_destination: Archived original file path.

    Returns:
        Annotated destination path under `annotated/`.
    """
    annotated_dir = original_destination.parent / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    annotated_name = f"{original_destination.stem}_boxed{original_destination.suffix}"
    return _resolve_unique_destination(annotated_dir / annotated_name)


def discover_pending_images(incoming_root: Path) -> list[Path]:
    """Discover supported image files under incoming root.

    Args:
        incoming_root: Root folder containing unprocessed images.

    Returns:
        List of discovered image paths.
    """
    if not incoming_root.exists():
        return []
    return [
        path
        for path in incoming_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]


def _process_single_image(
    image_path: Path,
    incoming_root: Path,
    processed_root: Path,
    threshold_provider: CameraThresholdProvider,
    detector: PersonDetector,
    result_store: ResultStore,
    processed_at: datetime,
) -> ProcessingResult:
    """Process one image and persist its result.

    Args:
        image_path: Input image path.
        incoming_root: Incoming root for camera inference.
        processed_root: Processed root for archiving.
        threshold_provider: Threshold provider implementation.
        detector: Person detector implementation.
        result_store: Persistence implementation.
        processed_at: Timestamp for file naming and DB payload.

    Returns:
        Processing result object.
    """
    camera_id = infer_camera_id(incoming_root=incoming_root, image_path=image_path)
    normalized_name = build_normalized_file_name(
        camera_id=camera_id,
        processed_at=processed_at,
        extension=image_path.suffix,
    )
    normalized_incoming_path = _resolve_unique_destination(
        image_path.with_name(normalized_name)
    )
    if normalized_incoming_path != image_path:
        shutil.move(str(image_path), str(normalized_incoming_path))
    image_path = normalized_incoming_path

    required_count = threshold_provider.get_required_count(camera_id)
    annotated_frame: np.ndarray | None = None
    if hasattr(detector, "count_and_annotate"):
        person_count, annotated_frame = detector.count_and_annotate(image_path)
    else:
        person_count = detector.count_people(image_path)

    if person_count < required_count:
        status = "SHORTAGE"
    elif person_count > required_count:
        status = "SURPLUS"
    else:
        status = "SUFFICIENT"

    destination_path = build_archive_path(
        processed_root=processed_root,
        camera_id=camera_id,
        normalized_file_name=image_path.name,
        processed_at=processed_at,
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path = _resolve_unique_destination(destination_path)

    source_file_path = str(image_path.resolve())
    shutil.move(str(image_path), str(destination_path))

    if annotated_frame is not None:
        annotated_destination = _build_annotated_destination(destination_path)
        cv2.imwrite(str(annotated_destination), annotated_frame)

    result_payload = {
        "processed_at": processed_at.isoformat(timespec="seconds"),
        "camera_id": camera_id,
        "source_file_name": Path(source_file_path).name,
        "source_file_path": source_file_path,
        "archived_file_path": str(destination_path.resolve()),
        "person_count": person_count,
        "required_count": required_count,
        "status": status,
    }
    result_store.save(result_payload, status=status)

    return ProcessingResult(
        camera_id=camera_id,
        source_file_name=Path(source_file_path).name,
        person_count=person_count,
        required_count=required_count,
        status=status,
    )


def process_pending_queue_once(
    queue: ImageProcessingQueue,
    incoming_root: Path,
    processed_root: Path,
    threshold_provider: CameraThresholdProvider,
    detector: PersonDetector,
    result_store: ResultStore,
) -> list[ProcessingResult]:
    """Process all currently queued items sequentially.

    Args:
        queue: Shared processing queue.
        incoming_root: Root folder containing unprocessed images.
        processed_root: Root folder containing archived images.
        threshold_provider: Source of camera required counts.
        detector: Detector implementation.
        result_store: Persistence target.

    Returns:
        List of successful processing results.
    """
    results: list[ProcessingResult] = []
    while True:
        image_path = queue.pop_next()
        if image_path is None:
            break
        if not image_path.exists():
            continue

        try:
            result = _process_single_image(
                image_path=image_path,
                incoming_root=incoming_root,
                processed_root=processed_root,
                threshold_provider=threshold_provider,
                detector=detector,
                result_store=result_store,
                processed_at=datetime.now(),
            )
            results.append(result)
        except Exception as e:
            print(
                "failed",
                f"file={image_path}",
                f"error={e}",
            )
            continue
    return results


def process_pending_images_once(
    incoming_root: Path,
    processed_root: Path,
    threshold_provider: CameraThresholdProvider,
    detector: PersonDetector,
    result_store: ResultStore,
    processed_at: datetime | None = None,
) -> list[ProcessingResult]:
    """Process all pending images once.

    Args:
        incoming_root: Root folder containing unprocessed images.
        processed_root: Root folder containing archived processed images.
        threshold_provider: Source of camera required counts.
        detector: Detector implementation that counts people.
        result_store: Persistence target (for example Supabase).
        processed_at: Optional fixed timestamp (useful for tests).

    Returns:
        List of processing results.

    Example:
        >>> isinstance([], list)
        True
    """
    incoming_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    queue = ImageProcessingQueue()
    queue.enqueue_many(discover_pending_images(incoming_root))
    return process_pending_queue_once(
        queue=queue,
        incoming_root=incoming_root,
        processed_root=processed_root,
        threshold_provider=threshold_provider,
        detector=detector,
        result_store=result_store,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI parser for folder-based people counting service.

    Returns:
        Configured argument parser.

    Example:
        >>> parser = build_argument_parser()
        >>> isinstance(parser.prog, str)
        True
    """
    parser = argparse.ArgumentParser(
        description=(
            "Watch incoming image folder, count people only, archive images, and "
            "persist results + shortage alerts into Supabase."
        )
    )
    parser.add_argument(
        "--incoming-dir",
        type=Path,
        default=Path("incoming"),
        help="Folder containing unprocessed images.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed"),
        help="Folder where processed images will be archived by date/camera.",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="yolo11n.pt",
        help="YOLO weights path.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.30,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="Detection IOU threshold.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=5,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--default-required-count",
        type=int,
        default=0,
        help="Fallback required headcount when camera is not configured.",
    )
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        default=None,
        help="Optional local JSON camera thresholds instead of Supabase table.",
    )
    parser.add_argument(
        "--supabase-url",
        type=str,
        default=os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL"),
        help="Supabase URL (or set SUPABASE_URL / NEXT_PUBLIC_SUPABASE_URL).",
    )
    parser.add_argument(
        "--supabase-key",
        type=str,
        default=os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_ANON_KEY"),
        help=(
            "Supabase API key (prefer service role key). "
            "Read from SUPABASE_SERVICE_ROLE_KEY, SUPABASE_KEY, or SUPABASE_ANON_KEY."
        ),
    )
    parser.add_argument(
        "--camera-targets-table",
        type=str,
        default="camera_staffing_targets",
        help="Supabase table name for camera thresholds.",
    )
    parser.add_argument(
        "--results-table",
        type=str,
        default="person_count_results",
        help="Supabase table name for count results.",
    )
    parser.add_argument(
        "--alerts-table",
        type=str,
        default="staffing_alerts",
        help="Supabase table name for shortage alerts.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Process available images once and exit.",
    )
    return parser


def _build_threshold_provider(args: argparse.Namespace, client: Client) -> CameraThresholdProvider:
    """Create threshold provider from CLI args.

    Args:
        args: Parsed CLI arguments.
        client: Supabase client.

    Returns:
        Threshold provider implementation.
    """
    if args.thresholds_json is not None:
        return JsonCameraThresholdProvider(
            config_path=args.thresholds_json,
            default_required_count=args.default_required_count,
        )

    return SupabaseCameraThresholdProvider(
        client=client,
        table_name=args.camera_targets_table,
        default_required_count=args.default_required_count,
    )


def main() -> None:
    """Run folder-based people counting service.

    Example:
        >>> isinstance(1 + 1, int)
        True
    """
    _bootstrap_env()

    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.supabase_url or not args.supabase_key:
        raise ValueError(
            "Supabase credentials missing. Set --supabase-url/--supabase-key "
            "or environment variables SUPABASE_URL + one of: "
            "SUPABASE_SERVICE_ROLE_KEY, SUPABASE_KEY, SUPABASE_ANON_KEY."
        )
    if create_client is None:
        raise ImportError(
            "Missing dependency 'supabase'. Install with: pip install -r requirements.txt"
        )

    client = create_client(args.supabase_url, args.supabase_key)

    threshold_provider = _build_threshold_provider(args=args, client=client)
    result_store = SupabaseResultStore(
        client=client,
        results_table=args.results_table,
        alerts_table=args.alerts_table,
    )
    detector = YOLOPersonDetector(
        weights_path=args.weights_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    queue = ImageProcessingQueue()

    while True:
        threshold_provider.refresh()
        discovered = discover_pending_images(args.incoming_dir)
        enqueued = queue.enqueue_many(discovered)
        if enqueued:
            print("queue", f"enqueued={enqueued}", f"size={queue.size}")

        results = process_pending_queue_once(
            queue=queue,
            incoming_root=args.incoming_dir,
            processed_root=args.processed_dir,
            threshold_provider=threshold_provider,
            detector=detector,
            result_store=result_store,
        )
        if results:
            for result in results:
                print(
                    "processed",
                    f"camera={result.camera_id}",
                    f"file={result.source_file_name}",
                    f"people={result.person_count}",
                    f"required={result.required_count}",
                    f"status={result.status}",
                )
        elif queue.size == 0:
            print("queue", "idle", "no-pending-images")

        if args.run_once:
            break

        time.sleep(max(args.poll_interval_seconds, 1))


if __name__ == "__main__":
    main()
