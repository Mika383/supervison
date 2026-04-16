from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import supervision as sv
from PIL import Image
from ultralytics import YOLO

from ui_components import (
    render_camera_config_page,
    render_folder_management_page,
    render_history_page,
    render_sidebar_navigation,
    render_sidebar_status,
)


@st.cache_resource
def load_model(weights_path: str) -> YOLO:
    """Load and cache a YOLO model by weights path."""
    return YOLO(weights_path)


def annotate_image(
    image_bgr: np.ndarray,
    detections: sv.Detections,
    labels: list[str],
) -> np.ndarray:
    """Annotate image with bounding boxes and labels."""
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )
    return annotated


def render_manual_detection_page() -> None:
    """Render manual detection page in Vietnamese."""
    st.subheader("Đếm Người Từ Ảnh")
    st.caption("Tải ảnh lên và đếm người (áp dụng nhận diện lớp 'person').")

    with st.container(border=True):
        col_model, col_conf, col_iou = st.columns(3)
        with col_model:
            weights_path = st.text_input(
                "Đường dẫn mô hình YOLO",
                value="yolo11n.pt",
            )
        with col_conf:
            confidence_threshold = st.slider(
                "Ngưỡng tin cậy",
                0.0,
                1.0,
                0.30,
                0.01,
            )
        with col_iou:
            iou_threshold = st.slider(
                "Ngưỡng IOU",
                0.0,
                1.0,
                0.50,
                0.01,
            )

    uploaded_file = st.file_uploader(
        "Tải ảnh đầu vào",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is None:
        st.info("Hãy tải ảnh lên để bắt đầu.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    model = load_model(weights_path)
    model_names = model.names
    person_class_id = next(
        (
            class_id
            for class_id, class_name in model_names.items()
            if str(class_name).strip().lower() == "person"
        ),
        None,
    )
    if person_class_id is None:
        st.error("Mô hình hiện tại không có lớp 'person'.")
        return

    if not st.button("Chạy đếm người", type="primary"):
        st.image(image_rgb, caption="Ảnh đầu vào", use_container_width=True)
        return

    result = model(
        image_bgr,
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False,
    )[0]
    detections = sv.Detections.from_ultralytics(result)
    if detections.class_id is None or len(detections) == 0:
        detections = detections[:0]
    else:
        detections = detections[detections.class_id == person_class_id]

    labels: list[str] = []
    for confidence, _ in zip(detections.confidence, detections.class_id):
        labels.append(f"person {float(confidence):.2f}")

    annotated_bgr = annotate_image(
        image_bgr=image_bgr,
        detections=detections,
        labels=labels,
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    left_col, right_col = st.columns(2)
    with left_col:
        st.image(image_rgb, caption="Ảnh gốc", use_container_width=True)
    with right_col:
        st.image(annotated_rgb, caption="Ảnh đã chú thích", use_container_width=True)

    st.metric("Số người phát hiện", value=len(detections))

    output_image = Image.fromarray(annotated_rgb)
    output_buffer = BytesIO()
    output_image.save(output_buffer, format="PNG")
    st.download_button(
        label="Tải ảnh kết quả",
        data=output_buffer.getvalue(),
        file_name="ket_qua_dem_nguoi.png",
        mime="image/png",
    )


def main() -> None:
    """Render app with sidebar navigation and functional UI bars."""
    st.set_page_config(
        page_title="Giám Sát Nhân Sự Theo Camera",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Giám Sát Nhân Sự Theo Camera")

    selected_page = render_sidebar_navigation()
    render_sidebar_status()

    if selected_page == "Đếm người từ ảnh":
        render_manual_detection_page()
    elif selected_page == "Quản lý thư mục":
        render_folder_management_page()
    elif selected_page == "Cấu hình camera":
        render_camera_config_page()
    else:
        render_history_page()


if __name__ == "__main__":
    main()
