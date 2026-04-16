from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

APP_ROOT = Path(__file__).resolve().parent
ENV_PATH = APP_ROOT / ".env"
# Always reload from this app's .env so old shell env values do not override.
load_dotenv(dotenv_path=ENV_PATH, override=True)

def _resolve_env_first(*keys: str) -> str:
    """Return first non-empty env var value by priority order."""
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


SUPABASE_URL = _resolve_env_first("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = _resolve_env_first(
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_KEY",
    "SUPABASE_ANON_KEY",
    "NEXT_PUBLIC_SUPABASE_ANON_KEY",
)

_supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Failed to load Supabase client: {e}")

def get_supabase() -> Client | None:
    return _supabase_client

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FOLDER_CONFIG_PATH = APP_ROOT / ".folder_ui_config.json"
CAMERA_CONFIG_PATH = APP_ROOT / ".camera_config.json"
GENERATED_THRESHOLDS_PATH = APP_ROOT / "camera_thresholds.generated.json"


def apply_sidebar_styles() -> None:
    """Apply custom visual styles for sidebar."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #0f0f10;
            color: #ffffff;
        }
        [data-testid="stHeader"] {
            background: #0f0f10;
        }
        [data-testid="stAppViewContainer"] {
            background: #0f0f10;
            color: #ffffff;
        }
        [data-testid="stMainBlockContainer"] {
            background: #0f0f10;
            color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #ffffff;
        }
        [data-testid="stMarkdownContainer"] p {
            color: #d8d8d8;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #1a1a1d;
            border: 1px solid #3a3a44;
        }
        [data-testid="stMetric"] {
            background: #17171a;
            border: 1px solid #2c2c33;
            border-radius: 10px;
            padding: 8px 10px;
        }
        [data-testid="stMetricLabel"] {
            color: #c7c7ce;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff;
        }
        [data-testid="stTextInputRootElement"] > div,
        [data-testid="stNumberInputContainer"] > div,
        [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            background: #17171a !important;
            color: #ffffff !important;
            border-color: #383845 !important;
        }
        [data-baseweb="select"] input,
        [data-baseweb="input"] input,
        [data-testid="stTextArea"] textarea {
            color: #ffffff !important;
        }
        [data-testid="stButton"] button,
        [data-testid="baseButton-secondary"] {
            background: #202028;
            color: #ffffff;
            border: 1px solid #3d3d49;
        }
        [data-testid="stButton"] button:hover {
            background: #2b2b35;
            border-color: #5a5a6a;
        }
        [data-testid="stCodeBlock"] pre,
        [data-testid="stCode"] {
            background: #16161a !important;
            color: #e8e8e8 !important;
            border: 1px solid #2f2f38;
        }
        [data-baseweb="tab-list"] {
            background: #141418;
            border: 1px solid #2c2c33;
            border-radius: 10px;
        }
        [data-baseweb="tab"] {
            color: #d2d2d8 !important;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            color: #ffffff !important;
            background: #23232c;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        [data-testid="stSidebar"] {
            background: #09090b;
            border-right: 1px solid #18181b;
        }
        [data-testid="stSidebar"] .sidebar-card {
            background: transparent;
            border: none;
            padding: 8px;
            margin-bottom: 16px;
            box-shadow: none;
        }
        [data-testid="stSidebar"] .sidebar-title {
            margin: 0;
            color: #fafafa;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        [data-testid="stSidebar"] .sidebar-subtitle {
            margin-top: 4px;
            margin-bottom: 0;
            color: #a1a1aa;
            font-size: 13px;
        }
        [data-testid="stSidebar"] .sidebar-section-title {
            margin: 16px 8px 8px 8px;
            color: #71717a;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }
        
        /* Ẩn hình tròn mặc định của Radio Button */
        [data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {
            display: none !important;
        }
        
        /* Style điều hướng menu phẳng siêu hiện đại */
        [data-testid="stSidebar"] [role="radiogroup"] > label {
            width: 100%;
            border: none !important;
            border-radius: 6px;
            background: transparent !important;
            margin-bottom: 4px;
            padding: 10px 12px 10px 16px !important;
            transition: all 0.2s ease;
            cursor: pointer;
            position: relative;
        }
        [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
            background: rgba(255, 255, 255, 0.05) !important;
        }
        
        /* Trạng thái Active - Hiển thị rõ ràng trang đang chọn */
        [data-testid="stSidebar"] [role="radiogroup"] > label[data-checked="true"],
        [data-testid="stSidebar"] [role="radiogroup"] > label[aria-checked="true"],
        [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
            background: rgba(255, 255, 255, 0.12) !important;
        }
        
        /* Bar line hiển thị bên trái của mục đang Active */
        [data-testid="stSidebar"] [role="radiogroup"] > label[data-checked="true"]::before,
        [data-testid="stSidebar"] [role="radiogroup"] > label[aria-checked="true"]::before,
        [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked)::before {
            content: "";
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 4px;
            height: 18px;
            background-color: #ffffff;
            border-radius: 0 4px 4px 0;
            box-shadow: 0 0 8px rgba(255,255,255,0.4);
        }
        
        /* Đổi màu text menu*/
        [data-testid="stSidebar"] [role="radiogroup"] > label p {
            color: #a1a1aa !important;
            font-size: 14px;
            font-weight: 500;
            margin: 0 !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] > label:hover p {
            color: #e4e4e7 !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] > label[data-checked="true"] p,
        [data-testid="stSidebar"] [role="radiogroup"] > label[aria-checked="true"] p,
        [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {
            color: #ffffff !important;
            font-weight: 700;
            letter-spacing: 0.2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def default_folder_settings() -> dict[str, str]:
    """Return default folder settings bound to app directory."""
    return {
        "incoming_dir": str((APP_ROOT / "incoming").resolve()),
        "processed_dir": str((APP_ROOT / "processed").resolve()),
    }


def load_folder_settings() -> dict[str, str]:
    """Load folder settings from local config file."""
    defaults = default_folder_settings()
    if not FOLDER_CONFIG_PATH.exists():
        return defaults

    raw = json.loads(FOLDER_CONFIG_PATH.read_text(encoding="utf-8"))
    incoming_dir = str(raw.get("incoming_dir", defaults["incoming_dir"]))
    processed_dir = str(raw.get("processed_dir", defaults["processed_dir"]))
    return {"incoming_dir": incoming_dir, "processed_dir": processed_dir}


def save_folder_settings(incoming_dir: str, processed_dir: str) -> None:
    """Persist folder settings to local JSON file."""
    payload = {"incoming_dir": incoming_dir, "processed_dir": processed_dir}
    FOLDER_CONFIG_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def count_pending_images(incoming_dir: Path) -> int:
    """Count images waiting for processing in incoming folder."""
    if not incoming_dir.exists():
        return 0
    return sum(
        1
        for path in incoming_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def list_pending_images(incoming_dir: Path, limit: int = 30) -> list[str]:
    """List pending image paths relative to incoming folder."""
    if not incoming_dir.exists():
        return []

    results: list[str] = []
    for path in sorted(incoming_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            results.append(str(path.relative_to(incoming_dir)))
        if len(results) >= limit:
            break
    return results


def load_camera_config() -> list[dict[str, Any]]:
    """Tải cấu hình camera từ db Supabase."""
    client = get_supabase()
    if not client:
        st.error("Chưa cấu hình Supabase (.env). Sẽ dùng dữ liệu rỗng.")
        return []

    try:
        response = client.table("camera_staffing_targets").select("*").execute()
        raw = response.data
    except Exception as e:
        st.error(f"Lỗi khi đọc từ Supabase DB: {e}")
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw:
        normalized.append({
            "camera_id": item.get("camera_id", ""),
            "required_count": item.get("required_count", 0),
            "active": item.get("is_active", True),
            "area_name": item.get("location_name", "") or "",
        })
    return normalized


def save_camera_config(cameras: list[dict[str, Any]]) -> None:
    """Lưu cấu hình camera trực tiếp lên DB Supabase."""
    client = get_supabase()
    if not client:
        st.error("Chưa cấu hình Supabase (.env). Không thể lưu DB.")
        return

    db_payload = []
    for cam in cameras:
        db_payload.append({
            "camera_id": cam["camera_id"],
            "required_count": cam["required_count"],
            "is_active": cam["active"],
            "location_name": cam["area_name"],
        })
    
    try:
        client.table("camera_staffing_targets").upsert(db_payload).execute()
        
        # Cập nhật thêm một file json backup cục bộ nội bộ nếu cần test chạy offline
        CAMERA_CONFIG_PATH.write_text(
            json.dumps(cameras, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        st.error(f"Lỗi khi lưu lên Supabase DB: {e}")


def build_threshold_map(cameras: list[dict[str, Any]]) -> dict[str, int]:
    """Build active camera threshold map from camera config list."""
    thresholds: dict[str, int] = {}
    for row in cameras:
        camera_id = str(row.get("camera_id", "")).strip()
        if not camera_id:
            continue
        if not bool(row.get("active", True)):
            continue
        required_count = int(row.get("required_count", 0))
        thresholds[camera_id] = max(required_count, 0)
    return thresholds


def save_generated_thresholds(threshold_map: dict[str, int]) -> None:
    """Save generated threshold map used by folder service."""
    GENERATED_THRESHOLDS_PATH.write_text(
        json.dumps(threshold_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def render_sidebar_navigation() -> str:
    """Render sidebar and return selected page key."""
    apply_sidebar_styles()

    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <p class="sidebar-title">Giám Sát Nhân Sự</p>
            <p class="sidebar-subtitle">Điều hướng chức năng và theo dõi camera</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<p class="sidebar-section-title">Điều Hướng</p>',
        unsafe_allow_html=True,
    )
    return st.sidebar.radio(
        "Menu",
        [
            "Đếm người từ ảnh",
            "Quản lý thư mục",
            "Cấu hình camera",
            "Lịch sử",
        ],
        label_visibility="collapsed",
    )


def render_sidebar_status() -> None:
    """Render sidebar status cards."""
    folder_settings = load_folder_settings()
    incoming_dir = Path(folder_settings["incoming_dir"])
    camera_count = len(load_camera_config())
    pending_count = count_pending_images(incoming_dir)

    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <p class="sidebar-section-title">Trạng Thái Nhanh</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.metric("Ảnh chờ xử lý", pending_count)
    st.sidebar.metric("Số camera đã cấu hình", camera_count)


def render_folder_management_page() -> None:
    """Render folder management UI page."""
    st.subheader("Quản Lý Thư Mục")
    st.caption("Cấu hình thư mục đầu vào và đầu ra cho quy trình đếm người.")

    folder_settings = load_folder_settings()

    tab_general, tab_pending, tab_guide = st.tabs(
        ["Thiết lập thư mục", "Ảnh chờ xử lý", "Hướng dẫn dịch vụ (Service Guide)"]
    )

    with tab_general:
        incoming_input = st.text_input(
            "Đường dẫn thư mục đầu vào (incoming)",
            value=folder_settings["incoming_dir"],
            key="incoming_dir_input",
        )
        processed_input = st.text_input(
            "Đường dẫn thư mục đầu ra (processed)",
            value=folder_settings["processed_dir"],
            key="processed_dir_input",
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Lưu thiết lập thư mục"):
                save_folder_settings(incoming_input, processed_input)
                st.success("Đã lưu thiết lập thư mục.")
        with col_b:
            if st.button("Khôi phục đường dẫn mặc định"):
                defaults = default_folder_settings()
                save_folder_settings(defaults["incoming_dir"], defaults["processed_dir"])
                st.success("Đã khôi phục thư mục mặc định theo vị trí ứng dụng.")
                st.rerun()
        with col_c:
            if st.button("Tạo thư mục nếu chưa có"):
                Path(incoming_input).mkdir(parents=True, exist_ok=True)
                Path(processed_input).mkdir(parents=True, exist_ok=True)
                st.success("Đã hoàn tất tạo thư mục.")

    with tab_pending:
        incoming_dir = Path(folder_settings["incoming_dir"])
        processed_dir = Path(folder_settings["processed_dir"])
        pending_count = count_pending_images(incoming_dir)
        pending_files = list_pending_images(incoming_dir=incoming_dir, limit=30)

        c1, c2, c3 = st.columns(3)
        c1.metric("Số ảnh chờ xử lý", pending_count)
        c2.metric("Thư mục Incoming tồn tại", "Có" if incoming_dir.exists() else "Không")
        c3.metric("Thư mục Processed tồn tại", "Có" if processed_dir.exists() else "Không")

        if pending_files:
            st.write("Danh sách ảnh chờ xử lý (tối đa 30 file):")
            st.code("\n".join(pending_files), language="text")
        else:
            st.info("Không có ảnh nào trong thư mục incoming.")

    with tab_guide:
        st.write("Quy ước đối với camera:")
        st.code("incoming/<camera_id>/<image>.jpg", language="text")
        st.write("Lệnh chạy service sử dụng tệp threshold được tạo từ UI:")
        st.code(
            (
                "python folder_service.py "
                "--incoming-dir incoming "
                "--processed-dir processed "
                "--thresholds-json camera_thresholds.generated.json"
            ),
            language="bash",
        )


def render_camera_config_page() -> None:
    """Render camera configuration UI page."""
    st.subheader("Cấu Hình Camera")
    st.caption("Quản lý `camera_id` và số người mục tiêu cho từng camera.")

    if "camera_rows" not in st.session_state:
        st.session_state["camera_rows"] = load_camera_config()

    tab_list, tab_add, tab_export = st.tabs(
        ["Danh sách camera", "Thêm nhanh", "Xuất cấu hình"]
    )

    with tab_list:
        edited_rows = st.data_editor(
            st.session_state["camera_rows"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "camera_id": st.column_config.TextColumn("Camera ID", required=True),
                "required_count": st.column_config.NumberColumn(
                    "Số người mục tiêu", min_value=0, step=1
                ),
                "active": st.column_config.CheckboxColumn("Đang hoạt động"),
                "area_name": st.column_config.TextColumn("Tên khu vực"),
            },
            key="camera_editor",
        )

        if st.button("Lưu danh sách camera"):
            normalized: list[dict[str, Any]] = []
            seen_ids: set[str] = set()

            for row in edited_rows:
                camera_id = str(row.get("camera_id", "")).strip()
                if not camera_id:
                    continue
                if camera_id in seen_ids:
                    st.error(f"Trùng camera_id: {camera_id}")
                    return
                seen_ids.add(camera_id)

                required_count = int(row.get("required_count", 0))
                normalized.append(
                    {
                        "camera_id": camera_id,
                        "required_count": max(required_count, 0),
                        "active": bool(row.get("active", True)),
                        "area_name": str(row.get("area_name", "")).strip(),
                    }
                )

            st.session_state["camera_rows"] = normalized
            save_camera_config(normalized)
            threshold_map = build_threshold_map(normalized)
            save_generated_thresholds(threshold_map)
            st.success("Đã lưu bảng cấu hình camera và cập nhật tệp threshold.")

    with tab_add:
        with st.form("add_camera_form"):
            camera_id = st.text_input("Camera ID mới")
            required_count = st.number_input(
                "Số người mục tiêu", min_value=0, value=0, step=1
            )
            area_name = st.text_input("Tên khu vực")
            active = st.checkbox("Kích hoạt camera", value=True)
            submitted = st.form_submit_button("Thêm camera")

        if submitted:
            camera_id_value = camera_id.strip()
            if not camera_id_value:
                st.error("Camera ID không được để trống.")
            else:
                existing_ids = {
                    str(row.get("camera_id", "")).strip()
                    for row in st.session_state["camera_rows"]
                }
                if camera_id_value in existing_ids:
                    st.error("Camera ID đã tồn tại.")
                else:
                    st.session_state["camera_rows"].append(
                        {
                            "camera_id": camera_id_value,
                            "required_count": int(required_count),
                            "active": active,
                            "area_name": area_name.strip(),
                        }
                    )
                    st.success("Đã thêm camera. Bấm 'Lưu danh sách camera' ở tab đầu tiên để ghi nhận vào hệ thống.")

    with tab_export:
        rows = st.session_state["camera_rows"]
        threshold_map = build_threshold_map(rows)

        st.write("Đường dẫn tệp cấu hình camera:")
        st.code(str(CAMERA_CONFIG_PATH), language="text")
        st.write("Đường dẫn tệp threshold được tạo ra cho dịch vụ xử lý:")
        st.code(str(GENERATED_THRESHOLDS_PATH), language="text")

        st.write("Nội dung threshold hiện tại:")
        st.code(
            json.dumps(threshold_map, indent=2, ensure_ascii=False),
            language="json",
        )

        if st.button("Xuất lại threshold ngay"):
            save_generated_thresholds(threshold_map)
            st.success("Đã xuất lại tệp camera_thresholds.generated.json.")


def _fetch_count_history(limit: int = 200) -> list[dict[str, Any]]:
    """Fetch count history rows from Supabase."""
    client = get_supabase()
    if not client:
        st.error("Chưa kết nối Supabase. Không thể tải lịch sử.")
        return []

    try:
        response = (
            client.table("person_count_results")
            .select(
                "id,processed_at,camera_id,source_file_name,person_count,required_count,status,archived_file_path"
            )
            .order("processed_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as e:
        st.error(f"Lỗi khi tải lịch sử từ Supabase: {e}")
        return []


def _list_processed_images(processed_dir: Path, limit: int = 200) -> list[Path]:
    """List processed image files from processed folder."""
    if not processed_dir.exists():
        return []

    files = [
        path
        for path in processed_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def _find_boxed_image(archived_path: Path) -> Path | None:
    """Find boxed image path from original archived image path.

    Args:
        archived_path: Original archived image path.

    Returns:
        Boxed image path if found, otherwise None.
    """
    annotated_dir = archived_path.parent / "annotated"
    if not annotated_dir.exists():
        return None

    exact = annotated_dir / f"{archived_path.stem}_boxed{archived_path.suffix}"
    if exact.exists():
        return exact

    candidates = sorted(
        annotated_dir.glob(f"{archived_path.stem}_boxed*{archived_path.suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def render_history_page() -> None:
    """Render history page to view results and processed images."""
    st.subheader("Lịch Sử Hệ Thống")
    st.caption("Tra cứu lịch sử kết quả đếm người và xem lại khung ảnh đã xử lý.")

    folder_settings = load_folder_settings()
    processed_dir = Path(folder_settings["processed_dir"])

    tab_results, tab_images = st.tabs(["Lịch sử thống kê", "Kho ảnh đã xử lý"])

    with tab_results:
        rows = _fetch_count_history(limit=500)
        if not rows:
            st.info("Chưa có dữ liệu lịch sử nào trên Database (person_count_results).")
        else:
            camera_options = sorted(
                {str(row.get("camera_id", "")) for row in rows if row.get("camera_id")}
            )
            selected_camera = st.selectbox(
                "Lọc theo Camera",
                options=["Tất cả"] + camera_options,
            )
            status_options = sorted(
                {str(row.get("status", "")) for row in rows if row.get("status")}
            )
            selected_statuses = st.multiselect(
                "Lọc theo Tình trạng (Status)",
                options=status_options,
                default=status_options,
                format_func=lambda x: {
                    "SHORTAGE": "Thiếu người (SHORTAGE)",
                    "SUFFICIENT": "Đủ mục tiêu (SUFFICIENT)",
                    "SURPLUS": "Thừa người (SURPLUS)"
                }.get(x, x),
            )

            filtered_rows = rows
            if selected_camera != "Tất cả":
                filtered_rows = [
                    row
                    for row in filtered_rows
                    if str(row.get("camera_id", "")) == selected_camera
                ]
            if selected_statuses:
                filtered_rows = [
                    row
                    for row in filtered_rows
                    if str(row.get("status", "")) in selected_statuses
                ]

            st.write(f"Số lượng bản ghi: {len(filtered_rows)}")
            # Hiển thị bảng tên cột tiếng Việt thay vì dùng tên cột Database gốc
            display_rows = []
            for r in filtered_rows:
                raw_stat = r.get("status", "")
                vn_stat = {
                    "SHORTAGE": "Thiếu người",
                    "SUFFICIENT": "Đủ mục tiêu",
                    "SURPLUS": "Thừa người"
                }.get(raw_stat, raw_stat)
                
                display_rows.append({
                    "Thời gian": r.get("processed_at"),
                    "Mã Camera": r.get("camera_id"),
                    "Tên ảnh nguồn": r.get("source_file_name"),
                    "Số người đếm được": r.get("person_count"),
                    "Mục tiêu (Yêu cầu)": r.get("required_count"),
                    "Tình trạng (Status)": vn_stat,
                })
            st.dataframe(display_rows, use_container_width=True, hide_index=True)

            if filtered_rows:
                preview_options = [
                    f"{row.get('processed_at', '')} | {row.get('camera_id', '')} | {row.get('source_file_name', '')}"
                    for row in filtered_rows
                ]
                preview_idx = st.selectbox(
                    "Chọn bản ghi để phục dựng ảnh bằng chứng",
                    options=list(range(len(preview_options))),
                    format_func=lambda i: preview_options[i],
                )
                selected_row = filtered_rows[preview_idx]
                archived_path = str(selected_row.get("archived_file_path", "")).strip()
                if archived_path and Path(archived_path).exists():
                    original_path = Path(archived_path)
                    boxed_path = _find_boxed_image(original_path)

                    col_original, col_boxed = st.columns(2)
                    with col_original:
                        st.image(
                            str(original_path),
                            caption=f"Ảnh camera gốc: {original_path.name}",
                            use_container_width=True,
                        )
                    with col_boxed:
                        if boxed_path and boxed_path.exists():
                            st.image(
                                str(boxed_path),
                                caption=f"Ảnh phân tích (có box): {boxed_path.name}",
                                use_container_width=True,
                            )
                        else:
                            st.info("Chưa tìm ra ảnh đã đóng hộp (box) cho bản ghi này.")
                else:
                    st.warning("Không tìm thấy file ảnh lưu trữ (có thể đường dẫn cục bộ đã thay đổi).")

    with tab_images:
        st.write(f"Đường dẫn lưu trữ tập trung (Processed Dir): {processed_dir}")
        files = _list_processed_images(processed_dir=processed_dir, limit=300)
        if not files:
            st.info("Kho lưu trữ trống. Chưa có ảnh nào được xử lý.")
        else:
            camera_filter_values = sorted(
                {
                    path.parts[-2] if len(path.parts) >= 2 else "unknown"
                    for path in files
                }
            )
            selected_camera = st.selectbox(
                "Lọc thư viện ảnh theo Camera",
                options=["Tất cả"] + camera_filter_values,
            )
            if selected_camera != "Tất cả":
                files = [
                    path
                    for path in files
                    if (path.parts[-2] if len(path.parts) >= 2 else "unknown")
                    == selected_camera
                ]

            st.write(f"Số lượng ảnh hiển thị: {len(files)}")
            columns = st.columns(4)
            for index, path in enumerate(files[:80]):
                col = columns[index % 4]
                with col:
                    st.image(str(path), caption=path.name, use_container_width=True)

