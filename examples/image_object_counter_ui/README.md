# image object counter

## overview

This example now includes two modes:

- `app.py`: Streamlit UI for manual image upload and inspection.
- `folder_service.py`: Production-style folder watcher that counts only `person`, archives processed images, and writes results to Supabase.

## install

From repository root:

```bash
cd examples/image_object_counter_ui
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## run ui mode

```bash
python -m streamlit run app.py
```

### ui pages (tieng viet)

UI da duoc tach thanh cac thanh chuc nang (component bars) voi sidebar dieu huong:

- `Dem nguoi tu anh`:
  - Cau hinh model/nguong.
  - Tai anh, dem class `person`, xem ket qua annotate.
- `Quan ly thu muc`:
  - Xem/chinh path `incoming` va `processed`.
  - Auto-bind folder mac dinh.
  - Tao folder neu thieu.
  - Xem so anh pending va danh sach anh cho xu ly.
- `Cau hinh camera`:
  - Quan ly `camera_id`, `required_count`, trang thai active, ten khu vuc.
  - Them nhanh camera.
  - Xuat file `camera_thresholds.generated.json` cho folder service.
- `Lich su`:
  - Xem lich su ket qua dem nguoi tu bang `person_count_results`.
  - Loc theo camera va trang thai.
  - Xem lai anh da xu ly tu thu muc `processed`.

## run folder service mode (supabase)

1. Create required Supabase tables by running SQL in [supabase_schema.sql](./supabase_schema.sql).
2. Prepare incoming folder structure by camera:

```text
incoming/
  camera-a/
    frame_001.jpg
  camera-b/
    frame_101.jpg
```

3. Set Supabase credentials:

```powershell
$env:SUPABASE_URL = "https://<project-ref>.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY = "<service-role-key>"
```

4. Start service:

```bash
python folder_service.py \
  --incoming-dir incoming \
  --processed-dir processed \
  --weights-path yolo11n.pt \
  --camera-targets-table camera_staffing_targets \
  --results-table person_count_results \
  --alerts-table staffing_alerts
```

## behavior of folder service

- Counts only `person` class from YOLO results.
- Reads every image from `incoming/<camera_id>/...`.
- Normalizes image filename to `cameraid_YYYYMMDD_HHMMSS.ext` during processing.
- Moves processed image to `processed/YYYY-MM-DD/<camera_id>/...`.
- Saves an annotated copy with person boxes to `processed/YYYY-MM-DD/<camera_id>/annotated/*_boxed.ext`.
- Inserts one row per image into `person_count_results`.
- If `person_count < required_count`, inserts alert row into `staffing_alerts`.
- Camera target can be changed in DB table `camera_staffing_targets` and the service picks it up on next poll cycle.
- Uses an in-memory FIFO queue and processes images strictly one-by-one when many images arrive at the same time.
- Prints queue logs: `queue enqueued=... size=...`, `processed ...`, and `queue idle no-pending-images`.

## optional local threshold config

Instead of reading camera targets from Supabase table, you can pass local JSON:

```bash
python folder_service.py \
  --incoming-dir incoming \
  --processed-dir processed \
  --thresholds-json camera_thresholds.example.json
```

## notes

- Default model path is `yolo11n.pt`.
- First run can download model weights.
- Use service role key for server-side writer process.
