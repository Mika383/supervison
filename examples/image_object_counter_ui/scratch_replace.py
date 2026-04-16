import pathlib
p = pathlib.Path('C:/AHSO/Tool/supervision/examples/image_object_counter_ui/ui_components.py')
content = p.read_text(encoding='utf-8')

replacements = {
    'Chua ket noi Supabase. Khong the tai lich su.': 'Chưa kết nối Supabase. Không thể tải lịch sử.',
    'Loi khi tai lich su tu Supabase:': 'Lỗi khi tải lịch sử từ Supabase:',
    'st.subheader("Lich Su")': 'st.subheader("Lịch Sử Hệ Thống")',
    'Xem lich su ket qua dem nguoi va xem lai anh da xu ly.': 'Tra cứu lịch sử kết quả đếm người và xem lại khung ảnh đã xử lý.',
    'Lich su ket qua': 'Lịch sử thống kê',
    'Anh da xu ly': 'Kho ảnh đã xử lý',
    'Chua co du lieu lich su trong person_count_results.': 'Chưa có dữ liệu lịch sử nào trên Database (person_count_results).',
    'Loc theo camera': 'Lọc theo Camera',
    'Tat ca': 'Tất cả',
    'Loc theo trang thai': 'Lọc theo Tình trạng (Status)',
    'So ban ghi:': 'Số lượng bản ghi:',
    'Chon ban ghi de xem anh': 'Chọn bản ghi để phục dựng ảnh bằng chứng',
    'Anh goc:': 'Ảnh camera gốc:',
    'Anh co box:': 'Ảnh phân tích (có box):',
    'Chua tim thay anh co box cho ban ghi nay.': 'Chưa tìm ra ảnh đã đóng hộp (box) cho bản ghi này.',
    'Khong tim thay file anh trong archived_file_path.': 'Không tìm thấy file ảnh lưu trữ (có thể đường dẫn cục bộ đã thay đổi).',
    'Thu muc processed:': 'Đường dẫn lưu trữ tập trung (Processed Dir):',
    'Chua co anh da xu ly.': 'Kho lưu trữ trống. Chưa có ảnh nào được xử lý.',
    'Loc anh theo camera': 'Lọc thư viện ảnh theo Camera',
    'So anh hien thi:': 'Số lượng ảnh hiển thị:'
}

for k, v in replacements.items():
    content = content.replace(k, v)

p.write_text(content, encoding='utf-8')
