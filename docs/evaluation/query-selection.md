# Tiêu chí chọn query cho nghiên cứu truy vấn chỉ dùng embedding

Phạm vi nghiên cứu: tìm khoảnh khắc trong video bằng câu mô tả tiếng Anh, chỉ dùng
embedding ảnh (SigLIP 2, BEiT-3), không dùng OCR, ASR hay object detection.

Nguồn: bộ đề sơ tuyển AIC 2026 (SOTUYEN1–3), 91 câu. File zip SOTUYEN3 đặt tên
`query-p2-*`, nên mã trong bộ dữ liệu được đổi thành `p3-*` và giữ tên file gốc ở
cột `source_file`.

## Tiêu chí

Một câu được **giữ** khi khoảnh khắc cần tìm xác định được chỉ bằng nội dung nhìn
thấy trong khung hình: bối cảnh, người và trang phục, vật thể, màu sắc, số lượng, vị
trí, hành động, bố cục slide hay biểu đồ (không cần đọc chữ), và thứ tự các cảnh.

Một câu bị **loại** khi muốn xác định khoảnh khắc thì bắt buộc phải:

| Lý do | Mã |
|---|---|
| Đọc nội dung chữ trên màn hình (tên, số, bảng, câu trên bảng) | `read_text` |
| Nghe lời nói, hoặc ngữ nghĩa sự kiện chỉ có trong lời dẫn | `speech_or_text` |
| Dựa vào chuyển động máy quay, nội dung không đủ cụ thể | `camera_motion_underspecified` |
| Trùng nội dung với câu khác | `duplicate` |

Câu khó (màu sắc, đếm, cảnh ngắn) **không** bị loại. Chi tiết không kiểm chứng được
bằng hình (quốc tịch, địa danh, định lượng, âm thanh) bị **bỏ khỏi câu query tiếng
Anh** và ghi lại ở cột `notes`; câu vẫn được giữ nếu phần còn lại đủ xác định cảnh.

Với câu QA, chỉ phần mô tả khoảnh khắc được dùng để đánh giá truy vấn. Câu hỏi nằm
riêng ở cột `question_en`, và `answer_type` cho biết đáp án cần gì (`read_text`,
`count`, `visual_identify`, `speech_or_text`).

Việc lọc được làm **trước** khi chạy hệ thống và không dựa trên kết quả truy vấn.

## Kết quả

- Giữ 79/91 câu: 58 KIS, 16 QA, 5 TRAKE.
- Cấu trúc: 43 `sequence` (nhiều cảnh có thứ tự), 31 `single`, 5 `multi-event` (TRAKE).
- Thẻ: `color` 51, `count` 24, `spatial` 11, `graphic` 6, `order` 5, `identity` 1.
- Loại 12 câu: `read_text` 8, `speech_or_text` 2, `duplicate` 1,
  `camera_motion_underspecified` 1.

Danh sách chi tiết nằm ở `artifacts/eval/aic2026-visual-queries.csv` và
`artifacts/eval/aic2026-excluded-queries.csv` (không commit vì là dữ liệu đề của BTC).
Câu tiếng Anh được dịch và rút gọn thủ công, nên cần một người thứ hai rà lại trước
khi dùng làm bộ đánh giá chính thức.
