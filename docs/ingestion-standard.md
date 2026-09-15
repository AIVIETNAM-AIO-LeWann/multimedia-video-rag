# Quy ước notebook ingest/extract

## Cấu trúc notebook

Mỗi notebook có các phần theo thứ tự: mô tả job, cài dependency, cấu hình và
runtime, tải model, hàm xử lý, vòng chạy resumable, audit cuối. Các hằng số cần
đổi khi vận hành phải tập trung trong cell cấu hình.

## Runtime và secret

Notebook nhận diện `colab`, `kaggle` hoặc `local`. Token được đọc theo thứ
tự: biến môi trường `HF_TOKEN`, Colab Secret, Kaggle Secret. Không in token ra
log. Job cần GPU phải dừng sớm với thông báo rõ nếu CUDA không có.

Thư mục tạm:

- Colab: `/content/<job-name>`;
- Kaggle: `/kaggle/temp/<job-name>`;
- local: thư mục temp của hệ điều hành.

## Tính lặp lại và resume

- Ghim input revision và model revision/checksum trong mỗi lần chạy.
- Artifact được tạo trong thư mục riêng của từng video.
- Upload artifact trước, marker thành công sau cùng trong cùng commit nếu API hỗ trợ.
- Chỉ skip khi marker tồn tại, đúng schema/model/source và đủ file bắt buộc.
- Retry network bằng exponential backoff; commit theo nhóm video để tránh rate limit.
- Dọn file tạm sau từng video/archive để không đầy disk phiên chạy.

## Notebook nguồn và notebook đã chạy

Notebook trong `notebooks/ingestion` không chứa output. Notebook download sau
khi chạy được đặt trong `notebooks/runs`; tên gợi ý:
`YYYYMMDD-platform-job-scope.ipynb`. Log quan trọng nên được tóm tắt trong issue
hoặc run note thay vì đưa toàn bộ output vào Git.


## Sử dụng GPU

Mỗi notebook tự lấy toàn bộ GPU CUDA khả dụng bằng `torch.cuda.device_count()`.
Colab T4 tạo một worker; Kaggle hai T4 tạo hai worker và hai model replica. Mỗi
worker giữ độc quyền một GPU trong suốt một video/archive. Batch được đặt theo
VRAM và tự giảm khi CUDA OOM. Upload được gom theo nhóm và điều phối ngoài worker
GPU để tránh xung đột commit.

GPU có thể tạm giảm utilization trong lúc tải archive, giải mã video, đọc TAR,
ghi Parquet hoặc upload mạng; không có notebook nào bảo đảm 100% GPU liên tục.
Mục tiêu là không để GPU thứ hai bị bỏ trống khi còn công việc inference độc lập.
