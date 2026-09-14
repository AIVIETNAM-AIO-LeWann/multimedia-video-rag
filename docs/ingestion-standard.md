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
