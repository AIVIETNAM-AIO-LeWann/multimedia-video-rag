# Repository working agreement

## Phạm vi hiện tại

Dự án nghiên cứu **text-to-video moment retrieval** trên dữ liệu AIC 2026, chỉ dựa trên
hai image embedding **SigLIP 2** và **BEiT-3** trích từ keyframe. Câu truy vấn là tiếng
Anh. Caption BLIP-2 chỉ được hiển thị cạnh kết quả, không tham gia chấm điểm.
OCR, ASR và object detection nằm ngoài phạm vi.

Package `src/multimedia_video_rag/retrieval` là mã thử nghiệm để dựng index, chạy
baseline và so sánh phương pháp. Không mô tả một backend, vector database, fusion
strategy hay serving architecture là quyết định đã chốt nếu chưa có đánh giá riêng.

## Quy tắc

- Notebook nguồn nằm trong `notebooks/`, phải tự chạy độc lập trên Colab hoặc Kaggle
  và không chứa output, execution count hay secret. Notebook đã chạy trên cloud không
  được đưa vào repository.
- Notebook ingest ghi rõ input/output repo, model ID, revision, tham số quan trọng và
  schema version trong marker; chỉ ghi marker sau khi mọi artifact của video đã upload.
- Resume dựa trên marker hợp lệ; không suy đoán video còn thiếu theo dãy số ID.
- Không đổi thuật toán/model hoặc schema chỉ để đồng bộ hình thức notebook.
- Không commit token, video, model weight, keyframe, embedding, index, cache hoặc dữ liệu
  câu hỏi của BTC.
- Index chỉ được build từ video có marker hợp lệ, phải khớp đúng tập `frame_uid` của
  keyframe, và chỉ được publish sau khi build xong toàn bộ.
- Mọi so sánh phương pháp phải ghi lại cấu hình, phiên bản câu truy vấn (ai viết lại,
  bằng cách nào) và revision của index để tái lập được.

## Kiểm tra

```text
uv run python scripts/check_notebooks.py
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
