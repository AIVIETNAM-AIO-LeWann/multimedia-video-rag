# Repository working agreement

## Phạm vi hiện tại

Kho mã quản lý ingest/extract offline, contract artifact và package thử nghiệm
`src/multimedia_video_rag/retrieval` dùng để dựng index và benchmark retrieval.
Index/fusion trong package này chỉ là cấu hình thử nghiệm. Không mô tả một
backend, vector database, fusion strategy hay serving architecture là quyết định
đã chốt nếu chưa có đánh giá và xác nhận riêng.

## Quy tắc

- Notebook nguồn nằm trong `notebooks/ingestion`, phải tự chạy độc lập trên
  Colab hoặc Kaggle và không chứa output, execution count hay secret.
- Notebook đã chạy trên Colab/Kaggle không được đưa vào repository.
- Mọi job ghi rõ input/output repo, model ID, revision, tham số quan trọng và
  schema version trong marker.
- Chỉ ghi marker thành công sau khi mọi artifact của video đã upload xong.
- Resume dựa trên marker hợp lệ; không suy đoán video còn thiếu theo dãy số ID.
- Audit completion theo đúng thành viên thực tế do BTC cung cấp.
- Không đổi thuật toán/model hoặc schema chỉ để đồng bộ hình thức notebook.
- Không commit token, video, model weight, keyframe, embedding, index hoặc cache.
- Index retrieval chỉ được build từ video có marker hợp lệ và phải khớp đúng tập
  `frame_uid` của keyframe; index chỉ được publish sau khi build xong toàn bộ.

## Kiểm tra

```text
uv run python scripts/check_notebooks.py
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
