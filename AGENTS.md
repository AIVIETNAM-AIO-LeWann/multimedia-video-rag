# Repository working agreement

## Phạm vi hiện tại

Kho mã chỉ quản lý ingest/extract offline và contract artifact. Không mô tả một
backend, vector database, fusion strategy hay serving architecture là quyết định
đã chốt nếu chưa có đánh giá và xác nhận riêng.

## Quy tắc

- Notebook nguồn nằm trong `notebooks/ingestion`, phải tự chạy độc lập trên
  Colab hoặc Kaggle và không chứa output, execution count hay secret.
- Notebook đã chạy nằm trong `notebooks/runs` và không commit.
- Mọi job ghi rõ input/output repo, model ID, revision, tham số quan trọng và
  schema version trong marker.
- Chỉ ghi marker thành công sau khi mọi artifact của video đã upload xong.
- Resume dựa trên marker hợp lệ; không suy đoán video còn thiếu theo dãy số ID.
- Audit completion theo đúng thành viên thực tế do BTC cung cấp.
- Không đổi thuật toán/model hoặc schema chỉ để đồng bộ hình thức notebook.
- Không commit token, video, model weight, keyframe, embedding hoặc cache.

## Kiểm tra

```text
uv run python scripts/check_notebooks.py
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
