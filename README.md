# Multimedia Video RAG

Kho mã này hiện quản lý các job **ingest/extract offline** cho dữ liệu AIC 2026.
Các notebook đọc video hoặc keyframe từ Hugging Face, chạy model trên GPU của
Google Colab/Kaggle và ghi artifact trở lại Hugging Face.

## Trạng thái hiện tại

| Công đoạn | Model | Dataset output | Trạng thái |
|---|---|---|---|
| Keyframe extraction | TransNetV2 + pHash/OpenCLIP dedup | `aqpahm/aic2026-keyframes-transnetv2` | Đã chạy |
| ASR ingestion | ChunkFormer RNNT Large Vietnamese + Whisper fallback | `aqpahm/aic2026-asr-chunkformer-rnnt-large` | Đã chạy |
| Object detection | WeDetect Large, threshold 0.30 | `aqpahm/aic2026-od-wedetect-large` | Đã chạy |
| Visual embedding | SigLIP 2 So400m Patch16 384 | `aqpahm/aic2026-visual-siglip2-so400m` | Đang chạy |
| Visual embedding | BEiT-3 Large Patch16 384, COCO Retrieval | `aqpahm/aic2026-visual-beit3-large-coco-retrieval` | Sẵn sàng chạy |

Chưa có quyết định chính thức về kiến trúc retrieval, vector database, fusion,
reranking, API, giao diện, cách triển khai hay model cho các module tiếp theo.
Những nội dung đó sẽ được đánh giá riêng sau khi artifact ingestion ổn định.

## Cấu trúc

`notebooks/ingestion/` chứa notebook chuẩn, sạch output và có thể upload trực
tiếp lên Colab hoặc Kaggle. `notebooks/runs/` dành cho notebook đã chạy được
download từ nền tảng; thư mục này bị Git bỏ qua. `src/multimedia_video_rag/ingestion/`
chứa contract và công cụ kiểm tra artifact, không chứa logic retrieval.

## Quy trình chạy notebook

1. Upload notebook từ `notebooks/ingestion/` lên Colab hoặc Kaggle.
2. Bật GPU và Internet, thêm secret `HF_TOKEN` có quyền phù hợp.
3. Chạy pilot với giới hạn nhỏ nếu notebook hỗ trợ, kiểm tra artifact trên HF.
4. Bỏ giới hạn và chạy tiếp. Notebook tự bỏ qua video có marker hợp lệ.
5. Download notebook đã chạy vào `notebooks/runs/` nếu cần lưu log.
6. Chỉ cập nhật bản nguồn sạch trong `notebooks/ingestion/`.

## Kiểm tra cục bộ

Yêu cầu Python 3.12 và [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --extra dev
uv run python scripts/check_notebooks.py
uv run pytest
uv run ruff check .
```

Xem [quy ước ingestion](docs/ingestion-standard.md),
[data contract](docs/data-contracts.md) và [trạng thái dự án](docs/project-status.md).
