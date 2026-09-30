# Trạng thái dự án

## Giai đoạn hiện tại

Dự án đang tạo và kiểm tra artifact offline theo từng module. Công việc hiện tại
không bao gồm thiết kế hay triển khai hệ thống retrieval hoàn chỉnh.

## Artifact đang có

| Module | Dataset Hugging Face | Quyền xem | Tình trạng |
|---|---|---|---|
| Keyframe | `aqpahm/aic2026-keyframes-transnetv2` | Public | Đã extract L21–L30 |
| ASR | `aqpahm/aic2026-asr-chunkformer-rnnt-large` | Private | Đã ingest; có Whisper fallback cho ca rỗng |
| OD | `aqpahm/aic2026-od-wedetect-large` | Private | Đã ingest |
| SigLIP | `aqpahm/aic2026-visual-siglip2-so400m` | Private | Đã ingest hoàn tất |
| BEiT-3 Large COCO Retrieval | `aqpahm/aic2026-visual-beit3-large-coco-retrieval` | Private | Đã ingest hoàn tất |
| Image captioning | `aqpahm/aic2026-caption-blip2-opt-2.7b-coco` | Private | Đã ingest hoàn tất |

Việc hoàn tất luôn được đối chiếu với video thực sự có trong archive của BTC;
ID bị khuyết trong dãy số không được xem là dữ liệu thiếu.

`scripts/audit_hf_ingestion.py` kiểm tra sáu repository keyframe, hai visual,
OD, ASR và caption. Công cụ lấy inventory keyframe thực tế làm nguồn chuẩn, xác
thực marker/artifact/Parquet và xuất báo cáo cục bộ trong `artifacts/`.

Quick audit ngày 2026-09-29 ghi nhận cả sáu repository đều có đúng 873 video,
không có video thừa hoặc đường dẫn dữ liệu sai cấu trúc. Audit sâu `L21_V001`
cho thấy SigLIP, BEiT-3, OD và caption đều khớp 937 keyframe; ASR có 100 segment.
Marker `_SUCCESS.json` của keyframe này là bản legacy và thiếu `schema_version`,
nên chưa đạt contract marker hiện tại dù artifact và các module dẫn xuất khớp.

## Notebook đang chuẩn bị

Notebook `ingest-ocr-ppocrv6-medium.ipynb` đã được viết để chạy preflight
PP-OCRv6 medium trên 5 keyframe trước khi pilot 5 video. Notebook tạo text tổng
hợp theo frame và từng vùng chữ có polygon, bounding box và confidence; mỗi GPU
giữ một pipeline, checkpoint định kỳ và upload song song với inference.
Dataset `aqpahm/aic2026-ocr-ppocrv6-medium` là cấu hình dự kiến;
chưa có artifact OCR hợp lệ. Recognizer PP-OCRv6 thiếu 44 chữ thường tiếng Việt
(dấu hỏi, dấu nặng, phần lớn tổ hợp mũ + dấu) nên được giữ lại chỉ để đối chiếu.

Notebook `ingest-ocr-hunyuanocr.ipynb` là hướng OCR chính đang chuẩn bị: HunyuanOCR-1.5
text spotting qua vLLM 0.18.1, mỗi GPU một server, dtype `float16`, sampling greedy
với `repetition_penalty=1.08`. Output thô được lưu cùng trạng thái
`truncated`/`repetition`/`parse_error` để đo tỉ lệ thoái hóa. Notebook chưa được
chạy trên GPU; tương thích vLLM 0.18.1 với T4/float16, tốc độ trên 335.477
keyframe, chất lượng tiếng Việt và tỉ lệ chữ bịa đều chưa được xác nhận.
Dataset dự kiến: `aqpahm/aic2026-ocr-hunyuanocr`.

## Retrieval thử nghiệm

- Index `indexes/dev` (không commit) dựng từ 873 video, 335.477 keyframe: FAISS
  inner product chính xác cho SigLIP 2 và BEiT-3, SQLite cho caption/ASR/OD.
- Đã tái hiện hai baseline: arXiv:2504.08384 (max-norm ensemble, gộp điểm lân cận,
  dual-query temporal search) và GRAB arXiv:2504.09298 (SuperGlobal, ABTS).
- Bộ 79 query tiếng Anh chọn lọc từ đề sơ tuyển (58 KIS, 16 QA, 5 TRAKE) nằm ở
  `artifacts/eval/`; **chưa có ground truth**, nên chưa có số liệu Recall/MRR.
- Web UI cục bộ (`scripts/serve_ui.py`) tìm bằng SigLIP 2 + BEiT-3 với RRF hoặc
  max-norm, encode query qua notebook server trên Colab, và ghi nhãn đáp án vào
  `artifacts/labels/labels.csv` để phục vụ gán ground truth.

## Chưa quyết định

- kiến trúc hệ thống online và ranh giới dịch vụ;
- vector database/search engine và metadata store;
- cách kết hợp các visual embedding;
- query rewriting, fusion, filtering và reranking;
- đánh giá chất lượng artifact đã ingest và các module còn thiếu;
- hạ tầng host/deploy.

Mọi phương án về các mục trên hiện chỉ là giả thuyết cần benchmark.
