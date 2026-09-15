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
| BEiT-3 Large COCO Retrieval | `aqpahm/aic2026-visual-beit3-large-coco-retrieval` | Private | Đang ingest |

Việc hoàn tất luôn được đối chiếu với video thực sự có trong archive của BTC;
ID bị khuyết trong dãy số không được xem là dữ liệu thiếu.

## Chưa quyết định

- kiến trúc hệ thống online và ranh giới dịch vụ;
- vector database/search engine và metadata store;
- cách kết hợp các visual embedding;
- query rewriting, fusion, filtering và reranking;
- model OCR, caption và các module chưa ingest;
- hạ tầng host/deploy.

Mọi phương án về các mục trên hiện chỉ là giả thuyết cần benchmark.
