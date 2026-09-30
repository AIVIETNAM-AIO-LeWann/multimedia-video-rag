# Data contract

## Định danh chung

`video_id` có dạng `Lxx_Vxxx`. Khi artifact gắn với một frame,
`frame_uid = "{video_id}:{frame_idx}"`; `frame_idx` là chỉ số frame gốc từ 0 và
`timestamp_sec` tính từ đầu video. Không join chỉ bằng `sample_n` hoặc tên ảnh vì
chúng có thể đổi khi chạy lại keyframe extraction.

## Keyframe

Dataset `aqpahm/aic2026-keyframes-transnetv2`, đường dẫn `data/{level}/{video_id}/`:

- `keyframes.tar`: ảnh JPEG (tar không nén, đọc được theo offset);
- `frames.parquet`: identity, timestamp và provenance chọn frame;
- `_SUCCESS.json`: marker được ghi sau cùng.

Cột bắt buộc: `video_id`, `sample_n`, `shot_id`, `frame_idx`, `timestamp_sec`,
`image_path`. Keyframe chọn theo shot (TransNetV2 + khử trùng lặp): trung vị khoảng
274 keyframe/video, khoảng cách giữa hai keyframe liên tiếp trung vị 1,56 s, gần như
không vượt 2 s; khoảng 101 shot/video.

## Visual embedding

Hai dataset riêng, cùng cấu trúc `data/{level}/{video_id}/`:

| Model | Dataset | Chiều |
|---|---|---|
| `google/siglip2-so400m-patch16-384` | `aqpahm/aic2026-visual-siglip2-so400m` | 1152 |
| BEiT-3 Large patch16 384, COCO retrieval | `aqpahm/aic2026-visual-beit3-large-coco-retrieval` | 1024 |

- `embeddings.safetensors`: tensor `embeddings` FP16, đã L2-normalize;
- `frames.parquet`: cột identity và `embedding_row` theo đúng thứ tự vector;
- `_VISUAL_SUCCESS.json`: dimension, dtype, số dòng, model revision, source revision.

Cột identity bắt buộc: `video_id`, `frame_uid`, `sample_n`, `frame_idx`, `shot_id`,
`timestamp_sec`, `image_path`, `embedding_row`. Hai không gian vector không được trộn
hoặc so cosine trực tiếp với nhau. Câu truy vấn phải được encode bằng đúng model và
revision ghi trong marker.

## Caption (chỉ để hiển thị)

Dataset `aqpahm/aic2026-caption-blip2-opt-2.7b-coco`: `captions.parquet` và
`_CAPTION_SUCCESS.json`, mỗi keyframe đúng một caption tiếng Anh (`caption_en`) kèm
identity, `model_id` và `model_revision`. Trong phạm vi hiện tại caption chỉ được
hiện cạnh kết quả, không dùng để xếp hạng.

## Index retrieval (`indexes/<tên>/`, không commit)

| File | Nội dung |
|---|---|
| `siglip.faiss`, `beit3.faiss` | Inner product chính xác trên vector FP16, id = `frame_row` |
| `metadata.sqlite` | Bảng `frames` (identity, `shot_id`, timestamp, `image_path`) và `captions` |
| `manifest.json` | `index_schema_version` (hiện là 2), revision từng dataset, model/revision của encoder, số lượng |

`frame_row` là số thứ tự 0..N−1, sắp theo video rồi theo thời gian; mọi thành phần
dùng chung id này.

## Thay đổi schema

Ưu tiên thêm cột theo hướng tương thích. Tăng `schema_version` khi đổi tên, kiểu,
identity hoặc ý nghĩa dữ liệu. Reader phải từ chối major version tương lai thay vì tự
đoán. Luôn lưu input revision và model revision/checksum.
