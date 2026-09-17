# Data contract của artifact ingestion

## Định danh chung

`video_id` có dạng `Lxx_Vxxx`. Khi artifact gắn với một frame,
`frame_uid = "{video_id}:{frame_idx}"`; `frame_idx` là chỉ số frame gốc từ 0
và `timestamp_sec` tính từ đầu video. Không join chỉ bằng `sample_n` hoặc tên
ảnh vì chúng có thể đổi khi chạy lại keyframe extraction.

## Keyframe

Đường dẫn `data/{level}/{video_id}/` chứa:

- `keyframes.tar`: ảnh JPEG;
- `frames.parquet`: identity, timestamp và provenance chọn frame;
- `_SUCCESS.json`: marker được ghi sau cùng.

Các cột bắt buộc của Parquet: `video_id`, `sample_n`, `shot_id`,
`frame_idx`, `timestamp_sec`, `image_path`.

## ASR

Đường dẫn `data/{level}/{video_id}/` chứa `asr.parquet`,
`asr_chunks.parquet` và `_ASR_SUCCESS.json`. ASR gắn với khoảng thời gian của
video, không giả lập segment thành frame. Kết quả rỗng chỉ hợp lệ nếu marker ghi
rõ trạng thái không phát hiện lời nói.

Các cột segment bắt buộc: `video_id`, `segment_id`, `start_sec`, `end_sec`,
`timestamp_sec`, `raw_text`, `normalized_text`, `normalized_no_accent`,
`language`, `model_id`.

## Object detection

Đường dẫn `data/{level}/{video_id}/` chứa `detections.parquet`,
`frames.parquet` và `_OD_SUCCESS.json`. Một frame có `status = empty_valid`
là inference hợp lệ với 0 detection, không phải dữ liệu thiếu.

Detection gồm identity, label đa ngôn ngữ, confidence, bounding box chuẩn hóa và
`box_area_ratio`. Marker lưu model/checkpoint, vocabulary hash, threshold và
source revision.

## Visual embedding

Đường dẫn `data/{level}/{video_id}/` chứa:

- `embeddings.safetensors`: tensor `embeddings` FP16, đã L2-normalize;
- `frames.parquet`: các cột identity và `embedding_row` theo đúng thứ tự vector;
- `_VISUAL_SUCCESS.json`: dimension, dtype, số dòng, model revision và source revision.

Các cột identity bắt buộc: `video_id`, `frame_uid`, `sample_n`,
`frame_idx`, `shot_id`, `timestamp_sec`, `image_path`, `embedding_row`.
Artifact này chưa quy định loại index hoặc cách dùng vector khi retrieval.
Mỗi model visual ghi vào một dataset riêng. SigLIP 2 dùng vector 1152 chiều;
BEiT-3 Large COCO Retrieval dùng vector 1024 chiều. Hai không gian vector không
được trộn hoặc so cosine trực tiếp với nhau.

## Image captioning

Notebook BLIP-2 ghi `data/{level}/{video_id}/captions.parquet` và
`_CAPTION_SUCCESS.json` vào dataset riêng. Mỗi keyframe có đúng một
caption tiếng Anh; Parquet giữ `video_id`, `frame_uid`, `sample_n`,
`frame_idx`, `shot_id`, `timestamp_sec`, `image_path`, `caption_en`,
`caption_en_normalized`, `language`, `model_id` và `model_revision`.
Caption rỗng là lỗi, không được ghi marker thành công. Marker lưu
source/model revision, số frame/caption, generation settings, hash
Parquet và schema version. Caption không phải kết quả OCR và hiện
chưa được ingest hoặc đánh giá trên toàn bộ dữ liệu.

## Thay đổi schema

Ưu tiên thêm cột theo hướng tương thích. Tăng `schema_version` khi đổi tên,
kiểu, identity hoặc ý nghĩa dữ liệu. Reader phải từ chối major version tương lai
thay vì tự đoán. Luôn lưu input revision và model revision/checksum.
