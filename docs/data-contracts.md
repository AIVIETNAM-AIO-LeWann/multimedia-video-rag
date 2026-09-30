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

## OCR

Notebook PP-OCRv6 medium ghi `data/{level}/{video_id}/ocr.parquet`,
`ocr_regions.parquet` và `_OCR_SUCCESS.json` vào dataset riêng. Mỗi keyframe có
đúng một dòng trong `ocr.parquet` với identity, text có dấu, text chuẩn hóa,
text không dấu, trạng thái, số region, kích thước ảnh, confidence trung bình và
tối thiểu, model ID và revision.

Mỗi dòng `ocr_regions.parquet` chứa identity của frame, `region_id`,
`reading_order`, text gốc/chuẩn hóa, `confidence`, polygon chuẩn hóa trong
`polygon_json` và bounding box `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` chuẩn
hóa trong `[0,1]`. `region_count` của frame phải khớp số dòng region thực tế.
Frame không chữ có `has_text=False`, `ocr_status=no_text` và 0 region.

Pipeline dùng `PP-OCRv6_medium_det` để phát hiện vùng chữ và
`PP-OCRv6_medium_rec` để nhận dạng. Từ điển ký tự của recognizer (18.708 ký tự,
không có dấu tổ hợp) thiếu 44 chữ thường tiếng Việt, gồm toàn bộ dấu hỏi, dấu
nặng và phần lớn tổ hợp mũ + dấu (ví dụ `ả`, `ạ`, `ệ`, `ố`, `ự`). Decoder CTC
không thể sinh các ký tự này, nên notebook không phù hợp làm OCR tiếng Việt chính.
Đây là OCR chuyên dụng nên không có prompt, sampling, thinking hay giới hạn token.
Region có confidence dưới ngưỡng bị lọc. Lỗi inference làm video thất bại và
không được biến thành kết quả rỗng hợp lệ. Marker lưu input/model revision,
phiên bản PaddleOCR/PaddlePaddle, toàn bộ threshold/batch quan trọng, hash hai
Parquet, thống kê frame/region và schema version 6.

Checkpoint JSONL cục bộ có fingerprint của source, model, cấu hình và schema.
Nó được ghi liên tục, `fsync` mỗi 25 frame, chỉ dùng để tiếp tục một video chưa
upload trong cùng runtime storage, và bị xóa sau khi remote commit thành công.
Một uploader riêng commit từng video trong khi GPU tiếp tục inference; backlog
giới hạn ở 2 video và mỗi GPU chỉ prefetch thêm 1 video để bảo vệ RAM/ổ đĩa.
Nếu cùng một lỗi inference xuất hiện 3 lần liên tiếp, circuit breaker ngừng cấp
video mới và hủy các job chưa bắt đầu. Pilot/full chạy smoke test 5 frame trước
khi mở hàng đợi. Một pipeline độc lập được nạp trên mỗi GPU; Kaggle hai T4 xử lý
hai video song song. Chưa xác nhận đã chạy thành công trên dữ liệu thực tế.

### OCR · HunyuanOCR-1.5

Notebook `ingest-ocr-hunyuanocr.ipynb` ghi cùng ba file
`ocr.parquet`, `ocr_regions.parquet`, `_OCR_SUCCESS.json` vào dataset riêng
`aqpahm/aic2026-ocr-hunyuanocr`. Model là `tencent/HunyuanOCR` (VLM OCR ~1B),
chạy text spotting bằng prompt chính thức `spotting_hunyuan`; box trả về trong
`[0, 1000]` được đổi về `[0, 1]`.

`ocr.parquet` có đúng một dòng mỗi keyframe với identity, text gộp theo thứ tự
model trả về, text chuẩn hóa/không dấu, `has_text`, `ocr_status`, `region_count`,
`dropped_region_count`, kích thước ảnh, `spotting_format`, `finish_reason`,
`completion_tokens`, `raw_output` (output thô để parse lại không cần GPU), model
ID và revision. `ocr_regions.parquet` có một dòng mỗi dòng chữ với `region_id`,
`reading_order`, text gốc/chuẩn hóa/không dấu và bbox chuẩn hóa. VLM không cung
cấp confidence theo vùng nên contract này không có cột `confidence` và `polygon_json`.

`ocr_status` nhận một trong các giá trị:

| Giá trị | Ý nghĩa |
|---|---|
| `ok` | Có ít nhất một vùng chữ hợp lệ |
| `no_text` | Model trả rỗng hoặc mảng rỗng |
| `truncated` | Hết `MAX_NEW_TOKENS`; vùng chữ đã parse vẫn được giữ |
| `repetition` | Đuôi output lặp một đơn vị ngắn |
| `parse_error` | Có output nhưng không có vùng chữ hợp lệ |

Vùng chữ rỗng, box diện tích bằng 0 hoặc trùng lặp bị loại và được đếm trong
`dropped_region_count`. Lỗi HTTP hoặc server chết làm video thất bại, không bị
ghi thành `no_text`.

Inference dùng vLLM 0.18.1 (cấu hình CUDA 12 Tencent đã kiểm chứng), mỗi GPU
một server, dtype mặc định `float16` vì T4 không có bfloat16. Sampling theo
khuyến nghị chính thức: greedy, `repetition_penalty=1.08`. Marker lưu input/model
revision, phiên bản vLLM/torch/transformers, dtype, prompt, toàn bộ tham số
sampling, hash hai Parquet, phân bố `ocr_status`, thống kê và schema version 1.
Chưa xác nhận đã chạy thành công trên dữ liệu thực tế; chất lượng tiếng Việt và
tỉ lệ chữ bịa chưa được benchmark.

## Image captioning

Notebook BLIP-2 ghi `data/{level}/{video_id}/captions.parquet` và
`_CAPTION_SUCCESS.json` vào dataset riêng. Mỗi keyframe có đúng một
caption tiếng Anh; Parquet giữ `video_id`, `frame_uid`, `sample_n`,
`frame_idx`, `shot_id`, `timestamp_sec`, `image_path`, `caption_en`,
`caption_en_normalized`, `language`, `model_id` và `model_revision`.
Caption rỗng là lỗi, không được ghi marker thành công. Marker lưu
source/model revision, số frame/caption, generation settings, hash
Parquet và schema version. Caption không phải kết quả OCR. Dataset caption đã
được ingest hoàn tất; chất lượng retrieval của artifact chưa được benchmark.

## Thay đổi schema

Ưu tiên thêm cột theo hướng tương thích. Tăng `schema_version` khi đổi tên,
kiểu, identity hoặc ý nghĩa dữ liệu. Reader phải từ chối major version tương lai
thay vì tự đoán. Luôn lưu input revision và model revision/checksum.
