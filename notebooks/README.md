# Hướng dẫn notebook ingest/extract

Thư mục [`ingestion`](ingestion/) chứa các notebook nguồn dùng để tạo artifact
offline trên Google Colab hoặc Kaggle. Mỗi notebook tự cài dependency, nhận diện
runtime, đọc secret, tải input từ Hugging Face, chạy inference, kiểm tra artifact
và upload kết quả theo lô.

Các notebook này chưa xây chỉ mục tìm kiếm và chưa thể hiện một kiến trúc video
retrieval đã chốt. Bản notebook tải xuống sau khi chạy cloud chỉ dùng làm log cá
nhân và được giữ ngoài repository.

## Danh mục notebook

| Notebook | Input | Model/phương pháp | Output | Artifact mỗi video |
|---|---|---|---|---|
| [`extract-kf-transnetv2.ipynb`](ingestion/extract-kf-transnetv2.ipynb) | `oh-i-ace/aic-videos` | TransNetV2 + pHash/OpenCLIP dedup | `aqpahm/aic2026-keyframes-transnetv2` | `keyframes.tar`, `frames.parquet`, `_SUCCESS.json` |
| [`ingest-asr-chunkformer-rnnt-large.ipynb`](ingestion/ingest-asr-chunkformer-rnnt-large.ipynb) | `oh-i-ace/aic-videos` | ChunkFormer RNNT Large Vietnamese + Whisper Large v3 fallback | `aqpahm/aic2026-asr-chunkformer-rnnt-large` | `asr.parquet`, `asr_chunks.parquet`, `_ASR_SUCCESS.json` |
| [`ingest-od-wedetect-large.ipynb`](ingestion/ingest-od-wedetect-large.ipynb) | Dataset keyframe | WeDetect Large, vocabulary AIC 400 nhãn | `aqpahm/aic2026-od-wedetect-large` | `detections.parquet`, `frames.parquet`, `_OD_SUCCESS.json` |
| [`ingest-visual-siglip2-so400m.ipynb`](ingestion/ingest-visual-siglip2-so400m.ipynb) | Dataset keyframe | SigLIP 2 So400m Patch16 384 | `aqpahm/aic2026-visual-siglip2-so400m` | `embeddings.safetensors`, `frames.parquet`, `_VISUAL_SUCCESS.json` |
| [`ingest-visual-beit3-large-coco-retrieval.ipynb`](ingestion/ingest-visual-beit3-large-coco-retrieval.ipynb) | Dataset keyframe | BEiT-3 Large Patch16 384, COCO Retrieval | `aqpahm/aic2026-visual-beit3-large-coco-retrieval` | `embeddings.safetensors`, `frames.parquet`, `_VISUAL_SUCCESS.json` |
| [`ingest-caption-blip2-opt-2.7b-coco.ipynb`](ingestion/ingest-caption-blip2-opt-2.7b-coco.ipynb) | Dataset keyframe | BLIP-2 OPT 2.7B fine-tuned on COCO | `aqpahm/aic2026-caption-blip2-opt-2.7b-coco` | `captions.parquet`, `_CAPTION_SUCCESS.json` |
| [`ingest-ocr-ppocrv6-medium.ipynb`](ingestion/ingest-ocr-ppocrv6-medium.ipynb) | Dataset keyframe | PP-OCRv6 medium detector + recognizer | `aqpahm/aic2026-ocr-ppocrv6-medium` (dự kiến) | `ocr.parquet`, `ocr_regions.parquet`, `_OCR_SUCCESS.json` |
| [`ingest-ocr-hunyuanocr.ipynb`](ingestion/ingest-ocr-hunyuanocr.ipynb) | Dataset keyframe | HunyuanOCR-1.5 text spotting qua vLLM 0.18.1 | `aqpahm/aic2026-ocr-hunyuanocr` (dự kiến) | `ocr.parquet`, `ocr_regions.parquet`, `_OCR_SUCCESS.json` |

## Thứ tự phụ thuộc

```mermaid
flowchart TD
    V[Video BTC]
    K[1. Extract keyframe]
    A[ASR độc lập]
    O[2a. Object detection]
    S[2b. SigLIP 2 embedding]
    B[2c. BEiT-3 embedding]
    C[2d. BLIP-2 captioning]
    OCR[2e. OCR preflight/pilot<br/>HunyuanOCR-1.5 hoặc PP-OCRv6]

    V --> K
    V --> A
    K --> O
    K --> S
    K --> B
    K --> C
    K --> OCR
```

- Keyframe phải hoàn tất trước OD, SigLIP 2 và BEiT-3.
- ASR đọc video gốc nên có thể chạy độc lập hoặc song song với keyframe.
- OD, SigLIP 2, BEiT-3, BLIP-2 captioning và OCR không phụ thuộc lẫn nhau.
- Hai visual model phải ghi vào hai dataset riêng vì số chiều và không gian
  embedding khác nhau.

## Chuẩn bị runtime

### Google Colab

1. Tải notebook nguồn lên Colab.
2. Chọn **Runtime → Change runtime type → T4 GPU**.
3. Bật Internet nếu tài khoản/runtime yêu cầu.
4. Trong **Secrets**, thêm `HF_TOKEN` và bật quyền truy cập notebook.
5. Chạy từ cell đầu tiên theo đúng thứ tự.

### Kaggle

1. Import notebook nguồn vào Kaggle Notebook.
2. Trong **Session options**, bật GPU và Internet.
3. Chọn hai T4 nếu Kaggle cung cấp tùy chọn đó.
4. Trong **Add-ons → Secrets**, thêm `HF_TOKEN` và cấp quyền sử dụng.
5. Chạy từ cell đầu tiên theo đúng thứ tự.

Token cần quyền đọc repository input. Những job ghi vào dataset private cần thêm
quyền ghi đúng repository output. Không dán token trực tiếp vào cell, log hoặc
file notebook.

## Cấu hình trước khi chạy

Mỗi notebook có một cell cấu hình tập trung. Tên biến có thể khác nhẹ theo job,
nhưng cần kiểm tra các nhóm sau:

| Nhóm cấu hình | Ý nghĩa |
|---|---|
| Input repository/revision | Nguồn dữ liệu và revision dùng cho lần chạy |
| Output repository | Dataset nhận artifact |
| Model ID/revision/checksum | Model và phiên bản có thể tái lập |
| Archive selection | Danh sách archive thực tế cần xử lý |
| Pilot limit | Giới hạn số video để kiểm tra trước khi chạy toàn bộ |
| Batch/upload size | Kích thước inference batch và số video mỗi commit |
| Threshold | Ngưỡng của thuật toán, nếu notebook có sử dụng |
| Strict validation | Mức kiểm tra artifact từ xa khi resume |

Không đổi tên output repository giữa chừng nếu muốn notebook nhận ra artifact đã
có. Khi thay model, source revision, schema hoặc tham số ảnh hưởng kết quả, marker
cũ có thể không còn hợp lệ và cần một dataset/version mới.

## Pilot, chạy toàn bộ và resume

Quy trình vận hành đề xuất:

1. Đặt giới hạn pilot từ 1–5 video nếu notebook hỗ trợ.
2. Chạy notebook và mở một vài Parquet/marker trên Hugging Face để kiểm tra.
3. Xác nhận số dòng, timestamp, model ID, source revision và file bắt buộc.
4. Bỏ giới hạn pilot (`None`) rồi chạy lại từ đầu notebook.
5. Notebook kiểm tra output từ xa và skip video có marker hợp lệ.
6. Nếu runtime ngắt, tạo phiên mới và chạy lại toàn bộ cell; job tiếp tục từ
   video chưa hoàn tất.
7. Đọc báo cáo audit cuối trước khi kết luận job đã xong.

Không dựa vào việc thư mục xuất hiện trên Hugging Face để kết luận thành công.
Một thư mục chỉ hợp lệ khi marker và toàn bộ artifact bắt buộc khớp contract.

## Sử dụng GPU

Các notebook tự lấy số GPU bằng `torch.cuda.device_count()`:

- Colab một T4: một worker và một model replica;
- Kaggle hai T4: hai worker độc lập, mỗi worker giữ một GPU;
- batch khởi tạo theo cấu hình/VRAM và tự giảm khi CUDA OOM;
- upload được gom theo nhóm và thực hiện ngoài GPU worker để tránh nhiều worker
  cùng commit lên một repository.

GPU utilization không phải lúc nào cũng đạt 100%. Các bước tải archive, giải mã
video, giải nén TAR, ghi Parquet, nén artifact và upload chủ yếu dùng CPU, disk
hoặc mạng. Với một GPU, các bước này có thể trở thành nút thắt dù CUDA hoạt động
bình thường.

## Chi tiết từng notebook

### Keyframe — TransNetV2

- Xử lý 14 archive từ L21 đến L30, trong đó L26 gồm các phần `a`–`e`.
- Phát hiện ranh giới shot bằng TransNetV2 với `CUT_THRESHOLD = 0.50`.
- Loại frame gần trùng bằng pHash (`PHASH_DISTANCE = 6`) và OpenCLIP ViT-B/32
  (`COSINE_THRESHOLD = 0.965`).
- Upload theo nhóm 3 video để giảm số commit lên Hugging Face.
- Audit theo thành viên thật trong ZIP; không yêu cầu dãy `Vxxx` phải liên tục.

### ASR — ChunkFormer RNNT Large Vietnamese

- Model chính: `khanhld/chunkformer-rnnt-large-vie`.
- Fallback cho ca model chính không tạo được transcript:
  `Systran/faster-whisper-large-v3`.
- Kết quả giữ cả text gốc, text chuẩn hóa có dấu và text không dấu.
- Segment gắn với `start_sec`, `end_sec` và `timestamp_sec` trên video gốc.
- Video thật sự không có lời nói được biểu diễn rõ trong marker; không tạo một
  transcript giả để vượt qua audit.

### Object detection — WeDetect Large

- Model: WeDetect Large từ repository `fushh7/WeDetect`.
- Kích thước ảnh suy luận: 1280.
- Confidence threshold: `0.30`.
- Core vocabulary: `aic-query-2025-2026-v1-400`, gồm 400 khái niệm với nhãn
  tiếng Anh, Việt và Trung phục vụ đối chiếu sau này.
- Batch khởi tạo là 2 và tự chia đôi khi gặp CUDA OOM.
- Frame có 0 detection hợp lệ được ghi `empty_valid`, không bị xem là lỗi.

### Visual embedding — SigLIP 2 So400m

- Model: `google/siglip2-so400m-patch16-384`.
- Batch khởi tạo: 16; tự giảm khi CUDA OOM.
- Vector đầu ra: 1152 chiều, FP16 và đã L2-normalize.
- Thứ tự tensor được ánh xạ chính xác bằng cột `embedding_row` trong
  `frames.parquet`.

### Visual embedding — BEiT-3 Large COCO Retrieval

- Model định danh: `microsoft/beit3-large-patch16-384-coco-retrieval`.
- Tải source UniLM và checkpoint chính thức; notebook có bootstrap TorchScale
  cho môi trường Colab/Kaggle.
- Batch khởi tạo: 16; tự giảm khi CUDA OOM.
- Vector đầu ra: 1024 chiều, FP16 và đã L2-normalize.
- Có smoke test trên từng GPU trước khi bắt đầu toàn bộ workload.

### Image captioning — BLIP-2 OPT 2.7B COCO

- Model: `Salesforce/blip2-opt-2.7b-coco`; caption tiếng Anh cho từng keyframe.
- Chế độ mặc định là pilot 5 video; cần xem nội dung caption rồi mới đặt
  `MAX_VIDEOS_PER_RUN = None` để chạy toàn bộ.
- Một model FP16 trên mỗi GPU T4; batch khởi tạo 2 và tự giảm khi OOM.
- Caption có thể bỏ sót chữ nhỏ, tên riêng và thông tin chuyển động;
  không thay thế OCR hoặc ASR.
- Artifact/marker: `captions.parquet`, `_CAPTION_SUCCESS.json`.
- Dataset output đã được ingest hoàn tất.

### OCR — PP-OCRv6 medium

- `ocr.parquet` có một dòng cho mỗi keyframe; frame không có chữ vẫn có
  `has_text=False`. `ocr_regions.parquet` có một dòng cho mỗi dòng/vùng chữ,
  kèm bounding box `xyxy` chuẩn hóa `[0,1]` và thứ tự đọc.
- Dùng detector `PP-OCRv6_medium_det` và recognizer
  `PP-OCRv6_medium_rec`. Text tổng hợp giữ nguyên dấu và xuống dòng; bản chuẩn
  hóa không dấu được lưu riêng để tìm kiếm.
- Mặc định `PREFLIGHT_ONLY=True`: OCR đúng 5 keyframe, gồm các frame
  `L26_V221:419`, `L30_V096:637` và `L30_V096:646` từng gây vòng lặp ở model
  thử nghiệm trước, hiển thị kết quả và không upload. Sau khi duyệt 5 mẫu, đặt
  `PREFLIGHT_ONLY=False` để chạy pilot 5 video. Sau đó pin input/model revision,
  đặt `MAX_VIDEOS_PER_RUN=None`, khởi động lại runtime và chạy Run All.
- Giữ `RUN_SMOKE_TEST_BEFORE_INGEST=True`: pilot/full luôn chạy lại 5 frame
  smoke test trước khi mở hàng đợi video, nên lỗi model/inference hệ thống sẽ
  dừng sớm trước khi tải hàng loạt archive.
- Khi chạy lại, 5 video có marker và hash Parquet hợp lệ được bỏ qua.
- Mỗi GPU giữ một pipeline OCR độc lập; Kaggle hai T4 xử lý hai video song song.
  Detector giữ cạnh dài tối đa 1920. Recognizer gom tối đa 32 crop chữ trong
  cùng frame; confidence dưới 0.45 bị lọc.
- PP-OCRv6 không sinh văn bản tự do nên không có thinking, sampling, token limit
  hoặc retry hallucination. Lỗi inference làm video thất bại và không ghi marker.
- Checkpoint được ghi liên tục và `fsync` mỗi 25 frame. Uploader một thread commit
  từng video song song với inference, backlog tối đa 2 video; mỗi GPU prefetch
  thêm 1 video. Nếu kernel hoặc mạng lỗi, chạy lại tiếp tục từ checkpoint hợp lệ.
- Circuit breaker ngừng cấp việc mới nếu cùng lỗi inference lặp 3 lần liên tiếp,
  tránh tiếp tục tải hàng trăm video khi model hoặc generation bị lỗi hệ thống.
- Chỉ resume/skip khi marker và cả hai artifact từ xa khớp model, input revision,
  cấu hình inference, schema và SHA-256 của từng Parquet.
- Hạn chế: từ điển của `PP-OCRv6_medium_rec` thiếu 44 chữ thường tiếng Việt
  (dấu hỏi, dấu nặng, phần lớn tổ hợp mũ + dấu), nên không đọc đúng tiếng Việt có dấu.

### OCR — HunyuanOCR-1.5

- Cùng hai artifact và marker như notebook PP-OCRv6 nhưng ghi vào
  `aqpahm/aic2026-ocr-hunyuanocr`. `ocr.parquet` lưu thêm `raw_output`,
  `finish_reason`, `completion_tokens` và `ocr_status`
  (`ok`/`no_text`/`truncated`/`repetition`/`parse_error`); không có confidence.
- Cell đầu cài `vllm==0.18.1` (kéo torch 2.10, transformers 4.57). Lần đầu cài xong
  cell dừng lại và yêu cầu **restart runtime**, sau đó chạy lại từ đầu. Không cài
  transformers 5.x vào cùng môi trường: bản đó làm hỏng HunYuanVL trong vLLM 0.18.1.
- Notebook không import torch. Mỗi GPU chạy một `vllm serve` trong tiến trình con;
  log nằm ở `logs/vllm-gpu*-port*.log` và phần cuối log được in ra nếu server chết.
- Mặc định `VLLM_DTYPE="float16"` cho mọi runtime vì T4 không có bfloat16. Nếu
  preflight cho output rác hoặc rỗng trên frame rõ ràng có chữ, đổi sang `"float32"`.
- Preflight OCR 5 frame (gồm ba frame từng gây lặp ở model thử nghiệm trước), in
  output thô khi trạng thái bất thường, rồi đo tốc độ trên 96 frame và ước tính số
  giờ cho mỗi 100k keyframe. Dataset keyframe hiện có 335.477 frame trên 873 video.
- Frame trong một video được gửi song song tới mọi server; checkpoint chỉ ghi phần
  tiền tố liên tục nên resume luôn đúng thứ tự. Input video kế tiếp được tải trước,
  upload chạy trên thread riêng.
- Circuit breaker dừng ngay khi server chết hoặc trả lỗi 4xx, và khi cùng lỗi lặp
  3 lần liên tiếp.

## Điều kiện hoàn tất

Audit cuối phải thỏa cả ba điều kiện:

1. Mọi video thực sự có trong archive được quan sát trong lần chạy hoặc đã có
   marker từ xa hợp lệ.
2. Không còn video failure hoặc archive failure chưa được xử lý.
3. Marker và các artifact bắt buộc có cùng schema, model/source provenance và
   số dòng nhất quán.

Một archive tải thành công không có nghĩa toàn bộ video bên trong đã ingest
thành công. Một lệnh upload báo thành công cũng chưa đủ nếu marker hoặc file bắt
buộc không qua validation.

## Xử lý sự cố thường gặp

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| HTTP 429 khi upload | Vượt giới hạn số commit Hugging Face | Giữ upload theo lô, chờ rate limit hết rồi chạy lại; marker hợp lệ sẽ được skip |
| HTTP 5xx hoặc timeout | Hugging Face/mạng tạm thời lỗi | Để retry/backoff chạy; nếu phiên ngắt, khởi động runtime mới và resume |
| CUDA OOM | Batch quá lớn hoặc VRAM phân mảnh | Notebook tự giảm batch; nếu vẫn lỗi, restart runtime và giảm batch khởi tạo |
| GPU hiển thị 0% | Job đang tải, decode, ghi file hoặc upload | Kiểm tra log bước hiện tại và dung lượng disk/network trước khi kết luận bị treo |
| Download archive dừng lâu | Mạng chậm, Xet reconstruction hoặc disk gần đầy | Kiểm tra tốc độ mạng/disk; restart và resume nếu không còn tiến triển |
| Marker có nhưng dữ liệu sai/thiếu | Marker cũ hoặc upload dở | Bật validation chặt hơn, xóa đúng output video lỗi rồi chạy lại |
| Model trả kết quả rỗng | Nội dung thật sự rỗng hoặc model thất bại | Giữ trạng thái rỗng hợp lệ khi có bằng chứng; dùng fallback theo logic của notebook |
| Thiếu package sau khi Colab cập nhật | Runtime image thay đổi | Chạy lại cell cài dependency và restart runtime nếu cell yêu cầu |

## Đồng bộ thay đổi từ cloud về repository

Notebook trong `ingestion/` phải là bản nguồn sạch. Nếu bạn sửa logic trực tiếp
trên Colab/Kaggle:

1. download notebook đã chạy;
2. tạo một bản sao dành cho source;
3. xóa output và execution count;
4. kiểm tra không còn token, đường dẫn tạm cá nhân hoặc log dung lượng lớn;
5. thay đúng notebook nguồn và chạy kiểm tra cục bộ;
6. giữ bản có output ở ngoài repository.

Có thể làm sạch bản sao bằng:

```powershell
uv run python scripts/strip_notebook_outputs.py path/to/notebook.ipynb
uv run python scripts/check_notebooks.py
```

## Checklist khi thêm hoặc sửa notebook

- [ ] Tên file mô tả đúng module và model.
- [ ] Input/output repository và revision được khai báo tập trung.
- [ ] Model ID, revision/checksum và tham số quan trọng được ghi vào marker.
- [ ] Không chứa secret, output cell hoặc execution count.
- [ ] Có pilot limit, retry/backoff, upload theo lô và dọn thư mục tạm.
- [ ] Có kiểm tra artifact trước khi công bố marker.
- [ ] Resume chỉ skip marker hợp lệ.
- [ ] Audit dựa trên video thật trong archive.
- [ ] Hỗ trợ đúng số GPU có sẵn và xử lý CUDA OOM.
- [ ] Data contract và tài liệu được cập nhật nếu schema thay đổi.

Xem thêm [quy ước ingestion](../docs/ingestion-standard.md) và
[data contract](../docs/data-contracts.md).
