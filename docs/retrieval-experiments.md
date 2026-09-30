# Thử nghiệm retrieval

> **Trạng thái:** thử nghiệm để benchmark. Index, trọng số và cấu hình dưới đây
> chưa phải quyết định kiến trúc (xem `docs/research/video-retrieval-architecture-review.md`).

## 1. Build index

```powershell
$env:HF_TOKEN = "<token đọc các dataset aqpahm/aic2026-*>"
uv sync --extra dev
uv run python scripts/build_retrieval_index.py --output-dir indexes/dev
```

Script chỉ tải parquet, safetensors và marker (không tải `keyframes.tar`) vào
`data/hf/`, ghim mỗi dataset theo commit và ghi tiến độ vào `data/hf/sources.json`
để chạy lại không tải lại module đã xong. Build dừng nếu thiếu marker hoặc lệch
`frame_uid`; index chỉ được đổi tên từ `.partial` khi đã xong.

| File | Nội dung |
|---|---|
| `siglip.faiss`, `beit3.faiss` | Tìm kiếm inner product chính xác, vector lưu FP16, id = `frame_row` |
| `metadata.sqlite` | `frames`, `captions` + FTS5, `asr_segments`/`asr_chunks` + FTS5 bỏ dấu, `od_detections`, `od_frame_labels` |
| `manifest.json` | Nguồn, revision dataset, model/revision của encoder, số lượng |

## 2. Tìm kiếm thử

```powershell
uv sync --extra dev --extra query
uv run python scripts/search_index.py --vi "lính cứu hỏa dập lửa" --en "firefighters putting out a fire" --objects "person:3"
```

- **SigLIP 2** và **BEiT-3** đều dùng câu tiếng Anh (`--en`); `--siglip-text vi` chỉ để thử nghiệm.
- **BEiT-3** và **caption** chỉ hiểu tiếng Anh, nên cần `--en`; thiếu thì nhánh bị bỏ qua.
- **ASR** tìm theo chunk khoảng 25 giây: FTS bỏ dấu để lấy ứng viên, rồi xếp lại ưu
  tiên khớp đúng cụm có dấu, sau đó số từ khớp đúng dấu, cuối cùng BM25.
- **OD** là ràng buộc mềm: frame thỏa nhiều ràng buộc hơn xếp trên, nhãn lạ không
  loại frame nào.

Lần đầu, encoder tải model về máy: SigLIP 2 khoảng 4,5 GB (checkpoint gộp vision và
text, chỉ phần text được nạp), BEiT-3 khoảng 1,35 GB vào `models/beit3/`. Hai
encoder được nạp lần lượt và giải phóng sau khi encode. `--dtype bfloat16` giảm
một nửa RAM nhưng vector query lệch nhẹ so với float32.

### Encode query trên Colab/Kaggle (không tải model về máy)

Khi máy local thiếu RAM hoặc băng thông, chỉ bước biến query thành vector cần model;
bước tìm kiếm chỉ cần index local.

1. Ghi query vào CSV (cột `query_id`, `query_vi`, `query_en`, tùy chọn `objects`; file
   gán nhãn dùng được trực tiếp).
2. Mở `notebooks/retrieval/encode-query-vectors.ipynb` trên Colab/Kaggle, tải lên
   CSV đó và `indexes/dev/manifest.json`, rồi Run all. Không cần `HF_TOKEN`.
3. Tải `query_vectors.npz` về máy và chạy:

```powershell
uv run python scripts/search_index.py --query-vectors query_vectors.npz --query-id q1
uv run python scripts/evaluate_retrieval.py --queries queries.csv --query-vectors query_vectors.npz
```

File vector ghi model/revision đã dùng; script từ chối file encode bằng revision khác
với ảnh trong index, hoặc khi nội dung query trong CSV đã sửa sau khi encode. Đổi
query thì phải encode lại trên notebook.

### Web UI tìm kiếm (hệ thống xuất phát điểm)

Hệ thống tương tác chỉ dùng hai embedding SigLIP 2 + BEiT-3, dùng để quan sát lỗi
trước khi quyết định hướng nghiên cứu tiếp; chưa có temporal/caption.

1. Tải ảnh keyframe (khoảng 63 GB, chạy lại để tải tiếp khi bị ngắt):

   ```powershell
   uv run python scripts/download_keyframes.py --workers 8
   ```

   Không cài `hf_xet`: khi thử trên máy này, tải qua Xet chậm hơn khoảng 5 lần so với
   HTTP thường.

2. Mở `notebooks/retrieval/query-encoder-server.ipynb` trên Colab (GPU), tải lên
   `indexes/dev/manifest.json`, Run all. Ô cuối in `ENCODER_URL` và `ENCODER_KEY`
   (key sinh ngẫu nhiên mỗi phiên; xoá output trước khi lưu notebook).
3. Chạy UI trên máy local rồi mở `http://127.0.0.1:8800`:

   ```powershell
   $env:ENCODER_KEY = "<key>"
   uv run python scripts/serve_ui.py --encoder-url <ENCODER_URL>
   ```

   Không có `--encoder-url` thì chỉ tìm được các câu trong
   `artifacts/eval/aic2026-visual-queries.csv` đã có vector trong
   `artifacts/eval/encode/aic2026-query_vectors.npz` (đánh dấu ● trong danh sách).

- Mỗi encoder lấy top 1.000 keyframe; mọi ứng viên trong hợp hai tập được chấm bằng
  **cả hai** encoder, rồi fusion theo `rrf`, `max-norm` hoặc một encoder, với trọng số
  SigLIP 2 chỉnh trên UI. Mặc định giữ một keyframe mỗi shot.
- Toàn kho được chấm bằng tích ma trận–vector trên bản FP16 của vector trong index
  (torch, khoảng 0,1 giây mỗi encoder trên CPU; FAISS bản pip không có AVX2 mất khoảng
  1,5 giây). Lần chạy đầu giải nén vector từ FAISS và cache vào `artifacts/cache/dev/`
  (khoảng 1,5 GB RAM và đĩa).
- Ảnh đọc thẳng từ `keyframes.tar` theo offset, không giải nén. Bấm một kết quả để xem
  keyframe lân cận cùng video; "Đánh dấu là đáp án" ghi vào `artifacts/labels/labels.csv`.
- Server chỉ nghe `127.0.0.1`. Encoder server kiểm tra `X-API-Key`; UI từ chối encoder
  có model/revision khác với index.

## 3. Cấu hình so sánh

| ID | Cấu hình |
|---|---|
| A | SigLIP 2 |
| B | BEiT-3 |
| C | SigLIP 2 + BEiT-3, RRF |
| D | Ứng viên của C (top 1.000) được xếp lại cùng caption/ASR/OD |
| E | Hợp ứng viên của mọi nhánh, weighted RRF |
| C-max | SigLIP 2 + BEiT-3, cộng điểm đã chia cho điểm cao nhất của mỗi model (arXiv:2504.08384, Alg. 3) |
| C-nbr | Như C, nhưng điểm visual của mỗi frame là trung bình độ khớp query của frame đó và ±2 keyframe lân cận cùng video (Neighbor Score Aggregation, Alg. 2) |
| C-max-nbr | Kết hợp C-max và C-nbr |
| E-nbr | Như E, nhánh visual dùng điểm đã gộp lân cận |

Paper không nêu rõ vùng lân cận và dùng tổng điểm; ở đây dùng trung bình để frame ở
đầu/cuối video không bị thiệt. Bước gộp lân cận làm thời gian tìm tăng khoảng 2 giây
mỗi query trên CPU (giải nén vector FP16 của khoảng 5.000 frame lân cận cho mỗi model).

Trọng số ban đầu (visual 1,0; ASR 0,7; caption 0,5; OD 0,3) chưa được tinh chỉnh.
Kết quả được gom theo shot trước khi tính hạng. Chỉ chỉnh trọng số trên split `dev`.

## 4. Gán nhãn và đánh giá

Sao chép `docs/evaluation/queries-template.csv` ra ngoài repository hoặc vào
`artifacts/` rồi điền, mỗi dòng là một khoảng đúng:

| Cột | Ý nghĩa |
|---|---|
| `query_id` | Mã query; lặp lại khi query có nhiều khoảnh khắc đúng |
| `split` | `dev` để chỉnh tham số, `test` để báo cáo |
| `query_type` | `visual`, `asr`, `object`, `semantic`, `ocr`, `temporal`, `mixed` |
| `query_vi` | Nguyên văn query tiếng Việt |
| `query_en` | Bản dịch tiếng Anh (cho BEiT-3 và caption); để trống nếu chưa có |
| `objects` | Ràng buộc vật thể theo nhãn OD, ví dụ `car:2; person` |
| `video_id`, `start_sec`, `end_sec` | Khoảng thời gian đúng trong video |
| `notes` | Ghi chú tùy ý |

```powershell
uv run python scripts/evaluate_retrieval.py --queries artifacts/queries.csv --split dev
```

Một keyframe được tính là đúng khi thuộc đúng video và có timestamp trong khoảng
đã gán ± 2 giây (`--tolerance-sec`). Script ghi `summary.csv` (Recall@1/5/10/20/50/100
và MRR theo cấu hình và loại query) và `per-query-ranks.csv` vào
`artifacts/retrieval-eval/`. Vector query được cache theo revision model.

## 5. Giới hạn đã biết

- Chưa có OCR; query cần đọc chữ sẽ phụ thuộc vào visual/ASR.
- Chưa có bước dịch query sang tiếng Anh; `query_en` phải điền tay.
- Tìm visual trên CPU khoảng 0,5 giây mỗi query do vector lưu FP16; script đánh giá
  gom query theo lô để giảm chi phí.
- Chưa có reranker và bộ giải temporal (cấu hình G, H trong tài liệu kiến trúc).
