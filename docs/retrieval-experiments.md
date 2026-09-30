# Thử nghiệm retrieval

> **Trạng thái:** mã thử nghiệm cho nghiên cứu, chỉ dùng hai image embedding SigLIP 2 và
> BEiT-3. Mọi cấu hình dưới đây chưa phải quyết định kiến trúc.

## 1. Chuẩn bị

```powershell
$env:HF_TOKEN = "<token đọc các dataset aqpahm/aic2026-*>"
uv sync --extra dev
uv run python scripts/build_retrieval_index.py --output-dir indexes/dev
uv run python scripts/download_keyframes.py --workers 8      # ảnh để xem, ~51 GB
```

`build_retrieval_index.py` tải parquet, safetensors và marker của keyframe, SigLIP 2,
BEiT-3 và caption (không tải ảnh) vào `data/hf/`, ghim mỗi dataset theo commit và ghi
tiến độ vào `data/hf/sources.json` để chạy lại không tải lại phần đã xong. Build dừng
nếu thiếu marker hoặc lệch `frame_uid`; index chỉ được đổi tên từ `.partial` khi xong.
Cấu trúc index: xem `docs/data-contracts.md`.

`download_keyframes.py` tải `keyframes.tar` của mọi video đúng revision trong manifest
và chạy lại được khi bị ngắt. Không cài `hf_xet`: khi thử, tải qua Xet chậm hơn khoảng
5 lần so với HTTP thường.

## 2. Encode câu truy vấn

Máy local không cần tải model: câu truy vấn được encode trên Colab/Kaggle bằng đúng
model và revision ghi trong `indexes/dev/manifest.json`. Cả hai encoder nhận câu tiếng
Anh (SigLIP 2 viết thường, pad 64 token; BEiT-3 giữ hoa/thường, tối đa 64 token).

- **Theo lô:** `notebooks/retrieval/encode-query-vectors.ipynb`. Tải lên CSV (cột
  `query_id`, `query_vi`, `query_en`) và `manifest.json`, Run all, tải
  `query_vectors.npz` về. File ghi model/revision đã dùng; code local từ chối file lệch
  revision với index hoặc khi câu đã bị sửa sau khi encode.
- **Tương tác:** `notebooks/retrieval/query-encoder-server.ipynb` mở API `/encode` qua
  Cloudflare quick tunnel, có `X-API-Key` sinh ngẫu nhiên mỗi phiên; ô cuối in
  `ENCODER_URL` và `ENCODER_KEY`. Xoá output trước khi lưu notebook.

Chỉ 64 token đầu của câu được đọc: 30/79 câu trong bộ đánh giá dài hơn mức này.

## 3. Web UI

```powershell
$env:ENCODER_KEY = "<key>"
uv run python scripts/serve_ui.py --encoder-url <ENCODER_URL>
```

Mở `http://127.0.0.1:8800`. Không có `--encoder-url` thì chỉ tìm được các câu trong
`artifacts/eval/aic2026-visual-queries.csv` đã có vector trong
`artifacts/eval/encode/aic2026-query_vectors.npz` (đánh dấu ● trong danh sách).

- Mỗi encoder lấy top 1.000 keyframe; mọi ứng viên trong hợp hai tập được chấm bằng
  **cả hai** encoder, rồi fusion theo `rrf` (w/(60 + hạng)), `max-norm` (điểm chia cho
  điểm cao nhất của mỗi model) hoặc một encoder. Trọng số SigLIP 2 chỉnh trên UI; mặc
  định giữ một keyframe mỗi shot.
- Toàn kho được chấm bằng tích ma trận–vector trên bản FP16 của vector trong index
  (torch, khoảng 0,1 giây mỗi encoder; FAISS bản pip không có AVX2 mất khoảng 1,5 giây).
  Lần đầu giải nén vector từ FAISS và cache vào `artifacts/cache/<index>/` (~1,5 GB).
- Ảnh đọc thẳng từ `keyframes.tar` theo offset. Câu trong bộ đánh giá hiện kèm nguyên
  văn tiếng Việt (chỉ để đọc). Bấm một kết quả để xem keyframe lân cận; "Đánh dấu là
  đáp án" ghi vào `artifacts/labels/labels.csv`.
- Server chỉ nghe `127.0.0.1` và chỉ dùng thư viện chuẩn Python.

## 4. Baseline từ paper

```powershell
uv run python scripts/run_paper_baseline.py --query-vectors artifacts/eval/encode/aic2026-query_vectors.npz --method paper1
uv run python scripts/run_paper_baseline.py --query-vectors artifacts/eval/encode/aic2026-query_vectors.npz --method paper2
```

- `paper1` (arXiv:2504.08384): top-M mỗi model, gộp điểm ±2 keyframe lân cận cùng video
  (tổng), max-norm, dual-query tìm cặp keyframe đầu/cuối quanh keyframe gốc.
- `paper2` (GRAB, arXiv:2504.09298): BEiT-3, SuperGlobal rerank top-100, ABTS với cửa sổ
  10/15/20 s và điểm ổn định.

Các chi tiết paper không nói rõ được ghi trong docstring của
`src/multimedia_video_rag/retrieval/baselines.py`.

## 5. So sánh kỹ thuật truy vấn

```powershell
uv run python scripts/compare_query_techniques.py
```

Xuất `artifacts/eval/techniques/index.html`, đặt cạnh nhau cho từng câu:

- **baseline**: câu đầy đủ, RRF hai encoder;
- **PRF**: vector ảnh trung bình của 3 keyframe đầu làm "ảnh mẫu", gộp RRF với baseline;
- **decomposition**: mỗi câu con (`#start`/`#end` hoặc `#E1..#E4`) tìm riêng, rồi ghép
  chuỗi cùng video, đúng thứ tự thời gian, mỗi bước cách nhau không quá 60 s.

Trang này chỉ để xem bằng mắt; chưa có ground truth nên không đo đúng sai.

## 6. Gán nhãn

Mẫu `docs/evaluation/queries-template.csv`: mỗi dòng là một khoảng đúng (`video_id`,
`start_sec`, `end_sec`); `query_id` lặp lại khi câu có nhiều khoảnh khắc đúng; `split`
là `dev` (chỉnh tham số) hoặc `test` (báo cáo). Tiêu chí chọn câu: `docs/evaluation/`.
Script đánh giá (Recall@K, MRR, kiểm định thống kê) sẽ được viết khi có nhãn.
