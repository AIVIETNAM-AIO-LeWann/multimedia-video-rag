# Trạng thái dự án

## Hướng nghiên cứu

Text-to-video moment retrieval trên 873 video AIC 2026 (335.477 keyframe), chỉ dùng hai
image embedding SigLIP 2 và BEiT-3, câu truy vấn tiếng Anh, không huấn luyện thêm. Mục
tiêu là tìm đóng góp hiệu quả cho truy vấn một cảnh và chuỗi cảnh, theo kiểu các bài
arXiv:2504.08384 và GRAB (arXiv:2504.09298).

## Dữ liệu đã có

| Thành phần | Dataset Hugging Face | Tình trạng |
|---|---|---|
| Keyframe | `aqpahm/aic2026-keyframes-transnetv2` | Đủ 873 video |
| SigLIP 2 | `aqpahm/aic2026-visual-siglip2-so400m` | Đủ |
| BEiT-3 | `aqpahm/aic2026-visual-beit3-large-coco-retrieval` | Đủ |
| Caption BLIP-2 (chỉ hiển thị) | `aqpahm/aic2026-caption-blip2-opt-2.7b-coco` | Đủ |

## Đã làm

- Index cục bộ (FAISS chính xác + SQLite) và chấm điểm toàn kho bằng tích ma trận FP16
  (khoảng 0,1 giây mỗi encoder trên CPU).
- Hai baseline tái hiện: arXiv:2504.08384 (max-norm, gộp điểm lân cận, dual-query) và
  GRAB (SuperGlobal, ABTS). Quan sát đầu tiên: bước tìm theo thời gian của cả hai
  thường dừng ngay tại keyframe gốc (15/48 và 17/48 câu chuỗi cảnh).
- Bộ 79 câu truy vấn tiếng Anh chọn lọc từ đề sơ tuyển (58 KIS, 16 QA, 5 TRAKE), kèm
  câu con tách sẵn cho 48 câu chuỗi cảnh; lý do chọn/loại ở `docs/evaluation/`.
- Web UI cục bộ để tìm, xem keyframe lân cận và ghi nhãn đáp án.
- Gallery so sánh kỹ thuật truy vấn (baseline, PRF, tách câu + ghép chuỗi theo thời gian).

## Chưa có

- **Ground truth** cho 79 câu, nên chưa có số liệu Recall/MRR.
- Quyết định về fusion, kỹ thuật truy vấn và bộ tìm chuỗi thời gian: chờ số liệu.

## Tài liệu nghiên cứu

`docs/research/` chứa hai báo cáo khảo sát (truy vấn khoảnh khắc; khung TCR) và ghi
chú nguồn.
