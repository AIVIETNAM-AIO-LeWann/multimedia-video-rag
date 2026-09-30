# Rà soát nghiêm ngặt query cho hướng chỉ dùng visual embedding

Nguồn là 91 file `.txt` trong ba ZIP `SOTUYEN1-bo-de-thi.zip`,
`SOTUYEN2-bo-de-thi.zip` và `SOTUYEN3-bo-de-thi.zip` do người dùng cung cấp.
Đã đọc từng câu gốc và đối chiếu 91/91 với bản chép trong
`artifacts/eval/aic2026-visual-queries.csv` hoặc
`artifacts/eval/aic2026-excluded-queries.csv` sau khi chuẩn hóa khoảng trắng.
ZIP SOTUYEN3 vẫn đặt tên file `query-p2-*`; mã `p3-*` chỉ dùng để tránh trùng.

Quyết định và lý do **cho từng câu** nằm trong
[visual-only-screening.csv](visual-only-screening.csv). CSV không sao chép nội
dung đề gốc. Đây là sàng lọc bằng **văn bản**; chưa kiểm tra video gốc và ảnh
keyframe tương ứng với từng câu.

| Trạng thái | Tổng | KIS | QA | TRAKE | Dùng thế nào |
|---|---:|---:|---:|---:|---|
| `core` | 30 | 27 | 2 | 1 | Bộ chính nghiêm ngặt để bắt đầu benchmark visual-only. |
| `needs_verification` | 36 | 30 | 2 | 4 | Chưa đưa vào bộ chính; cần xem video/keyframe để xác nhận. |
| `retrieval_only` | 12 | 0 | 12 | 0 | Có thể đo tìm đúng moment, không đo QA trọn vẹn. |
| `exclude` | 13 | 8 | 5 | 0 | Không phù hợp với phép đánh giá visual-only hiện tại hoặc trùng câu. |

## Tiêu chí quyết định

1. **Giữ nguyên điều kiện quyết định của câu gốc.** Không gọi là `core` nếu
   phải bỏ tên riêng, con số, nguyên liệu, hành động hoặc mốc thời gian cần
   thiết để xác định đáp án. Chi tiết không nhìn thấy nhưng có nhiều dấu hiệu
   hình khác đủ định danh được đưa vào `needs_verification`, không tự động giữ.
2. **Visual-only không đồng nghĩa với không có vật thể.** Vật thể, màu sắc, số
   lượng, bố cục, hành động và thứ tự cảnh đều là nội dung hình. Một câu khó
   đếm hoặc khó phân biệt màu là phép thử có ích nếu bằng chứng còn rõ trên ảnh.
   Không loại chỉ vì một OD hoặc caption model có thể giúp thêm.
3. **QA có hai tiêu chí riêng:** hình có đủ để tìm đúng moment; hình có đủ để
   trả lời câu hỏi. `p2-28` chuyển sang `retrieval_only`: topping dạng sợi nhìn
   thấy được, nhưng không thể suy chắc nó là thịt con gì chỉ từ thành phẩm.
   `p1-15` cũng là `retrieval_only` vì phải đọc mức động đất 4 trong chú giải
   trước khi đếm chấm trên bản đồ.
4. **TRAKE cần đủ cả sự kiện và thời điểm.** Index hiện tại chọn candidate
   keyframe cách nhau khoảng 2 giây trong shot, rồi còn loại ảnh gần trùng.
   Những yêu cầu như “lần đầu chạm”, “tép cam thứ tư”, “hoàn tất xoay” được
   chuyển sang `needs_verification`; chỉ có thể chốt sau khi xem độ phủ frame
   hoặc bổ sung bước dò frame dày quanh ứng viên.
5. Câu phụ thuộc vào nội dung bảng, chữ trên biển, lời phỏng vấn, lượng gia vị
   hoặc âm thanh không được coi là giải trọn vẹn bằng visual embedding.

So với danh sách lọc trước, `p3-16` chuyển từ giữ sang `exclude`: phần điều kiện
chính là lượng và vị của gia vị, không thể xác nhận chắc từ ảnh. `p2-28` vẫn có
thể dùng để đo retrieval, nhưng không dùng để chấm QA đầy đủ. Các câu có
`needs_verification` có tiềm năng visual, song **không nằm trong 30 câu bộ
chính** cho tới khi kiểm tra video/keyframe.

Giới hạn quan trọng: `core` chỉ có nghĩa là **phù hợp theo văn bản và hợp lý với
thiết kế index**, không phải đã chứng minh ảnh chứa đủ bằng chứng hoặc model sẽ
tìm đúng. Không có video gốc/keyframe của đủ 91 câu trong ba ZIP này để xác nhận
điều đó.

## Khi thêm VLM để trả lời QA

Bảng trên là sàng lọc cho **retrieval bằng visual embedding** và khả năng trả
lời trực tiếp từ hình mà chưa kiểm tra VLM. VLM nhìn ảnh/video gốc là một bước
sau retrieval; nó có thể đọc chữ lớn trong frame dù hệ thống không có OCR index.
Vì vậy bảy câu `retrieval_only` sau cần được **xét lại có điều kiện** khi chọn
VLM và có frame đủ nét: `p1-3`, `p1-9`, `p1-15`, `p2-7`, `p2-27`, `p2-29`,
`p3-8`. Chúng chưa được tự động chuyển sang `core` vì có thể cần crop/độ phân
giải cao hoặc chữ chỉ xuất hiện giữa các keyframe.

Các câu cần nội dung lời nói hoặc tri thức không thấy được từ hình vẫn không
được giải quyết chỉ bằng VLM nhìn frame. Với TRAKE, VLM có thể kiểm tra vài
ứng viên khó, nhưng vẫn cần bộ căn chỉnh nhiều sự kiện theo cùng video và
thứ tự thời gian; baseline start/end hiện có không đủ cho 3–4 mốc của TRAKE.
