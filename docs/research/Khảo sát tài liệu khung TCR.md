# TCR đứng vững nhờ DP, không nhờ caption

Khảo sát từng thành phần cho thấy **không thành phần đơn lẻ nào của TCR là mới**. DP có thứ tự chính xác O(N·T) trên keyframe theo scene đã có trong DANTE (AIC 2025). Phạt khoảng cách theo giây đã có trong vitrivr/Cineast IDA (2021) và hệ Cascaded 2512.12935. Trần khoảng cách cứng có trong MADTempo, VISIONE và vitrivr-engine. Trạng thái drop có chi phí percentile là của Drop-DTW (NeurIPS 2021). "Không cần pivot" đã có ở DANTE, MADTempo và SCAC. Về mặt toán học, C2 chính là một bước lặp Jacobi của diffusion Zhou/Iscen. Những gì **không tìm thấy trong khảo sát của chúng tôi** gồm ba thứ. Thứ nhất là *tổ hợp* sau: một DP chính xác, phạt theo giây dạng cộng tính, trần theo giây giải bằng sliding-window max, drop giá percentile, chạy trên keyframe shot có timestamp không đều và xuất span. Thứ hai là kernel joint-bilateral (thời gian × độ giống embedding) áp lên *điểm liên phương thức* text→keyframe. Thứ ba, và giá trị nhất, là một **đánh giá định lượng đối đầu giữa các bộ chấm chuỗi** (pivot walk, DP theo chỉ số, beam decay, greedy chaining): không paper AIC hay VBS nào công bố điều này. Tài liệu cũng buộc phải sửa ba chỗ. (1) C2 như đang đặc tả chuẩn hóa chỉ trên hàng xóm, nên ở ranh giới shot thật nó vẫn kéo điểm theo đủ trọng số μ; tức là nó *không* bảo toàn biên như đã tuyên bố, và phải thêm self term. (2) Caption BLIP-2 COCO ngắn khoảng 10 từ gần như chắc chắn trùng thông tin với BEiT-3 finetune COCO. Vì vậy cả vai trò xác minh ① lẫn unary ② phải thành số hạng ablate có mặc định 0. Số liệu LexiCLIP trong báo cáo trước bị nhầm chiều: 67,4/65,7 là I→T, còn T→I trên COCO là 52,7 (FT) và 41,7 (ZS), so với 47,8 của SigLIP ViT-B/16. (3) Fusion thích ứng ③ chỉ có tiền lệ gián tiếp trong QPP với tương quan khoảng 0,2–0,45, nên kỳ vọng thực tế là vài điểm. Rủi ro novelty lớn nhất là **SCAC (IEEE TIP 2026)**: training-free, event chain do LLM sắp theo thứ tự, xếp hạng cấp corpus, nhưng công thức chấm chưa công bố. Với 79 query, khoảng tin cậy 95% của một tỷ lệ quanh 50% là ±11 điểm phần trăm, nên chỉ khác biệt khoảng 10–15 điểm trở lên mới phân xử được. Bộ query và giao thức đánh giá do đó là một phần của đóng góp, không phải phụ lục.

> Phạm vi: báo cáo tổng hợp sáu bộ ghi chú trong `docs/research/notes/Khảo sát tài liệu khung TCR/` và kế thừa `docs/research/notes/Nghiên cứu truy vấn video khoảnh khắc/`. Chỗ nào hai bộ ghi chú mâu thuẫn, bộ mới được ưu tiên. Mọi cấu hình ở đây là thử nghiệm để benchmark, không phải quyết định kiến trúc đã chốt. Bằng chứng nội bộ gồm: span trung vị 2 giây và 15/48 query chuỗi sụp về pivot với dual-query walk của 2504.08384; đổi RRF ↔ max-norm làm đổi video top-1 ở khoảng 40% query. Đây là số đo của nhóm, chưa công bố. "Không tìm thấy" luôn có nghĩa là không tìm thấy trong khảo sát này, không phải "chưa ai làm".

## C1: DP có thứ tự đã có chủ, phạt theo giây mới là điểm tựa

### Training-free VTG chỉ xử lý một đường cong trong một video đã biết

Mọi phương pháp VTG training-free tìm được đều cùng một khuôn. Chúng chấm một query (hoặc 2–3 sub-query do LLM tách) trên frame lấy mẫu dày, thường 3 fps, trong một video *đã biết*. Sau đó chúng biến đường cong similarity một chiều thành một span bằng contrast, đạo hàm, ngưỡng hoặc mở rộng từ anchor. Không phương pháp nào chạy DP có thứ tự chung cho N sự kiện. TFVTG (ECCV 2024) tách query bằng LLM thành sub-event kèm quan hệ "simultaneously" (lấy giao) hoặc "sequentially" (lấy hợp), giữ top-3 proposal mỗi sub-event và loại tổ hợp sai thứ tự. Đó là lọc hậu kỳ, không có mô hình khoảng cách hay chấm chung. Riêng phần LLM chỉ nâng R@0,5 trên Charades từ **42,32 lên 43,17, rồi 44,12 khi thêm ràng buộc thứ tự/quan hệ** ([TFVTG](https://arxiv.org/html/2408.16219)). Kết quả chính của TFVTG là Charades-STA R@0,3/0,5/0,7/mIoU = 67,04/49,97/24,32/44,51.

Paper gần nhất với bài toán "span từ keyframe" là **Towards Robust Zero-Shot VTG (CVPRW 2026)**. Anchor được chọn tự động bằng argmax điểm text–frame. Sau đó đường cong text nhiễu được *thay* bằng độ tự-giống thị giác với anchor, s^im_j = v_k·v_j. Span mở rộng từ anchor cho tới khi điểm rơi dưới ngưỡng τ. τ được tìm bằng phép tách trung vị lặp: τ₀ = median, rồi τ_{i+1} = (median(s₁) + median(s₂))/2 cho tới khi hội tụ. Phương pháp đạt **Charades 68,63/50,27/28,92/46,93 và ActivityNet 50,25/30,18/13,90/34,49**. Phát hiện quan trọng hơn là về độ bền: khi tinh chỉnh trên một tập rồi thử trên tập khác, **TFVTG mất 7,91 mIoU, TAG mất 6,73, còn phương pháp này mất tối đa 2,32** ([CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/papers/Banditakkarakul_Towards_Robust_Zero-Shot_Video_Temporal_Grounding_CVPRW_2026_paper.pdf)). Hai hệ quả cho TCR:

- Về cấu trúc, đây vẫn là "pivot rồi mở rộng", chỉ khác là pivot tự động và việc mở rộng dựa trên độ giống thị giác, không dựa trên text. Đây chính là cách sửa lỗi mà walk của 2504.08384 thiếu.
- Các siêu tham số của C1 (β, λ/giây, trần G, chi phí drop) sẽ chịu đúng rủi ro "tinh chỉnh trên test" này. Chúng phải được cố định trên dev hoặc tự thích nghi theo từng video (percentile, tách trung vị).

### Căn chỉnh bước cho thấy giải chung thắng giải tham lam

**Drop-DTW** là DP chính xác chuẩn mực có chi phí drop. Ký hiệu D⁺ là nhánh match, D⁻ là nhánh drop frame:

- D⁺_{i,j} = C_{i,j} + min{D_{i−1,j−1}, D_{i,j−1}, D⁺_{i−1,j}}
- D⁻_{i,j} = d^x_j + D_{i,j−1}
- D_{i,j} = min(D⁺, D⁻)

Chi phí drop là một percentile của ma trận chi phí, tính riêng cho từng cặp chuỗi: p = 30% cho định vị bước. Biến thể drop cả phía step nằm trong phụ lục. Khi chạy thuần suy luận trên đặc trưng MIL-NCE đóng băng, Drop-DTW đạt **CrossTask Acc/IoU 70,2/30,5**. Để so sánh: DTW đạt 11,2/10,1, Needleman–Wunsch 68,8/9,5, và "greedy drop rồi DTW" 60,1/13,8. Trên YouCook2, Drop-DTW đạt 66,0/47,5, còn greedy drop + DTW chỉ 54,3/34,1 ([Drop-DTW](https://arxiv.org/abs/2108.11996)). Đây là bằng chứng phương pháp luận trực tiếp nhất cho việc đặt trạng thái drop *bên trong* DP thay vì "lọc event yếu trước rồi mới căn chỉnh".

**HiERO-StepG** (Ego4D Step Grounding 2026) là một Viterbi zero-shot trên N bước có thứ tự. Ràng buộc đơn điệu được viết theo giây: φ = 0 nếu t′ ≤ t + 1 s, ngược lại −10⁹. Mô hình không có phạt khoảng cách và không có drop. Riêng việc giải mã có thứ tự đã **nâng R1@0,3 từ 15,70 lên 35,28 (LaViLa) và từ 13,98 lên 33,95 (EgoVLP)**. Hybrid similarity đưa kết quả lên 41,00, mở rộng đỉnh cộng NMS lên 48,51, và hệ đứng thứ 2 bảng test với 56,27 ([HiERO-StepG](https://arxiv.org/pdf/2605.31227)). Lưu ý là backbone của HiERO được huấn luyện trên EgoClip, nên "zero-shot" chỉ đúng với nhãn step grounding. **BaGLM** (NeurIPS 2025) thì không phải DP. Nó là bộ lọc Bayes trực tuyến, với ma trận chuyển trạng thái T = Dᵀ lấy từ ma trận tiên quyết do LLM ước lượng. Riêng mô hình chuyển trạng thái nâng HT-Step R@1 từ **52,0 (LMM thô) lên 57,4** ([BaGLM](https://arxiv.org/html/2510.16989)).

Đọc chung, bằng chứng tách được hai hiệu ứng. Tách query thành sub-event chỉ lợi nhỏ (TFVTG, khoảng +2). Ép thứ tự bằng giải mã chính xác thì lợi lớn (HiERO, khoảng gấp đôi). Vì vậy TCR phải đo C1 trong hai điều kiện, "tách chuẩn" (oracle, dùng chính event của query) và "tách bằng LLM". Không paper căn chỉnh nào mô hình hóa khoảng thời gian không đều giữa các mẫu trong chi phí chuyển trạng thái. Drop-DTW và OTAM dùng chỉ số; HiERO chỉ dùng giây trong một mặt nạ cứng.

### Hệ tương tác: bốn bộ chấm chuỗi AIC 2025 không có một con số ablation

Truy vấn thời gian trong VBS có từ VIRET (2019). VIRET/SOMHunter nhận hai query có thứ tự. Điểm của frame o_i kết hợp score_{q1}[i] với max_{j=i+1..i+k} score_{q2}[j], với k tính theo số frame ([Gsteiger 2021](https://dbis.dmi.unibas.ch/teaching/studentprojects/evaluating-algorithms-for-temporal-queries-in-ad-hoc-video-retrieval/Thesis.pdf)). VISIONE (vô địch VBS 2024) lượng tử thời gian thành khoảng T giây và ghép cặp (a_i, b_j) cùng video cách nhau dưới **12 giây** ([VISIONE](https://openportal.isti.cnr.it/data/2023/486089/2023_486089.preprint.pdf?id=people______::8ca1b62a7655b1600a1c301e5b806fc5)). vitrivr (ICME 2020) dựng chuỗi theo lối tham lam ([vitrivr ICME20](http://lucaro.ch/papers/ICME20_vitrivr.pdf)). Luận văn Basel 2021 so sánh 8 thuật toán trên 109 query thời gian. Thuật toán thắng khi người dùng cho trước khoảng cách m là **IDA**: nó nhân điểm với exp(±0,1·(t−m)), t tính bằng *giây*, có "max time" để cắt ([Gsteiger 2021](https://dbis.dmi.unibas.ch/teaching/studentprojects/evaluating-algorithms-for-temporal-queries-in-ad-hoc-video-retrieval/Thesis.pdf)). Vậy "phạt theo giây + trần" đã là thực hành VBS từ 2021.

Mã hiện tại của **vitrivr-engine** (`TemporalSequenceAggregator`, commit d22860e, tháng 8/2026) là bộ nối chuỗi tham lam với các đặc điểm sau ([vitrivr-engine](https://github.com/vitrivr/vitrivr-engine/blob/HEAD/vitrivr-engine-query/src/main/kotlin/org/vitrivr/engine/query/aggregate/TemporalSequenceAggregator.kt)):

- Mỗi stage lấy ứng viên điểm cao nhất bắt đầu trong khoảng từ đầu phần tử trước tới **10 giây sau khi nó kết thúc**.
- Không có phạt khoảng cách.
- Stage thiếu ứng viên bị bỏ qua *miễn phí*; chỉ cần ít nhất 2 stage khớp.
- Điểm chuỗi là **max** của các stage, không phải tổng.

Hệ quả là một chuỗi khớp 2/5 có thể thắng chuỗi khớp đủ. Đọc mã còn thấy điều kiện gộp đoạn `(lastEndTime + PADDING_TIME) > startTime` có vẻ ngược với chú thích ngay trên nó. Đây là cách đọc của nhóm, không phải bug đã được xác nhận.

Bốn bộ chấm của AIC 2025 được tóm tắt dưới đây. Không bộ nào có ablation định lượng.

| Hệ | Cơ chế chấm chuỗi | Khoảng cách | Kết quả công bố |
|---|---|---|---|
| **DANTE** ([2512.13169](https://arxiv.org/abs/2512.13169)) | DP[i,t] = S[i,t] + max_{τ<t}(DP[i−1,τ] − λ(t−τ)); running max cho O(N·T); video xếp theo max_t DP[N,t]; unary chỉ là cosine BEiT-3; 4 keyframe mỗi scene TransNetV2 | λ = 0,001 hoặc 0,01 trên **chỉ số keyframe**; không trần, không drop, không span | Chỉ có xếp loại "Outstanding" |
| **MADTempo** ([2512.12929](https://arxiv.org/abs/2512.12929)) | Cặp biên (k₁, k̂_n) với 0 < t_{k_n} − t_{k₁} ≤ (n−1)·τ; BoundaryScore; beam cho các event giữa; ContextScore do LLM chấm trên caption và ASR | Trần cứng theo timestamp | 75,4 ở vòng loại |
| **Cascaded** ([2512.12935](https://arxiv.org/abs/2512.12935)) | Beam B = 8; SS = Σ s_i·e^{−α(t_i − t_{i−1})}; BLIP-2 ITM làm cổng nhân hậu kỳ | Giây (suy ra), α = 0,01; không trần | 76,4/88; chỉ 7 query TRAKE |
| **2512.06334** ([2512.06334](https://arxiv.org/html/2512.06334)) | Tổng kiểu RRF 1/(100 + r_i) có chỉ báo khoảng cách, neo vào f₁ | w_d = 10 frame | Không có số |

Có hai chi tiết đáng giữ. Thứ nhất, với α = 0,01/giây thì khoảng cách 34 giây vẫn giữ trọng số e^{−0,34} ≈ 0,71. Decay này yếu hơn nhiều so với lời mô tả "phạt mũ khi Δt > 10 s" của chính paper. Thứ hai là một quan sát thuật toán của nhóm, không lấy từ nguồn:

- **Dạng nhân s_i·e^{−αΔt} không dùng được running max của DANTE**, vì số hạng phụ thuộc đồng thời vào điểm tại t và vào θ_t − θ_τ. Cài đặt ngây thơ tốn O(N·T²). Có thể hạ xuống O(N·T·log T) bằng convex-hull trick, nhưng kỹ thuật này khó kết hợp với trần G. Có lẽ vì vậy mà tác giả phải dùng beam, và beam có thể cắt mất đường đi đúng.
- **Phạt cộng tính theo giây thì tách được**: DP[i,t] = U[i,t] − λθ_t + max_{τ<t, θ_t − θ_τ ≤ G}(DP[i−1,τ] + λθ_τ). Phần max là cực đại trên cửa sổ trượt, giải bằng deque đơn điệu, nên vẫn **chính xác O(N·T)**. Đây là lập luận kỹ thuật cụ thể nhất để TCR chọn phạt cộng tính thay vì decay nhân.

Hai baseline 2504.08384 và GRAB cần pivot do người dùng chọn. Với 2504.08384, từ pivot ta đi sang trái khi điểm với query₁ còn trên ngưỡng (tối đa 20 frame), sang phải tương tự với query₂, rồi chọn cặp có điểm tổng lớn nhất trong khoảng gap_C. Paper chỉ có nghiên cứu tình huống ([2504.08384](https://arxiv.org/abs/2504.08384)). ABTS của GRAB dùng cửa sổ W = {10, 15, 20} giây quanh pivot, độ tin cậy c_i = λ_s·s_i + λ_t·t_i với stability t_i = 1 − min(1, 2σ(...)), và chọn argmax *độc lập* cho điểm đầu và điểm cuối ([GRAB](https://arxiv.org/abs/2504.09298)). Bằng chứng nội bộ (span trung vị 2 giây, 15/48 query sụp về pivot) khớp với thiết kế này. Walk dừng ngay khi similarity text với sub-query giảm. Trên keyframe theo shot, hai keyframe kề nhau thường thuộc hai shot khác nhau, nên walk dừng sau 0–1 bước.

### Xếp hạng corpus và span: điểm đường đi cần chuẩn hóa, span cần neo vào shot

VCMR có giám sát kết hợp điểm video và điểm moment bằng phép nhân. CONQUER dùng luật "General" b̂·ê·r₁ và đạt TVR VCMR R@1 (IoU 0,7) **7,76, so với 7,18 của luật "Disjoint" chỉ dùng điểm trong video**. Tăng pool top-k từ 10 lên 100 gần như không thêm gì (7,76 → 7,77) ([CONQUER](https://arxiv.org/abs/2109.10016)). ReLoCLNet dùng δ = P_se·e^{γφ} với γ = 30 trên TVR ([ReLoCLNet](https://arxiv.org/abs/2105.06247)). DANTE xếp video theo max_t DP[N,t] mà không chuẩn hóa. Điểm đường đi tăng theo N và theo độ dài video (video dài có nhiều cơ hội hơn). Không paper AIC nào xử lý vấn đề này.

**SCAC** (Zhao et al., IEEE TIP 2026) là VCMR training-free và là mối đe dọa novelty gần nhất:

- **Đã xác nhận qua mã công khai**: LLM (llama3.1:8b) viết lại query, tách thành "atomic queries" và **sắp theo thứ tự thời gian**. Video được biểu diễn bằng event chain dựng từ caption BLIP của keyframe. Đặc trưng là BLIP ITM ở 3 fps. Mỗi query giữ top-200 video. Định vị bằng cửa sổ trượt có stride. Dữ liệu là ActivityNet Captions, Charades-STA và DiDeMo.
- **Chưa xác nhận được**: module `scac/core/retrieval.py` và `localization.py` chưa được phát hành, và toàn văn là closed access. Công thức "mean-variance joint scoring" và "profit-setback calibration" vì thế chưa biết ([DOI](https://doi.org/10.1109/tip.2026.3723243); [GitHub SCAC](https://github.com/cyanlll/SCAC); [prompt](https://github.com/cyanlll/SCAC/blob/HEAD/SCAC/prompts/query_event_chain.txt)).
- **Suy luận chưa kiểm chứng**: cách diễn đạt gợi ý một điểm tập hợp *không phụ thuộc thứ tự* (thưởng trung bình, phạt phương sai), không phải căn chỉnh có thứ tự.

Về span, pivot walk của 2504.08384 và ABTS đều sinh span quanh pivot. HiERO mở rộng đỉnh tới {0,6; 0,5; 0,4} × đỉnh, đệm tối thiểu 2 giây, rồi NMS 0,65; riêng bước này **thêm +7,5 R1@0,3** (41,00 → 48,51) ([HiERO-StepG](https://arxiv.org/pdf/2605.31227)). MADTempo lấy span từ keyframe khớp đầu tiên tới keyframe khớp cuối cùng. Moment-GPT thì cảnh báo ngược: dùng PySceneDetect làm bộ sinh span chỉ đạt **32,1 R1@0,5, so với 38,4** của bộ sinh span thích nghi, và phương pháp dựa trên shot "không phù hợp khi chuyển shot nhanh" ([Moment-GPT](https://arxiv.org/pdf/2501.07972)). Không paper nào đánh giá span IoU khi chỉ có keyframe shot. Cách hợp lý cho TCR là lấy [shot_start(keyframe khớp đầu), shot_end(keyframe khớp cuối)], mở rộng từng event theo độ tự-giống thị giác giữa các keyframe kề (kiểu CVPRW'26, đo bằng giây), và báo cáo kèm cận trên "oracle keyframe".

## C2: một bước diffusion với kernel bilateral, và một lỗi chuẩn hóa

C2 được đặc tả là s̃_t = (1−μ)s_t + μ·Σ_{t′∈N(t)} w_{tt′}s_{t′} / Σ w_{tt′}, với w_{tt′} = exp(−Δθ²/2σ_t²)·exp(−(1−cos(e_t,e_{t′}))/σ_v) trên khoảng 8 hàng xóm cùng video. Đặt S_rw = D⁻¹W là ma trận random-walk của đồ thị hàng xóm. Khi đó C2 chính xác là **f₁ = (1−α)y + αS_rw·y với α = μ và f₀ = y = s**, tức bước lặp Jacobi đầu tiên của diffusion. Diffusion hội tụ về f* = (1−α)(I − αS)⁻¹y và cực tiểu hóa Σ a_ij(f_i − f_j)² ([Iscen et al., CVPR 2017](https://arxiv.org/abs/1611.05113)). Toán tử lan truyền vì vậy không mới. Phần riêng của C2 nằm ở cách dựng đồ thị (tích kernel thời gian × thị giác, giới hạn trong cùng video, timestamp không đều) và ở tín hiệu được lan truyền (điểm text→keyframe thay vì điểm ảnh→ảnh).

### Diffusion và query expansion thường *hại* truy hồi liên phương thức

Iscen giữ cạnh k-NN tương hỗ "để xử lý nhiễu và ngoại lai" (khoảng 25 cạnh mỗi ảnh). Global diffusion đạt INSTRE/Oxf5k/Par6k 80,5/87,1/96,5 mAP, trong khi R-MAC + AQE đạt 70,5/89,6/95,3. Tức là diffusion *thua* AQE trên Oxford. Regional diffusion lên 89,6/95,8/96,9 ([Iscen 2017](https://arxiv.org/abs/1611.05113)). FSR biến diffusion thành lọc thông thấp trên đồ thị với xấp xỉ hạng thấp tính offline ([FSR](https://arxiv.org/abs/1703.06935)). k-reciprocal trộn khoảng cách Jaccard với khoảng cách gốc, d* = (1−λ)d_J + λd, λ = 0,3 ([Zhong et al.](https://arxiv.org/abs/1701.08398)). SuperGlobal lấy trung bình điểm gốc với điểm sau GeM pooling hàng xóm ([SuperGlobal](https://arxiv.org/abs/2308.06954)). GRAB đưa SuperGlobal vào hệ truy vấn khoảnh khắc nhưng mô tả nó cho *query ảnh* và không có ablation với query text ([GRAB](https://arxiv.org/abs/2504.09298)).

Bằng chứng quan trọng nhất cho C2 là bằng chứng âm tính trong **LeaPRR (SIGIR 2023)**, trên truy hồi ảnh–văn bản ([LeaPRR](https://arxiv.org/pdf/2304.12570)):

- Với VSE∞ trên Flickr30K T2I, α-QE giữ nguyên R@1 = 61,4 nhưng làm R@5 giảm từ 85,9 xuống 83,4. FSR làm R@10 giảm từ 91,5 xuống 87,6.
- Với DIME\*, diffusion DFS và FSR **giảm rSum từ 520,0 xuống 476,4 và 460,5**.
- Chỉ các reranker dùng láng giềng tương hỗ mới giúp: KRNN (64,6 R@1) và LeaPRR có huấn luyện (66,6).

PRF kiểu Rocchio trong không gian embedding cho CLIP/SigLIP cũng "không cải thiện đáng kể": CLIP-B trên Flickr30k từ 0,671 xuống 0,669 Hits@1 ([A Little More Like This](https://arxiv.org/html/2511.17255)). ICFRR là reranker training-free dựa trên hạng láng giềng gallery–gallery. Nó mạnh trên sketch→ảnh (Sketchy mAP 0,777 → 0,836) nhưng là ảnh↔ảnh và không có khái niệm thời gian ([ICFRR](https://arxiv.org/pdf/2303.17703)). Kết luận: lan truyền không giới hạn trong không gian đặc trưng dễ làm hại truy hồi liên phương thức. Hai lựa chọn của C2 là lan truyền *điểm* và giới hạn trong cửa sổ thời gian của cùng một video. Cả hai đều bảo thủ đúng hướng, nhưng vẫn phải chứng minh bằng số đo.

### Tiền lệ thời gian: làm mượt theo hàng xóm cùng video đã có từ 2011

Safadi & Quénot (CIKM 2011) chấm lại điểm shot theo "tính đồng nhất và bản chất của video chứa nó". Notebook TRECVID 2011 của Quaero xác nhận mức tăng **khoảng 18% trên TRECVID 2010 và 11–13% trên TRECVID 2008**, tập không đồng nhất hơn ([Quaero TRECVID 2011](https://www-nlpir.nist.gov/projects/tvpubs/tv11.papers/quaero.pdf)). Dạng generalized mean với cửa sổ Gaussian hoặc chữ nhật chỉ được xác nhận qua snippet. Multi-graph propagation (ACM MM 2007) lan truyền điểm text trên đồ thị độ giống thị giác và khái niệm giữa các shot, nhưng không có kernel thời gian ([Microsoft Research](https://www.microsoft.com/en-us/research/publication/video-search-re-ranking-via-multi-graph-propagation/)). Thuật toán 2 của 2504.08384 cộng điểm của hàng xóm: không trọng số, không chia cho số hàng xóm, không có điểm của chính frame. Cách này thiên vị shot dài một cách cơ học, và giải thích được lỗi đã quan sát là close-up một shot tụt từ hạng 1 xuống hạng 3 ([2504.08384](https://arxiv.org/abs/2504.08384)). ABTS dùng độ lệch chuẩn của độ giống embedding trong cửa sổ thời gian như một *đặc trưng cộng thêm*, không dùng làm trọng số lan truyền ([GRAB](https://arxiv.org/abs/2504.09298)). Prompts-to-Summaries nhân một hệ số thời gian với một hệ số đặc trưng *trên từng frame*, không phải trọng số cặp ([arXiv 2506.10807](https://arxiv.org/html/2506.10807v1)). Tìm kiếm bilateral filter trên chuỗi điểm chỉ ra kết quả trong miền pixel ([ví dụ](https://www.researchgate.net/publication/347627977_A_Temporal_Pre-Filter_For_Video_Coding_Based_On_Bilateral_Filtering)). Tuy vậy, có snippet (chưa lần ra paper cụ thể) nhắc tới "hybrid adjacency" thời gian × đặc trưng *học được* trong mô hình VTG có huấn luyện. Vì thế novelty của C2 phải dựa vào tính chất training-free, hậu kỳ, cấp điểm.

### Lỗi thiết kế: chuẩn hóa chỉ trên hàng xóm vô hiệu hóa khả năng bảo toàn biên

Bilateral filter cổ điển có mẫu trung tâm (trọng số 1) trong mẫu số. Nhờ vậy một mẫu cô lập, với mọi hàng xóm đều khác biệt, có Σw ≈ 0 và giữ gần nguyên giá trị của nó. C2 như đặc tả chuẩn hóa *chỉ trên hàng xóm* rồi trộn với μ cố định. Ở ranh giới shot thật, mọi w_{tt′} đều rất nhỏ, nhưng phép chia cho Σw thổi chúng trở lại thành một phân phối có tổng bằng 1. Kết quả là close-up vẫn bị kéo với **đủ trọng số μ** về trung bình của các hàng xóm khác biệt, và kernel thị giác chỉ còn tác dụng phân bổ lại trọng số giữa các hàng xóm. Đây đúng là lỗi "làm hại cảnh ngắn" mà C2 được thiết kế để sửa. Báo cáo trước đã khẳng định sai rằng shot ngắn "tự động quay về điểm frame đơn".

Có ba cách sửa, và hai cách đầu thực ra cùng một họ:

- **(a) Bilateral đúng dạng**: s̃_t = (s_t + Σw·s′)/(1 + Σw).
- **(b) μ hiệu dụng**: μ_t = μ·Σw/(Σw + κ). Viết lại (a) sẽ thấy nó bằng (b) với μ = 1, κ = 1.
- **(c) Không chuẩn hóa lại**: dùng ma trận random-walk có self-loop.

Với bất kỳ cách nào, kernel thị giác thực sự trở thành "edge stop". Phải ablate trực tiếp giữa chuẩn hóa-hàng-xóm và self-term, cắt theo độ dài shot.

Có thêm một hệ quả, do nhóm tự suy ra. Nếu điểm fusion tuyến tính theo từng embedding (cosine trên vector chuẩn hóa, trộn bằng tổng có trọng số, qua bất kỳ chuẩn hóa *affine* nào theo query như z-score hay theoretical min-max, kể cả NNN vốn trừ một bias theo item), thì C2 với trọng số cố định **tương đương với một phép DBA cục bộ theo thời gian, trọng số bilateral** trên embedding ảnh. Tính tương đương này mất đi với RRF hoặc chuẩn hóa phi tuyến. Nó có hai ý nghĩa. Paper phải thừa nhận quan hệ với DBA. Đồng thời C2 có thể tính trước thành vector đặc trưng, nên chi phí lúc query bằng 0. Hubness giữa các video (QB-Norm) không ảnh hưởng C2 vì đồ thị chỉ nằm trong một video. Việc đó phải xử lý ở tầng fusion.

## Caption ngắn kiểu COCO gần như trùng lặp với BEiT-3

### Caption chỉ thắng khi dài, và khi nền thị giác yếu

Hai paper training-free đo trực tiếp việc thay similarity frame–query bằng similarity query–caption. **VTG-GPT** dùng caption MiniGPT-v2 "describe in detail" ở 0,5 fps, so khớp bằng Sentence-BERT. Trên QVHighlights val, R1@0,5 tăng từ **45,59 (CLIP thị giác) lên 54,26**. CLIP-T trên caption chỉ đạt 52,85. Đổi MiniGPT-4 lấy MiniGPT-v2 làm R1@0,7 thay đổi khoảng 4,6 điểm ([VTG-GPT](https://arxiv.org/pdf/2403.02076)). Hàng/cột của bảng này được dựng lại từ pdftotext. **Moment-GPT** có hai kết quả trên Charades-STA ([Moment-GPT](https://arxiv.org/pdf/2501.07972)):

- Thay similarity thị giác BLIP-2 bằng caption MiniGPT-v2 + LLaMA-3 nâng R1@0,5 từ **34,8 lên 38,4**.
- Bảng 7 là tương tự gần nhất với vai trò ①. Chấm lại span bằng caption *ảnh* chỉ thêm **+2,2 R1@0,5** (30,2 → 32,4), trong khi caption *video* thêm +8,2 (→ 38,9).

Cả hai chỉ lợi khi hội đủ ba điều kiện: caption dài, chi tiết từ MLLM 7B; bộ mã hóa chỉ-văn-bản; nền thị giác yếu (CLIP-B, BLIP-2 ITC). Cả hai *thay thế* số hạng thị giác chứ không cộng thêm. TFVTG, phương pháp training-free mạnh nhất có sub-event theo thứ tự, **không dùng caption** ([TFVTG](https://arxiv.org/html/2408.16219)).

**Đính chính LexiCLIP.** Báo cáo trước viết LexiCLIP đạt "COCO R@1 67,4 so với 65,7 của SigLIP ViT-B/16". Đó là số **I→T** (truy hồi văn bản). Số **T→I** đúng, kiểm tra theo PDF, như sau ([LexiCLIP](https://arxiv.org/pdf/2509.19203)):

| Mô hình | Flickr30K T→I R@1/R@10 | COCO T→I R@1/R@10 |
|---|---|---|
| SigLIP ViT-B/16 | 74,6 / 95,6 | 47,8 / 81,0 |
| BLIP-2 (ViT-L) | 74,5 / 97,0 | 50,0 / 86,1 |
| LexiCLIP-ZS (0,3B) | 69,5 / 94,2 | **41,7** / 76,7 |
| LexiCLIP-FT (0,3B) | 79,2 / 97,4 | **52,7** / 84,5 |

Như vậy chỉ bản FT (finetune trên 1,5 triệu mẫu văn bản) mới vượt SigLIP-B/16. Bản zero-shot thua 6,1 điểm, dù dùng mô tả tới 256 token kèm danh sách đối tượng dạng JSON từ InternVL-2.5-8B. Ablation độ dài cho thấy **cắt mô tả xuống 64 token làm Flickr R@1 rơi từ 69,5 xuống 63,3**. Bỏ phần mô tả đối tượng làm COCO giảm từ 41,7 xuống 38,7. Mô tả phong phú cũng chỉ khớp 69% lớp đối tượng GT của COCO. Caption của ta khoảng 15 token, ngắn hơn 4 lần so với mức tệ nhất trong ablation. Trong khi đó nền thị giác của ta là SigLIP 2 so400m (COCO T→I 55,8 zero-shot) và BEiT-3 large finetune COCO (63,4) ([SigLIP 2](https://arxiv.org/pdf/2502.14786); [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3)). Caption đứng riêng sẽ nằm xa dưới cả hai.

### Xác minh top-K chỉ lợi khi tín hiệu mới mang thông tin mới

Khuôn mẫu gần nhất với ① là **CLIPRerank**: S_re = λ·M + (1−λ)·S trên top-1000, λ = 0,4 ([CLIPRerank](https://arxiv.org/pdf/2401.08449)). Mức lợi phụ thuộc vào base model:

- Base yếu: W2VV++ tăng từ 0,154 lên 0,183 infAP (+18,8%).
- Base đã chứa đặc trưng CLIP: LAFF chỉ +2,1% (0,221 → 0,226) và SEA +0,6%.

Hàng giữa của bảng được dựng lại từ bản trích PDF bị xáo cột. Tín hiệu chấm lại có ích theo tỷ lệ *thông tin mới* nó mang. BLIP-2 OPT-2.7B-**COCO** được huấn luyện trên caption COCO, và BEiT-3 cũng được finetune để truy hồi COCO. Giả thuyết hợp lý (cần đo) là cos(query, caption) tương quan cao với điểm BEiT-3, tức rơi vào vùng "LAFF/SEA" lợi thấp. Về hallucination, BLIP-2 OPT-2.7B có CHAIR_i/CHAIR_s = 1,7/2,6 trên COCO, nhưng **OpenCHAIR là 17,0%** ([MOCHa](https://arxiv.org/html/2312.03631v3)). Tức ngay trong miền COCO, khoảng 1/6 caption chứa đối tượng bịa theo từ vựng mở. Trên keyframe tin tức và nấu ăn tiếng Việt chưa có số đo nào.

### Unary caption trong DP không có tiền lệ, và có lẽ không phân biệt được bước

TFVTG và BaGLM là hai tiền lệ cho phần *thứ tự* của vai trò ②. Cả hai đều không dùng caption. Thứ phân biệt các bước giống nhau trong BaGLM là một *phán đoán tương phản* của LMM trên toàn bộ danh sách bước. Một caption chung chung như "a person is cooking food in a pan" sẽ cho cos gần như bằng nhau với "stir-fry the garlic" và "add the fish sauce", nên β·cos chỉ cộng một hằng số theo từng video. Một điểm caption *tương phản trong video*, ví dụ cos(sq_i, cap_t) − mean_j cos(sq_j, cap_t) hoặc softmax trên các event, sẽ loại được hằng số đó. Cách này tương tự static score "trong trừ ngoài" của TFVTG và cách chuẩn hóa trên các bước của BaGLM, nhưng là suy luận chưa được kiểm chứng. Chi phí không phải trở ngại. MiniLM-L6 mã hóa khoảng 18.000 câu/giây trên GPU, nên 335k caption mất dưới 1–2 phút ([SBERT](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)). Trở ngại là thông tin.

## Fusion: luật trộn quan trọng hơn hiệu chỉnh, và fusion thích ứng chỉ đáng vài điểm

Trên text IR, Bruch et al. (TOIS 2023) cho thấy **convex combination (CC) trên điểm đã chuẩn hóa vượt RRF(60)**, ví dụ NDCG@1000 trên MS MARCO 0,454 so với 0,425. CC gần như không phụ thuộc cách chuẩn hóa sau khi đã tinh chỉnh α. α hội tụ với dưới 5% dữ liệu huấn luyện, nhưng 5% của MS MARCO vẫn là hàng nghìn query ([Bruch et al.](https://arxiv.org/abs/2210.11934)). Không tìm thấy nghiên cứu nào fusion hai image encoder mạnh (SigLIP/SigLIP 2 với BEiT-3, EVA-CLIP hay BLIP) có số COCO/Flickr. Quan sát nội bộ rằng khoảng 40% top-1 đổi khi chuyển RRF ↔ max-norm cho thấy hai encoder bất đồng đủ thường ở đỉnh bảng. Max-norm ghim top-1 của mỗi encoder về 1,0, nên encoder có đỉnh phẳng được ảnh hưởng ngang encoder có đỉnh nhọn. RRF thì bỏ qua hoàn toàn khoảng cách điểm. Vì TCR đã tính điểm cho toàn corpus bằng matmul, z-score theo query trên cả 335k cosine là xác định rõ và không nhạy độ sâu top-K. Đây là lợi thế thật của bước [1].

Hiệu chỉnh hubness là bước thứ hai. **NNN** trừ bias b(r) = α·mean của top-k similarity giữa ảnh r và một ngân hàng query tham chiếu. Nó nâng COCO T→I R@1 của **SigLIP từ 47,15 lên 50,24 và BEiT-3 từ 47,62 lên 50,64**. Với ngân hàng tham chiếu ngoài phân phối, nên dùng α nhỏ (0,25–0,5) và k = 8–16 ([NNN](https://arxiv.org/html/2410.24114v1)). QB-Norm cảnh báo rằng inverted softmax với ngân hàng query lệch miền làm TT-CE+ trên MSR-VTT tụt **từ 14,9 xuống 11,6 R@1**. Dynamic Inverted Softmax chỉ chuẩn hóa khi top-1 là một hub đã biết, nên giữ nguyên 14,9 ([QB-Norm](https://arxiv.org/pdf/2112.12777)). DBNorm thêm ngân hàng gallery và đạt COCO CLIP 30,31 → 37,93 ([DBNorm](https://arxiv.org/html/2310.11612)). Mức +3 R@1 của NNN nhỏ hơn biên dao động giữa các luật fusion. Hiệu chỉnh vì vậy là hiệu ứng bậc hai, trừ ở tập con tin tức nhiều frame người dẫn chương trình. Checkpoint BEiT-3 trong NNN (47,6) rõ ràng không phải bản large-COCO (63,4).

Với ③, dự đoán hiệu năng truy vấn (QPP) trên truy hồi dày khá yếu. Trên PQPP, benchmark QPP text→ảnh duy nhất tìm được, tương quan Pearson với reciprocal rank chỉ **0,20–0,22**, và với P@10 là 0,45–0,47 ([PQPP](https://arxiv.org/html/2406.04746v2)). Mức lợi đo được rõ nhất cho trọng số thích ứng theo query là DAT. Nhưng DAT dùng một *LLM chấm top-1* của mỗi retriever, và đạt SQuAD P@1 0,8461 → 0,8740 so với α cố định. Trên tập "nhạy fusion" mức tăng là 0,6229 → 0,6976 ([DAT](https://arxiv.org/pdf/2503.23013)). Không tìm thấy nghiên cứu nào dùng độ đồng thuận giữa hai image encoder, hay giữa ảnh và caption, làm trọng số fusion. Tiền lệ gần nhất là reference-list QPP và Dense-QPP trong IR (chỉ ở mức snippet). Về tính bù trừ, SigLIP2-So400M vẫn gần mức ngẫu nhiên ở phép thử "confusion" của Auto-Comp: màu 50,5%, vị trí 57,7% ([Auto-Comp](https://arxiv.org/html/2602.02043)). BEiT-3 chưa được đo trên phép thử này. Fusion không sửa được điểm yếu mà cả hai encoder cùng có (binding, quan hệ, phủ định). Nó chỉ lợi ở chỗ tiên nghiệm dữ liệu huấn luyện của hai model khác nhau.

## Đánh giá: 79 query chỉ phân xử được khác biệt lớn

Không có benchmark công khai nào hội đủ ba tính chất: tìm kiếm cấp corpus, query tiếng Anh tự do nhiều sự kiện có thứ tự, và GT cho từng sự kiện. TVR là VCMR chuẩn mực (109K query, 21,8K video, metric R@k với IoU 0,5/0,7), nhưng frame bị kiểm soát truy cập ([TVR](https://arxiv.org/abs/2001.09099)). MAD chỉ phát hành đặc trưng CLIP, nên không tính lại được embedding SigLIP 2/BEiT-3 ([MAD](https://github.com/Soldelli/MAD)). ActivityNet Captions có đoạn văn nhiều câu có thứ tự kèm timestamp, nên là lựa chọn kiểm tra ngoài tốt nhất ([ActivityNet Captions](https://arxiv.org/abs/1705.00754)). Charades-STA test có 3.720 cặp nhưng mang thiên lệch vị trí đã được ghi nhận ([arXiv 2207.14698](https://arxiv.org/abs/2207.14698)). Các cài đặt corpus trên dữ liệu grounding một video đã có tiền lệ ([arXiv 2008.08716](https://arxiv.org/pdf/2008.08716)).

Về thống kê, các con số khoảng tin cậy dưới đây do nhóm tự tính. Với n = 79 và tỷ lệ quanh 50%, CI 95% là **±11 điểm**. Theo tầng: ±18 điểm với 31 query đơn cảnh, ±15 với 43 query chuỗi. Mỗi query tương đương 1,3 điểm R@1. Nếu hai biến thể chỉ khác nhau ở 10 query và chia 8–2, McNemar chính xác hai phía cho p ≈ 0,11, tức vẫn không có ý nghĩa. Mô phỏng của Urbano et al. (SIGIR 2019) cho thấy t-test và permutation test giữ đúng sai lầm loại I qua mọi metric. Hành vi của chúng ổn định ở 50 topic nhưng suy giảm ở 25. Bootstrap-shift thiên về p nhỏ, còn Wilcoxon và sign test không đáng tin ([Urbano 2019](https://arxiv.org/html/1905.11096v2); [Smucker 2007](https://dl.acm.org/doi/10.1145/1321440.1321528)). Sakai cho thấy muốn đủ power cho hiệu ứng nhỏ thường cần khoảng 200 topic hoặc hơn ([Sakai 2016](https://link.springer.com/article/10.1007/s10791-015-9273-z)).

Về gán nhãn, có ba cách xử lý độ mơ hồ ranh giới đã thành chuẩn:

- DiDeMo lấy 4 chú thích mỗi câu và yêu cầu dự đoán khớp ít nhất 2 ([DiDeMo](https://arxiv.org/pdf/1708.01641)).
- QVHighlights cho phép nhiều moment GT rời nhau và chấm bằng mAP ([QVHighlights](https://arxiv.org/abs/2107.09609)).
- Otani et al. gán lại nhãn và chỉ ra bất đồng đáng kể giữa người gán, cùng các baseline chỉ-dùng-thiên-lệch vẫn cạnh tranh được. Số liệu này lấy từ kiến thức nền, cần kiểm tra lại ([Otani 2020](https://arxiv.org/abs/2009.00325)).

## Đối chiếu từng đóng góp với công trình gần nhất

Bảng dưới là phần (b). Cột "không tìm thấy" chỉ nói điều khảo sát này không tìm thấy, không phải điều chắc chắn chưa ai làm.

| Đóng góp | Công trình gần nhất | Đã công bố | Không tìm thấy trong khảo sát | Phải thay đổi |
|---|---|---|---|---|
| **[1] Chấm toàn corpus bằng matmul + CC hiệu chỉnh hai encoder** | Bruch TOIS'23; NNN; QB-Norm/DBNorm; 2504.08384 (max-norm); Cascaded (min-max theo query) | CC > RRF trên text IR; NNN +3 R@1 cho SigLIP/BEiT-3; z-score/min-max là chuẩn | Nghiên cứu fusion SigLIP 2 + BEiT-3 có số đo; hiệu chỉnh hubness trên corpus keyframe có nhiều frame gần trùng | Không tuyên bố là đóng góp phương pháp; coi là *baseline mạnh đã hiệu chỉnh*. Thêm NNN theo từng encoder trước fusion, với ngân hàng query tham chiếu không lấy từ 79 query test. Chỉ dùng một α duy nhất. Tính oracle-of-two để biết cận trên |
| **[2] Xác minh top-K bằng caption ①** | CLIPRerank; Moment-GPT Bảng 7; VTG-GPT | Chấm lại top-k bằng tổng có trọng số là khuôn cũ; caption ảnh chỉ +2,2 R1@0,5; lợi co về 0,6–3% khi tín hiệu trùng lặp | Số đo "dual encoder mạnh + α·cos(query, caption sinh)" trên corpus keyframe | Hạ xuống số hạng ablate, **mặc định α = 0**. Mã hóa caption bằng encoder chỉ-văn-bản (MiniLM/BGE/E5) và chuẩn hóa trong top-K. Báo cáo tương quan Spearman với BEiT-3 như chẩn đoán trùng lặp. **Sửa số LexiCLIP**: T→I COCO là 52,7 (FT) / 41,7 (ZS) so với 47,8 (SigLIP-B/16); 67,4/65,7 là I→T |
| **[3] C2 bilateral temporal support** | Iscen diffusion; Safadi & Quénot 2011; multi-graph propagation 2007; 2504.08384 Alg. 2; ABTS; bilateral filter | Toán tử lan truyền (một bước diffusion); trộn lồi điểm gốc với điểm hàng xóm; làm mượt Gaussian theo thời gian trong cùng video (+11–18% TRECVID); tổng điểm hàng xóm cho KIS text | Kernel tích thời gian × embedding theo từng cạnh, áp lên điểm text→keyframe, giới hạn trong cùng video, trên timestamp shot không đều, tính trước thành đồ thị thưa | **Sửa lỗi chuẩn hóa**: thêm self term (s_t + Σws′)/(1 + Σw) hoặc μ_t = μΣw/(Σw + κ). Trình bày là "one-step joint-bilateral diffusion of cross-modal scores" và thừa nhận tương đương DBA khi fusion affine. Cân nhắc cạnh k-NN tương hỗ. Ablate: chỉ thời gian / chỉ đặc trưng / bilateral / k bước |
| **[4b] C1 DP N event có thứ tự** | DANTE; Drop-DTW; HiERO-StepG; vitrivr IDA và vitrivr-engine; 2512.12935; MADTempo; **SCAC** | DP O(N·T) running max, xếp video theo điểm đường đi, không pivot (DANTE); drop percentile (Drop-DTW); phạt theo giây (IDA, 2512.12935); trần theo giây (vitrivr-engine 10 s, VISIONE 12 s, MADTempo); LLM event chain có thứ tự + corpus (SCAC) | DP chính xác với phạt giây *cộng tính* (tách được nên giữ O(N·T)), trần qua deque trượt, drop state giá percentile theo video, unary fusion đa encoder, span neo shot; **đánh giá định lượng đối đầu** các bộ chấm chuỗi | Bỏ mọi tuyên bố "đầu tiên" về DP, phạt theo thời gian, không pivot, drop. Thêm chuẩn hóa điểm đường đi giữa các video (theo N, độ dài video, prior kiểu CONQUER). Thay span "đầu–cuối" bằng span neo shot + mở rộng theo tự-giống thị giác. Đo riêng tách oracle và tách LLM. Nêu SCAC là mối đe dọa chưa giải quyết |
| **Unary caption ② trong C1** | TFVTG; BaGLM (cả hai không dùng caption) | Thứ tự làm tăng lớn (HiERO ×2; BaGLM +5,4) | Unary thị giác + β·caption trong DP có thứ tự | Mặc định β = 0. Nếu dùng, chuyển sang điểm caption *tương phản trong video* (trừ trung bình trên các event hoặc softmax). Chạy trước chẩn đoán độ khác biệt caption giữa các keyframe kề |
| **③ Fusion thích ứng theo độ đồng thuận** | DAT (LLM judge); PQPP; reference-list QPP; Dense-QPP | QPP trên truy hồi dày tương quan 0,2–0,45; α thích ứng theo LLM +2–3 P@1 | Dùng độ đồng thuận thứ hạng giữa hai image encoder, hoặc giữa ảnh và caption, làm trọng số fusion | Định vị là "reference-list QPP với danh sách tham chiếu là modality kia". Dùng đồng thuận SigLIP 2 ↔ BEiT-3 thay vì ảnh ↔ caption. Tối đa 2 tham số, LOO-CV, báo cáo riêng tập nhạy fusion |

## Khung TCR sửa đổi: giữ lõi DP, sửa C2, hạ caption, thêm hiệu chỉnh

Bảng dưới là phần (c). Luồng mới được viết thành [1′] → [3′] → xếp hạng video → [4b′] → [2′], với ③ để giai đoạn sau.

| Hành động | Thành phần | Nội dung | Lý do |
|---|---|---|---|
| **Giữ** | [1] Matmul toàn corpus | Giữ vector điểm đầy đủ 335k × 2 encoder mỗi query | Giúp z-score xác định rõ, C2/C1 thành phép toán trên mảng, và mọi ablation chạy offline trên điểm đã tính |
| **Sửa** | [1′] Fusion | NNN riêng từng encoder (α 0,25–0,5, k 8–16, ngân hàng query tổng hợp) → z-score toàn corpus → CC một α; RRF(60) làm baseline không nhãn | Bruch: CC ≥ RRF và ít cần nhãn; NNN +3 R@1; QB-Norm: tránh IS khi ngân hàng lệch miền |
| **Sửa** | [3′] C2 | Self-term bilateral; cửa sổ theo giây; tùy chọn cạnh tương hỗ; tính trước thành DBA-cục-bộ nếu fusion affine | Sửa lỗi chuẩn hóa; LeaPRR cho thấy lan truyền không giới hạn làm hại; tương đương DBA cho chi phí 0 lúc query |
| **Thêm** | Xếp hạng video | So max / mean-top-k / log-sum-exp; cho C1: chuẩn hóa điểm đường đi theo N và theo thống kê điểm của video, cộng prior video kiểu CONQUER | CONQUER General > Disjoint (+0,58 R@1); DANTE không chuẩn hóa |
| **Giữ + sửa** | [4b′] C1 | Phạt giây cộng tính + trần G (deque); drop d_i = percentile điểm trong video; không pivot; lưu k-best; span = biên shot + mở rộng tự-giống thị giác với ngưỡng tách trung vị | Drop-DTW: giải chung > tham lam; HiERO: thứ tự ×2; Robust ZS-VTG: ngưỡng tự thích nghi bền hơn |
| **Hạ cấp** | [2′] Caption ① và ② | Số hạng ablate mặc định 0; encoder chỉ-văn-bản; ② dùng điểm tương phản trong video | Caption khoảng 15 token, trùng lặp COCO với BEiT-3, OpenCHAIR 17%; mọi tiền lệ có lợi đều cần caption dài hoặc nền yếu |
| **Thêm** | Vai trò thực tế của caption | (i) Nguồn **query tham chiếu tổng hợp trong miền** cho NNN; (ii) hiển thị/giải thích trong UI; (iii) chẩn đoán đồng thuận cho ③; (iv) tie-breaker khi α* > 0 trên dev | Dùng caption ở chỗ nó rẻ và không cần mang thông tin phân biệt |
| **Hoãn** | ③ Fusion thích ứng | Chỉ thử sau khi [1′] đã cố định; đặc trưng đồng thuận SigLIP 2 ↔ BEiT-3 (Jaccard/RBO top-k, cùng video top-1, biên top-1/top-2) | PQPP: tương quan yếu; 79 query không chịu nổi nhiều tham số |
| **Bỏ** | Các tuyên bố | "DP đầu tiên cho TRAKE", "phạt thời gian đầu tiên", "đầu tiên không pivot", "caption phân biệt bước giống nhau", "C2 bảo toàn biên" (khi chưa sửa) | Đã có DANTE, IDA/2512.12935, MADTempo, Drop-DTW; C2 bản gốc không có tính chất đó |

Sau sửa đổi, tuyên bố novelty còn lại có thể viết như sau: *"một bộ chấm chuỗi training-free, chính xác O(N·T), dùng phạt khoảng cách cộng tính theo giây và trần cứng theo giây trên keyframe shot không đều, có trạng thái drop giá percentile theo video, xuất span neo shot; đi kèm lan truyền điểm joint-bilateral một bước trong video; và một so sánh định lượng đầu tiên, trong phạm vi khảo sát, giữa các bộ chấm chuỗi của AIC/VBS"*. Caption xuất hiện trong paper như một kết quả âm hoặc trung tính được đo cẩn thận, không phải một modality lõi.

## Thiết kế thí nghiệm: tiền đăng ký ít so sánh, đo nhiều mô tả

Phần (d) bắt đầu từ kỷ luật chia dữ liệu. Chia 79 query phân tầng thành khoảng 25 dev và 54 test. Toàn bộ 5 query multi-event đặt ở test và chỉ báo cáo mô tả. Mọi α, β, μ, σ_t, σ_v, λ, G và percentile drop chỉ được tinh chỉnh trên dev; cấu hình được đóng băng bằng git hash, rồi test chạy đúng một lần. Vì 79 query này nhiều khả năng đã được nhìn thấy trong lúc phát triển, cần viết thêm một đợt 70–120 query mới *sau khi* đóng băng, để tiến gần mốc 150–200 query mà Sakai khuyến nghị. Nếu muốn dùng toàn bộ 79 query để chọn siêu tham số thì dùng cross-validation lặp 2-fold hoặc LOO, và chỉ báo cáo hiệu năng trên fold giữ lại.

### Baseline cần hiện thực lại

Tất cả baseline dưới đây chạy trên cùng vector điểm fusion [1′] để cô lập bộ chấm chuỗi.

| Mã | Baseline | Cách hiện thực | Khả thi |
|---|---|---|---|
| B1 | 2504.08384 dual-query walk + tổng điểm hàng xóm | Pivot = top-1 fusion (thay cho người dùng); walk ≤ 20 keyframe theo ngưỡng; cặp tốt nhất trong gap_C | Cao (đã có bằng chứng đầu tiên) |
| B2 | GRAB ABTS | Cửa sổ 10/15/20 giây quanh pivot; c_i = λ_s·s_i + λ_t·stability; argmax độc lập | Cao |
| B3 | DANTE | Phạt theo chỉ số, λ ∈ {0,001; 0,01}; không trần, không drop | Cao |
| B4 | Cascaded 2512.12935 | Beam B = 8, SS = Σ s_i·e^{−0,01Δt}, bỏ cổng ITM (hoặc thêm nếu có GPU) | Cao |
| B5 | MADTempo không LLM | Cặp biên + trần (n−1)·τ + beam cho event giữa; bỏ ContextScore | Trung bình (τ, M, b chưa công bố, phải tinh chỉnh trên dev) |
| B6 | vitrivr-engine greedy chaining | Đúng theo mã: cửa sổ 10 giây sau khi stage trước kết thúc, bỏ stage miễn phí, ≥ 2 stage, điểm = max; biến thể "điểm = tổng" | Cao (mã công khai) |
| B7 | HiERO-style Viterbi | Đơn điệu cứng với dung sai 1 giây, không phạt, không drop | Cao |
| B8 | TFVTG-style | Top-3 mỗi event, lọc thứ tự, lấy hợp | Cao |
| B9 | "SCAC-like" | Điểm tập hợp không thứ tự: mean − γ·std của max-similarity từng event trong video. **Đây là proxy do nhóm tự đoán, không phải SCAC**, phải ghi rõ như vậy | Thấp về tính trung thực; chỉ dùng để kiểm tra "thứ tự có quan trọng không" |
| — | Exquisitor 2026 | Công thức chưa biết; có thể thử proxy "RRF trên thứ hạng từng stage + ràng buộc thứ tự" | Không tái hiện được |

### Ablation

| Nhóm | Chuỗi ablation | Tầng query chính |
|---|---|---|
| Fusion | SigLIP 2 → BEiT-3 → RRF(60) → max-norm → theoretical-min-max CC → z-CC → + NNN; oracle-of-two | Tất cả 79 |
| C2 | Không làm mượt → trung bình ±2 (C-nbr) → tổng hàng xóm (B1) → chỉ thời gian → chỉ đặc trưng (tự-giống kiểu anchor) → bilateral chuẩn hóa-hàng-xóm (bản gốc) → **bilateral self-term** → k bước diffusion; cắt theo độ dài shot (< 2 giây, 2–5 giây, > 5 giây) | 31 đơn cảnh + unary của C1 |
| C1 | B1–B8 → DP phạt giây → + trần G → + drop → + unary C2 → + chuẩn hóa điểm đường đi → + span neo shot; tách oracle vs LLM; đảo thứ tự event (độ bền) | 43 chuỗi + 5 multi-event |
| Caption | α, β ∈ {0, tuned-trên-dev}; encoder CLIP-T vs MiniLM/BGE; cos tuyệt đối vs tương phản trong video; chẩn đoán Spearman với BEiT-3 và độ khác biệt caption giữa keyframe kề | Tất cả |
| ③ | α cố định vs α(q) = σ(a + b·đồng thuận); báo cáo riêng tập nhạy fusion (khoảng 40%) | Tất cả |

### Metric

Metric được chia theo tầng. Tầng video gồm R@1/5/10 và MRR. Tầng frame gồm hit R@K@±τ với τ ∈ {0; 2; 5} giây, kèm cờ "reachable" (có keyframe nào nằm trong [s−τ, e+τ] hay không) để tách lỗi encoder khỏi lỗi bộ trích keyframe. Tầng span gồm R@1@IoU {0,3; 0,5; 0,7} và mIoU, tính cả theo kiểu VCMR (sai video thì 0) lẫn *có điều kiện đúng video*, kèm cận trên oracle-keyframe. Riêng cho chuỗi có năm metric:

- tỷ lệ event trúng;
- tỷ lệ trúng đủ *và* đúng thứ tự;
- Kendall τ giữa thứ tự dự đoán và thứ tự GT;
- IoU trung bình từng event;
- hai metric trực tiếp từ bằng chứng đầu tiên: **độ dài span trung vị** và **tỷ lệ sụp về pivot** (15/48 ở B1).

Ngoài ra nên báo cáo điểm kiểu AIC mean-best-R@{1,5,20,50,100} làm chỉ số ứng dụng. Tuy nhiên luật TRAKE 2026 và dung sai ε chưa được công bố.

### Kiểm định thống kê

Mỗi con số chính đi kèm CI 95%: Wilson cho tỷ lệ, hoặc bootstrap BCa 10k lần trên query. So sánh cặp A với B dùng hai loại kiểm định. Metric nhị phân dùng McNemar chính xác (binomial trên cặp bất đồng). RR và IoU dùng paired permutation test (≥ 10k lần đổi dấu) hoặc paired t-test. Không dùng Wilcoxon hay sign test. Hiệu ứng được báo cáo bằng trung bình chênh lệch cặp kèm CI bootstrap, số thắng/hòa/thua theo query, và tùy chọn Cohen's d_z.

Chỉ tiền đăng ký **3–4 so sánh xác nhận**, kiểm soát bằng Holm–Bonferroni:

1. TCR đầy đủ so với B1 trên 48 query chuỗi, metric trúng-đủ-đúng-thứ-tự.
2. C1 phạt giây so với B3 (DANTE theo chỉ số).
3. C2 self-term so với không làm mượt, trên 31 query đơn cảnh.
4. CC hiệu chỉnh so với RRF(60).

Mọi thứ khác là thăm dò: dùng Benjamini–Hochberg hoặc chỉ báo cáo CI. Với 5 query multi-event, chỉ trình bày từng query một. Bản viết phải nói thẳng rằng phần lớn ablation nhỏ sẽ *không kết luận được*.

### Giao thức gán nhãn

Schema cho mỗi query gồm: `query_id`, `type`, danh sách mọi moment hợp lệ `(video_id, start, end)` tính bằng *giây trên timeline gốc*, và với query chuỗi thêm `events[{text, video_id, start, end}]` theo thứ tự, cùng `ambiguity_note` và `annotator`. Hướng dẫn gán như sau. Điểm đầu là frame đầu tiên mà nội dung mô tả nhìn thấy và nhận ra được. Chỉ snap theo shot khi mô tả khớp cả shot. Liệt kê mọi lần xuất hiện lặp lại. Đánh dấu "partial" cho các khớp một phần.

Người gán thứ hai làm độc lập ít nhất 20–30% số query: đủ 5 multi-event cộng một mẫu phân tầng. Báo cáo các chỉ số sau:

- mức đồng thuận cấp video;
- tIoU trung bình từng cặp người gán, và tỷ lệ cặp có tIoU ≥ 0,5 / 0,7;
- trung vị |Δstart| và |Δend|; con số này dùng để biện minh cho τ;
- mức đồng thuận về thứ tự.

Sau khi mọi hệ chạy xong, xét gộp top-10 của tất cả hệ theo lối pooling TREC để bổ sung GT bị sót, ghi lại số lần bổ sung, và làm việc này *trước* so sánh cuối cùng.

### Đánh giá tùy chọn trên benchmark công khai

Nên chọn ActivityNet Captions val_2, lấy mẫu nếu cần, và chạy theo cài đặt corpus: mỗi câu tìm trên toàn bộ video của split, đoạn văn là chuỗi có thứ tự. Báo cáo riêng hai phần: grounding một video (so với TFVTG 49,34/27,02/13,39/34,10) và VCMR corpus. Charades-STA test là lựa chọn thứ hai cho đơn cảnh. Bỏ MAD, còn TVR và Ego4D thì cần giấy phép. Ước tính của nhóm, chưa đo: khoảng 170–290k keyframe cho val_2, và SigLIP 2 + BEiT-3 mất khoảng 20–60 phút GPU cho mỗi 100k frame trên T4 hoặc RTX 4050. Nên bỏ caption vì BLIP-2 chậm hơn khoảng 10 lần. Nút thắt thật là tải video YouTube (nhiều ID đã chết) và chạy shot detection. Hãy đo throughput trên 1.000 frame trước khi cam kết.

## Danh sách đọc ưu tiên cho nhóm

Bảng dưới là phần (e).

| Ưu tiên | Tài liệu | Đọc để lấy gì |
|---|---|---|
| 1 | DANTE ([2512.13169](https://arxiv.org/abs/2512.13169)) | Công thức DP gốc, running max, λ theo chỉ số: baseline B3 |
| 1 | Drop-DTW ([2108.11996](https://arxiv.org/abs/2108.11996)), gồm phụ lục | Công thức drop, drop percentile, bảng "greedy vs joint" |
| 1 | Cascaded 2512.12935 ([arXiv](https://arxiv.org/abs/2512.12935)) và MADTempo ([arXiv](https://arxiv.org/abs/2512.12929)) | Beam decay theo giây, cặp biên + trần: B4, B5 |
| 1 | Iscen 2017 ([1611.05113](https://arxiv.org/abs/1611.05113)) và LeaPRR ([2304.12570](https://arxiv.org/pdf/2304.12570)) | Quy C2 về diffusion; bằng chứng lan truyền hại truy hồi liên phương thức |
| 1 | Robust ZS-VTG CVPRW'26 ([CVF](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/papers/Banditakkarakul_Towards_Robust_Zero-Shot_Video_Temporal_Grounding_CVPRW_2026_paper.pdf)) | Span anchor-expand, ngưỡng tách trung vị, cảnh báo tinh chỉnh trên test |
| 1 | Urbano et al. SIGIR 2019 ([1905.11096](https://arxiv.org/html/1905.11096v2)) | Chọn kiểm định cho 79 query |
| 2 | HiERO-StepG ([2605.31227](https://arxiv.org/pdf/2605.31227)) | Viterbi theo giây, ablation thứ tự, mở rộng đỉnh |
| 2 | vitrivr-engine `TemporalSequenceAggregator` ([mã](https://github.com/vitrivr/vitrivr-engine/blob/HEAD/vitrivr-engine-query/src/main/kotlin/org/vitrivr/engine/query/aggregate/TemporalSequenceAggregator.kt)) và luận văn Gsteiger ([PDF](https://dbis.dmi.unibas.ch/teaching/studentprojects/evaluating-algorithms-for-temporal-queries-in-ad-hoc-video-retrieval/Thesis.pdf)) | B6; IDA; tiền lệ "giây + trần" |
| 2 | Bruch TOIS'23 ([2210.11934](https://arxiv.org/abs/2210.11934)) và NNN ([2410.24114](https://arxiv.org/html/2410.24114v1)) | CC vs RRF; hiệu chỉnh hubness |
| 2 | Moment-GPT ([2501.07972](https://arxiv.org/pdf/2501.07972)) và LexiCLIP ([2509.19203](https://arxiv.org/pdf/2509.19203)) | Giới hạn thực tế của caption; số T→I đúng |
| 2 | TFVTG ([2408.16219](https://arxiv.org/html/2408.16219)) | Baseline B8, protocol public benchmark |
| 3 | CONQUER ([2109.10016](https://arxiv.org/abs/2109.10016)) | Luật kết hợp điểm video × điểm moment |
| 3 | QB-Norm ([2112.12777](https://arxiv.org/pdf/2112.12777)), CLIPRerank ([2401.08449](https://arxiv.org/pdf/2401.08449)), PQPP ([2406.04746](https://arxiv.org/html/2406.04746v2)), DAT ([2503.23013](https://arxiv.org/pdf/2503.23013)) | Rủi ro IS; lợi co lại khi trùng lặp; giới hạn của ③ |
| 3 | BaGLM ([2510.16989](https://arxiv.org/html/2510.16989)), Safadi & Quénot qua Quaero ([PDF](https://www-nlpir.nist.gov/projects/tvpubs/tv11.papers/quaero.pdf)), 2504.08384 và 2504.09298 | Tiền lệ thứ tự mềm; tiền lệ làm mượt thời gian; B1, B2 |
| 3 | DiDeMo, QVHighlights, Otani 2020, Sakai 2016 | Giao thức gán nhãn và cỡ mẫu |
| Cần xin toàn văn | SCAC ([DOI](https://doi.org/10.1109/tip.2026.3723243)) và Exquisitor VBS 2026 ([Springer](https://link.springer.com/chapter/10.1007/978-981-95-6963-2_27)) | Hai mối đe dọa novelty chưa giải quyết; thử qua thư viện trường hoặc email tác giả |

## Những điều chưa biết và cách chúng thay đổi kết luận

**SCAC** là ẩn số lớn nhất. Toàn văn là closed access, và module truy hồi/định vị cốt lõi chưa phát hành: git history chỉ chứa `.gitkeep`. Nếu "mean-variance joint scoring" thực ra là một căn chỉnh có thứ tự trên event chain, tuyên bố C1 sẽ phải thu hẹp thêm về phạt giây, trần, drop và keyframe thưa. Nếu nó là điểm tập hợp không thứ tự như cách diễn đạt gợi ý, C1 vẫn đứng. Công thức của **Exquisitor VBS 2026** ("sequence-chain + RRF") cũng chưa biết, vì chapter closed access và không có mã 2026. Chữ RRF gợi ý fusion theo hạng, nhưng không loại trừ được một tầng điểm.

Có một số điểm chỉ được kiểm chứng một phần:

- Chưa kiểm được nhiều chi tiết: dạng chính xác của Safadi & Quénot (chỉ có snippet); trọng số đồ thị thời gian của OMG-SSL; các mô hình VTG có huấn luyện dùng "hybrid adjacency" (chỉ có snippet); đơn vị Δt và cách trích keyframe của 2512.12935 (giây là suy ra); công thức phụ lục Drop-DTW cho drop cả hai phía; số liệu của ChatVTG và Enrich & Detect.
- Hàng giữa bảng VTG-GPT và CLIPRerank được dựng lại từ bản trích PDF bị xáo cột.
- Checkpoint BEiT-3 trong NNN chưa rõ, và chưa có số hubness cho SigLIP 2 hay BEiT-3 large-COCO.
- Điều kiện gộp đoạn có vẻ ngược trong vitrivr-engine là cách đọc mã của nhóm.

Còn một số điểm là ước tính hoặc suy diễn của nhóm, chưa có nguồn:

- Throughput GPU.
- Tương quan caption–BEiT-3.
- Tính tương đương DBA của C2.
- Nhận định rằng decay nhân chỉ giải chính xác được bằng convex-hull trick.

Luật TRAKE 2026 và dung sai ε cũng chưa được công bố.

Cuối cùng, ngoài keyword search thì khảo sát không rà hệ thống các số IEEE TIP/TMM/TCSVT năm 2026. Vì vậy vẫn có thể tồn tại một paper VCMR training-free khác chỉ đăng trên tạp chí, sau paywall.

## Kết luận

Khảo sát chuyển trọng tâm của TCR từ "phát minh thành phần" sang "đo cho đúng". Mỗi mảnh của C1 đã có chủ. Giá trị kỹ thuật thật nằm ở một quan sát nhỏ nhưng quyết định: phạt theo giây *cộng tính* là thứ duy nhất giữ được DP chính xác O(N·T) trên keyframe không đều. Decay nhân của hệ gần nhất phải dùng beam, và DANTE phải phạt theo chỉ số, vốn vô nghĩa khi độ dài shot thay đổi. C2 đã qua khảo sát nhưng lộ ra lỗi chuẩn hóa làm mất chính tính chất nó được tạo ra để có. Sau khi sửa, nó là một bước diffusion bilateral có thể tính trước, nghĩa là rẻ tới mức không có lý do gì để không ablate kỹ. Caption thì nên chuyển từ "tín hiệu xác minh" sang vai trò hậu cần: làm ngân hàng query tham chiếu cho NNN, làm giải thích trong UI, và làm chẩn đoán đồng thuận.

Hàm ý lớn hơn là về đóng góp. Trong một lĩnh vực mà bốn hệ AIC 2025 và cả dòng VBS không công bố một ablation định lượng nào cho bộ chấm chuỗi, một bảng so sánh đối đầu B1–B8 trên 48 query chuỗi, có CI và kiểm định cặp, có giá trị ngang một thuật toán mới. Mệnh đề "pivot walk sụp về pivot ở 15/48 query" là hạt giống của bảng đó. Nhưng bảng chỉ thuyết phục được nếu hội đủ ba điều: siêu tham số khóa trên dev, có thêm một đợt query giữ lại, và người đọc được nói thẳng rằng với 79 query thì chỉ những khác biệt cỡ 10–15 điểm mới có ý nghĩa.
