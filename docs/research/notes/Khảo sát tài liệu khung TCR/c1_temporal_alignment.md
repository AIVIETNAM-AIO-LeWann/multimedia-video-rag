# C1 scrutiny: training-free temporal grounding and ordered multi-event alignment over sparse shot keyframes

Scope: these notes add depth to `docs/research/notes/Nghiên cứu truy vấn video khoảnh khắc/temporal_localization.md` and do not repeat it. That file already covers the TFVTG, TAG, DSE-VTG, Moment-GPT, VTG-GPT and P2S formulas and numbers, the BaGLM summary, the DANTE recurrence, and the MADTempo scores. Here, formulas were re-read from full text wherever possible: local PDFs, or arXiv/CVF PDFs downloaded and parsed with pypdf. Any claim based only on a search snippet or abstract is flagged.

C1, as stated by the project, has these elements:
- **(a)** N ordered sub-queries, with **(b)** an exact dynamic program over keyframes that runs in **(c)** O(N·T).
- **(d)** Unary U[i,t] = smoothed image similarity (SigLIP 2 + BEiT-3) + **(e)** β·caption similarity.
- **(f)** A gap penalty measured in seconds, **(g)** a maximum-gap cap, and **(h)** a drop/skip state for weak events.
- **(i)** No user-selected pivot, and **(j)** outputs are ordered frames plus a span.
- **(k)** Runs on sparse, irregular shot keyframes (about 384 per video), with **(l)** corpus-level ranking of videos by path score.

## Q1. Zero-shot / training-free VTG (2023–2026): proposal formulations, sub-event combination, numbers, ablations

### Takeaway
Every training-free VTG method found scores one query (or 2–3 LLM sub-queries) over densely sampled frames (typically 3 fps) in a single, already-known video. They then turn a 1-D similarity curve into one span, using contrast, derivative, threshold or anchor expansion. None of them run a joint ordered DP over N sub-events.

A new CVPRW 2026 paper shows that TFVTG and TAG depend heavily on test-set hyperparameter tuning. It proposes a keyframe-anchor-and-expand scheme with an automatic threshold, which maps most closely onto C1's "span from sparse keyframes" requirement.

### Cited Findings
- **Robust ZS-VTG (Banditakkarakul, Chen, Gould, CVPRW 2026 GRAIL-V)**: training-free.
  - *Key-frame anchor*: k = argmax_i s^cm_i, the highest cross-modal (text–frame) score. The whole curve is then replaced by visual self-similarity to the anchor, s^im_j = v_k·v_j, which is "lower-variance".
  - *Segmentation*: τ = p-th percentile of s; expand outward from k until s < τ. That is, t_s = max{i<k | s_i<τ}+1 and t_e = min{j>k | s_j<τ}−1.
  - *Adaptive threshold*: start at τ₀ = the median of the scores. Split into s₁ ≤ τ and s₂ > τ, then iterate τ_{i+1} = (median(s₁) + median(s₂))/2 until convergence (a K-means-like update).
  - *Key-frame ensembling*: proposals from several VLMs are merged when their score difference is < d = α·stdev(s_diff) and their gap is < g seconds, with g adapted to video duration.
  - *Results (Charades-STA / ActivityNet, R@0.3/0.5/0.7/mIoU)*: 68.63/50.27/28.92/46.93 and 50.25/30.18/13.90/34.49. TFVTG 67.04/49.97/24.32/44.51 and TAG 67.82/48.58/26.67/45.69 on Charades, as reprinted there.
  - *Cross-dataset (tune on one dataset, test on another)*: TFVTG's mIoU drops 7.91 and TAG's drops 6.73, while this method drops at most 2.32. The ablation (Charades→ActivityNet) goes from 24.28 mIoU with neither adaptive component to 28.61 with adaptive merging only.
  - *TACoS*: TFVTG† 11.88 and TAG† 13.13 mIoU. — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/papers/Banditakkarakul_Towards_Robust_Zero-Shot_Video_Temporal_Grounding_CVPRW_2026_paper.pdf)
- **Same table, other zero-shot baselines (Charades mIoU / ActivityNet mIoU)**: ChatVTG 34.87/27.21, Luo et al. (WACV'24) 37.92/32.37, VTG-GPT 39.81/30.49. — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/papers/Banditakkarakul_Towards_Robust_Zero-Shot_Video_Temporal_Grounding_CVPRW_2026_paper.pdf)
- **TFVTG sub-event combination**, from the prior notes, verified there against arXiv HTML:
  - "Simultaneously" relation → intersection of sub-event proposals; "sequentially" → union.
  - Top-3 proposals per sub-event, with combinations that break the order filtered out.
  - No gap model and no joint scoring.
  - Adding LLM plus order/relation filtering takes R@0.5 from 42.32 to 44.12 (+1.8). — [arXiv 2408.16219](https://arxiv.org/html/2408.16219)
- **Enrich & Detect (ICCV 2025)** covers video paragraph grounding (N > 1 sentences → N windows) with an MLLM and claims an advantage in zero-shot evaluation. **Snippet only**: I did not read the formulation or numbers. It is MLLM-based and trained, not a DP. — [CVF](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Pramanick_Enrich_and_Detect_Video_Temporal_Grounding_with_Multimodal_LLMs_ICCV_2025_paper.pdf)
- **Multi-Sentence Grounding for Long-term Instructional Video (ECCV 2024)**: trained on narration alignment and step grounding, using ordered dense narrations or shuffled sparse steps. It reports zero-shot action-step localization on CrossTask. Supervised or weakly supervised, not training-free. **Snippet-level.** — [arXiv 2312.14055](https://arxiv.org/html/2312.14055v2)

### Inferences
- In the CVPRW'26 anchor-expansion method, the anchor is chosen automatically as the argmax, not by the user. It is still structurally a "pivot then expand" method, like the 2504.08384/2504.09298 walks. Its expansion runs on visual self-similarity to the anchor rather than on text similarity, which is the fix it proposes for noisy text curves.
- A narrow sub-event span is therefore a known failure of single-pivot methods. The CVPRW'26 authors address it with visual self-similarity and a median-split threshold. For C1 this is a directly usable, training-free span rule applied to each matched keyframe.
- The cross-dataset drops (TFVTG −7.9 mIoU) are a warning for C1's hyperparameters (β, λ per second, max gap, drop cost). They should be fixed on a development split and reported once, or chosen per video (percentiles, median split).

### Gaps
- No training-free VTG paper found that evaluates on sparse shot keyframes (about 384 per video) or on corpus-level ranking. All work on 1–3 fps frames within a known video.
- Enrich & Detect's zero-shot paragraph-grounding numbers were not extracted.

## Q2. Sequence alignment for step or procedure grounding: exact recurrences, drops, order, gaps, irregular sampling

### Takeaway
- **Drop-DTW** (NeurIPS'21) is the canonical exact DP with drop costs. Algorithm 1 in the paper drops only video frames; a supplementary variant also drops steps. Its per-instance percentile drop cost is training-free. It has no time-gap penalty and no maximum gap.
- **HiERO-StepG** (Ego4D challenge 2026) is a zero-shot Viterbi DP over N ordered step queries, with a monotonicity tolerance expressed in seconds. In its ablation, adding the ordered decoding roughly doubles R1@0.3 (e.g. 15.70 → 35.28). It still has no gap penalty and no drop state.
- **BaGLM** is online Bayesian filtering (order as a soft prior), not an alignment DP.

### Cited Findings
- **Drop-DTW (Dvornik et al., NeurIPS 2021): exact subsequence recurrence (Alg. 1)**, where C ∈ R^{K×N} is the step×frame cost and d^x_j the frame drop cost:
  - Initialisation: D⁺₀,₀ = 0, D⁺_{i,0} = D⁺_{0,j} = ∞; D⁻₀,₀ = 0, D⁻_{0,j} = Σ_{k≤j} d^x_k; D_{0,j} = D⁻_{0,j}.
  - Match: D⁺_{i,j} = C_{i,j} + min{D_{i−1,j−1}, D_{i,j−1}, D⁺_{i−1,j}}.
  - Drop frame: D⁻_{i,j} = d^x_j + D_{i,j−1}.
  - Best of the two: D_{i,j} = min{D⁺_{i,j}, D⁻_{i,j}}.
  - Objective: M* = argmin ⟨M,C⟩ + P_z(M)·d^z + P_x(M)·d^x. The general version, which also drops steps z_i, is given in the supplement. — [arXiv 2108.11996](https://arxiv.org/abs/2108.11996)
- **Drop-DTW costs**:
  - Symmetric match cost C_{i,j} = 1 − cos(z_i, x_j).
  - *Percentile drop cost* s = percentile({C_{i,j}}, p). It is set per instance; the step-localisation experiments use p = 30%, and 70% is used elsewhere.
  - A learnable drop cost f_ω on sequence means is also defined. — [arXiv 2108.11996](https://arxiv.org/abs/2108.11996)
- **Drop-DTW as a pure inference procedure** (frozen HowTo100M MIL-NCE features, no training; Acc/IoU on CrossTask, COIN, YouCook2):

  | Method | CrossTask Acc/IoU | COIN Acc/IoU | YouCook2 Acc/IoU |
  |---|---|---|---|
  | DTW | 11.2/10.1 | 21.6/18.3 | 35.0/31.1 |
  | OTAM | 19.5/11.6 | 26.5/19.5 | 43.4/34.7 |
  | LCSS | 50.3/4.1 | 47.0/4.5 | 43.4/9.0 |
  | Needleman–Wunsch | 68.8/9.5 | 52.1/7.4 | 50.1/11.7 |
  | Greedy drop + DTW | 60.1/13.8 | 45.0/18.9 | 54.3/34.1 |
  | Drop-DTW | 70.2/30.5 | 52.7/27.7 | 66.0/47.5 |

  The authors attribute the gain to solving drop and alignment jointly rather than greedily. — [arXiv 2108.11996](https://arxiv.org/abs/2108.11996)
- **Drop-DTW trained results (Table 1)**, for context: percentile drop 48.9/71.3/34.2 and learned drop 49.7/74.1/36.9 (CrossTask R/Acc/IoU). SmoothDTW 43.1, D3TW 43.2, OTAM 43.8 (supervised representation learning). — [arXiv 2108.11996](https://arxiv.org/abs/2108.11996)
- **OTAM** is described by the Drop-DTW authors as "extends D3TW with the ability to handle outliers strictly present around the endpoints". It skips only a prefix or suffix, not interior elements. **Needleman–Wunsch** rejects with a drop cost but forces one-to-one matches. **LCSS** thresholds the cost matrix before aligning. — [arXiv 2108.11996](https://arxiv.org/abs/2108.11996)
- **HiERO-StepG (Zenotto et al., Ego4D Step Grounding challenge 2026, arXiv 2605.31227)**:
  - *Objective*: max_P Σ_i S_{i,p_i} s.t. t(p_{i−1}) ≤ t(p_i) + τ, where t(·) is the timestamp in seconds and τ = 1.0 s is a tolerance for overlap.
  - *Recurrence*: D_{1,t} = S_{1,t}; D_{i,t} = S_{i,t} + max_{t′}(D_{i−1,t′} + φ(t′,t)), with φ = 0 if t′ ≤ t+τ and −10⁹ otherwise. Solved by Viterbi with backpointers.
  - *Unary*: "hybrid similarity" = α·local node similarity + (1−α)·coarse-cluster similarity, α = 0.7.
  - *Spans*: peaks are expanded left and right until similarity < {0.6, 0.5, 0.4} × peak, padded to at least 2 s, then IoU-NMS 0.65.
  - *Leaderboard*: Ego4D Goal-Step test R1@0.3 = 56.27, ranked 2nd.
  - *Ablation (LaViLa features, val R1@0.3)*: raw 14.96, HiERO baseline 15.70, +Viterbi 35.28, +hybrid similarity 41.00, +query-conditioned expansion+NMS 48.51. — [arXiv 2605.31227](https://arxiv.org/pdf/2605.31227)
  - *Caveat*: the HiERO backbone is trained on EgoClip (3.8M clip–text pairs). The approach is "zero-shot" only with respect to step-grounding labels. — [arXiv 2605.31227](https://arxiv.org/pdf/2605.31227)
- **BaGLM (NeurIPS 2025)**:
  - Transition matrix initialised as T = Dᵀ, where D is the LLM-estimated prerequisite matrix. Self-transitions T_{i,i} = 1 are allowed, as are transitions to steps with no prerequisites.
  - T is then modulated by "readiness" (the prerequisites' progress) and "validity".
  - It is online (predict/update); there is no global alignment and no gap-in-time model. — [arXiv 2510.16989](https://arxiv.org/abs/2510.16989)

### Inferences
- **Irregular sampling**: no alignment paper found models irregular inter-sample durations in its transition cost. HiERO-StepG uses seconds only in a hard monotonic mask. Drop-DTW and OTAM are index-based.
- **C1's drop state is the step-dropping (z-side) variant of Drop-DTW**, which is in the Drop-DTW supplement. C1 also implicitly drops frames, since any keyframe between matches is unused. C1 is therefore "Drop-DTW with dz finite, a one-keyframe-per-event matching (like Needleman–Wunsch on the event side), and a time-dependent transition cost". Drop-DTW's percentile drop cost is a ready-made, training-free way to set C1's drop cost per video, and it should be cited as such.
- Drop-DTW's Table 2 shows that a joint drop plus alignment decision beats greedy drop-then-align. This is the right methodological analogue for justifying C1's drop state over "threshold weak events, then DP".

### Gaps
- The supplement recurrence for dropping both sequences was not re-read. Only its existence (the D^{zx}, D^{z−}, D^{−x}, D^{−−} tables) was confirmed in the extracted text.
- No zero-shot step-grounding paper found that uses a transition penalty proportional to elapsed seconds.

## Q3. Multi-event / sequential queries in interactive video search (VBS, AIC): exact scoring

### Takeaway
Temporal queries in VBS go back to VIRET (2019). Systems have long used hard time windows (VISIONE: 12 s; VIRET: k frames). vitrivr had seconds-based exponential-decay temporal scoring with a maximum-time cap by 2021.

AIC 2025 produced four sequence scorers:
- **DANTE**: exact O(NT) DP with a linear penalty on keyframe index.
- **MADTempo**: boundary pairs plus beam search with a hard max gap in timestamp units.
- **Temporal-Aware Score Fusion** (2512.12935): beam search with exponential decay e^{−αΔt} in seconds.
- **2512.06334**: rank-fusion with a frame window.

None reports quantitative ablations. None combines an exact DP with a seconds-based penalty, a hard cap and an event-drop state.

### Cited Findings
- **VIRET / SOMHunter**: supports two ordered queries.
  - For a frame o_i, the score combines score_{q1}[i] with max_{j=i+1..i+k} score_{q2}[j], where k is a configurable frame threshold. It is evaluated per modality, and a default constant score is used if q2 is absent. Summarised in the Basel thesis (§5.1.1). — [Gsteiger 2021 thesis PDF](https://dbis.dmi.unibas.ch/teaching/studentprojects/evaluating-algorithms-for-temporal-queries-in-ad-hoc-video-retrieval/Thesis.pdf)
  - At VBS 2020, SOMHunter reused VIRET's temporal text queries "to describe a sequence of shots". — [TOMM VBS2020](http://lucaro.ch/papers/TOMM_VBS2020.pdf)
  - A search snippet describes the combination as a "max score product". **Snippet-level for the product form.** — [SOMHunter (ResearchGate)](https://www.researchgate.net/publication/344689021_SOMHunter_Lightweight_Video_Search_System_with_SOM-Guided_Relevance_Feedback)
- **VIREO**: queries at time T and T+1, "inspired by VIRET". — [TOMM VBS2020](http://lucaro.ch/papers/TOMM_VBS2020.pdf)
- **vitrivr (ICME 2020)**: k query containers form a temporal sequence when the segments (a) come from the same object, (b) lie within a certain time span, and (c) are the top-scoring segment for each container.
  - The algorithm sorts segments by timestamp. It builds candidate sequences incrementally from each starting segment, keeps the best, normalises by the number of containers, and keeps one sequence per start segment.
  - This is a greedy/heuristic construction, not an exact DP. — [ICME20 PDF](http://lucaro.ch/papers/ICME20_vitrivr.pdf)
- **vitrivr follow-up (Gsteiger 2021, Basel)**: compares 8 temporal scoring algorithms (STA, VITRIVR, A*, CLUSTER, IDA, LNA, NA, SQA) on 109 temporal queries.
  - *IDA*: the user specifies a desired distance m between consecutive containers. A segment's score is scaled by exp(l(t−m)) before the target time and exp(−l(t−m)) after it, with t measured in seconds and l = 0.1. Segments at the exact distance are not penalised. The score is normalised by the number of containers.
  - *Max time*: "The max time is there to cap the results that are too large".
  - *Winners*: IDA was judged best for queries with given distances. SQA, which needs no distance input, was implemented in Cineast.
  - *Deployment rule*: Cineast applies IDA when time distances are given and SQA otherwise. — [Thesis PDF](https://dbis.dmi.unibas.ch/teaching/studentprojects/evaluating-algorithms-for-temporal-queries-in-ad-hoc-video-retrieval/Thesis.pdf)
- **VISIONE (ICMR 2023; VBS 2024 winner)**: temporal query with two queries a and b.
  - Time is quantised into T-second intervals (e.g. T = 3). Each query keeps its best result per interval.
  - Pairs (a_i, b_j) from the same video with temporal distance < 12 s are shown, keeping one best pair per interval.
  - The aggregation formula is not given. — [VISIONE preprint](https://openportal.isti.cnr.it/data/2023/486089/2023_486089.preprint.pdf?id=people______::8ca1b62a7655b1600a1c301e5b806fc5); [VBS 2024 win](https://www.cnr.it/en/news/12517/the-visione-system-won-the-video-browser-showdown-2024)
- **DANTE (AIC 2025, team AIO_Owlgorithms, arXiv 2512.13169)**, re-read from the local PDF:
  - *Keyframes*: 4 per TransNetV2 scene, at k = K_{a+⌊i(b−a)/3⌋}, i ∈ {0,1,2,3}.
  - *Unary*: BEiT-3 cosine only.
  - *Recurrence*: DP[1,t] = S[1,t]; DP[i,t] = S[i,t] + max_{τ∈[s_v,t−1]}(DP[i−1,τ] − λ(t−τ)).
  - *Speed-up*: running max gives O(N(e_v−s_v+1)).
  - *Video score*: DANTE[v] = max_t DP[N,t], i.e. corpus ranking by best path. Keyframes are recovered by backtracking.
  - *λ*: 0.001 for ground-truth index gaps of 3–15 and 0.01 for gaps of 1–3. Units are keyframe indices.
  - Strict τ < t. No max gap, no drop, no span output, no caption term.
  - *Evaluation*: qualitative ratings only ("Outstanding" on TRAKE). — [arXiv 2512.13169](https://arxiv.org/abs/2512.13169)
- **MADTempo (AIC 2025, AIO_Trình, arXiv 2512.12929)**, re-read locally:
  - *Candidate segments*: for every keyframe k₁ matching E₁, the feasible ends are K_n(k₁) = {k_n : same video, 0 < t_{k_n} − t_{k₁} ≤ (n−1)·τ}, where t is the keyframe timestamp and τ is the maximum duration between consecutive events.
  - *End choice and ranking*: k̂_n = argmax SimScore_n. Segments are ranked by BoundaryScore = Sim₁(k₁) + Sim_n(k̂_n), and the top-M are kept.
  - *Context score*: an LLM rates the image captions of the boundary keyframes plus the ASR between them on a 0–100 scale, divided by 100.
  - *Intermediate events*: beam search (width b) over a filtered corpus, with t_{k₁} < … < t_{k̂_n} and a max gap τ.
  - *Scores*: EventScore = max Σ_j Sim_j(k_j); Final = α·EventScore + (1−α)·ContextScore. CLIP-LAION embeddings.
  - *Evaluation*: preliminary-round score 75.4; TRAKE rated "Excellent". No ablation. — [arXiv 2512.12929](https://arxiv.org/abs/2512.12929)
- **Unified Interactive Multimodal Moment Retrieval (AIC 2025, arXiv 2512.12935)**, full text read:
  - *Search*: beam search over K events × M candidates per event, beam B = 8, complexity O(B·K·M).
  - *Decay*: λ_i = exp(−α·Δt_i), where Δt_i = t_i − t_{i−1} is the time gap and α = 0.01. The authors state Δt < 2 s gives weight ≈ 1 and Δt > 10 s is "exponentially penalised". Units are implied to be seconds.
  - *Sequence score*: SS = Σ_i s_i·e^{−α(t_i−t_{i−1})}, additive. They argue against a multiplicative form Π s_i·λ_i.
  - *Final per-event score*: s_i·λ_i·b_i, where b_i is a BLIP-2 ITM rerank.
  - *Features*: first-stage BEiT-3 + SigLIP top-100. They also argue against the pivot-local ABTS (2504.09298).
  - *Evaluation*: qualification total 76.4/88. Only 7 TRAKE queries across rounds. No sequence-level ablation. — [arXiv 2512.12935](https://arxiv.org/abs/2512.12935)
- **2504.08384 (CVPRW 2025 IViSE)**, local PDF:
  - Keyframes come from TransNetV2 plus cosine dedup.
  - *Temporal search*: assumes the user-chosen input frame is correct. From it, walk left while similarity to query_1 stays above a threshold (up to 20 frames), and right likewise for query_2. Then choose the pair maximising combined similarity subject to distance ≤ gap_C.
  - *Evaluation*: case studies only, no quantitative evaluation of temporal search. — [arXiv 2504.08384](https://arxiv.org/abs/2504.08384); [CVF](https://openaccess.thecvf.com/content/CVPR2025W/IViSE/papers/Tran_Towards_Efficient_and_Robust_Moment_Retrieval_System_A_Unified_Framework_CVPRW_2025_paper.pdf)
- **2504.09298 (ABTS)**, local PDF:
  - Takes a user-provided pivot and splits the query into start and end sub-queries.
  - Window sizes W = {10, 15, 20} s around the pivot.
  - *Per-frame confidence*: c_i = λ_s·s_i + λ_t·t_i, with stability t_i = 1 − min(1, 2·σ({e_j·e_i : j ∈ N_i})).
  - *Selection*: argmax independently for start and for end. No joint order DP. — [arXiv 2504.09298](https://arxiv.org/abs/2504.09298)

### Inferences
- **Seconds-based gap penalty (f) is not new by itself.**
  - vitrivr IDA (2021) uses exponential decay on seconds of deviation from a user-given distance.
  - 2512.12935 (2025) uses exponential decay on seconds gap, but inside beam search.
  - What is not found is a seconds-based penalty inside an exact O(NT) DP. DANTE's exact DP penalises index distance, which on shot keyframes is effectively unevenly spaced in time.
- **The maximum gap (g) is common**: MADTempo in timestamps, VISIONE 12 s, VIRET k frames, vitrivr max time, 2512.06334 w_d = 10 frames.
- **"No pivot" (i) is shared with DANTE, MADTempo and 2512.12935.** It is novel only relative to the 2504.08384/2504.09298 lineage, and should be framed that way.
- **The drop state (h) was not found in any VBS or AIC scorer read.**
  - vitrivr normalises by the number of containers.
  - VIRET substitutes a default constant score when the second query is missing. This is a user-omitted query, not a data-driven drop.
- **A caption term inside the unary (e)**: MADTempo uses captions only through the LLM ContextScore on boundary keyframes, and 2512.12935 uses BLIP-2 ITM as a post-hoc multiplicative gate. A fused unary U = img + β·caption inside the DP was not found.

### Gaps
- There is no head-to-head quantitative comparison in the literature of DP vs beam vs pivot walk vs rank fusion for TRAKE-style queries. All AIC 2025 papers report only competition-level ratings or totals.
- The exact aggregation in VISIONE's pair score and SOMHunter's product form was not verified from the primary system papers.
- The vitrivr-engine (2024+) implementation of temporal scoring was not inspected.

## Q4. Training-free or lightweight VCMR: combining corpus video ranking with in-video localisation

### Takeaway
Supervised VCMR combines a video score with a moment score multiplicatively. CONQUER uses b̂·ê·r₁; ReLoCLNet uses P_se·e^{γφ}. CONQUER's ablation shows that the "general" product with the first-stage video score gives the best VCMR R@1.

A training-free VCMR framework (SCAC, IEEE TIP 2026) exists: LLM query event chains, video event chains and "mean-variance joint scoring". Its details were available only at abstract level. The AIC systems rank videos by best-path score (DANTE max_t DP[N,t]).

### Cited Findings
- **CONQUER (ACM MM 2021, supervised)**: re-ranks the top-k videos from a first-stage engine (HERO). Moment probability is the outer product of within-video normalised begin/end distributions, followed by NMS at IoU 0.7.
  - *Three ranking rules*: General b̂·ê·r₁; Exclusive b̂·ê·r₂ (its own video-scoring head); Disjoint b + e.
  - *TVR val R@1, IoU 0.7 (VCMR / VR / SVMR)*: General 7.76/29.01/22.84, Disjoint 7.18/26.67/21.84, Exclusive 7.02/29.29/20.10.
  - *Top-k pool (R1)*: 1 → 7.24, 10 → 7.76, 100 → 7.77. Speed rises from 3 to 142 ms.
  - *Oracle*: VCMR recall with HERO top-10 videos is 63.07.
  - *VS head*: max-pools clip features and then regresses. — [arXiv 2109.10016](https://arxiv.org/abs/2109.10016)
- **ReLoCLNet (SIGIR 2021, supervised)**: retrieves the top-K = 100 videos by the video–query similarity φ.
  - Final VCMR score δ = P_se × e^{γ·φ}, where P_se is the moment's start×end probability; γ = 30 (TVR) and 20 (ActivityNet). — [arXiv 2105.06247](https://arxiv.org/abs/2105.06247)
- **SCAC (Zhao, Li, Zhang, Wang, Hua; IEEE TIP 2026, "Training-Free VCMR via Synergistic Collaboration and Adaptive Calibration")**: training-free.
  - LLM-generated query event chains.
  - Video event chains from subtitle segmentation plus keyframe aggregation.
  - "Event-chain-based cross-modal retrieval with mean-variance joint scoring to suppress local mismatches and reinforce global consistency".
  - A "Synergy-Calibration Mechanism" refines boundaries via "profit-setback feedback".
  - Claims results comparable to supervised methods. The code README mentions ActivityNet Captions, Ollama and RefCap.
  - **Abstract and search-snippet level only**: the PubMed page was blocked and the README gave no details. — [DOI](https://doi.org/10.1109/tip.2026.3723243); [PubMed](https://pubmed.ncbi.nlm.nih.gov/42611658/); [GitHub](https://github.com/cyanlll/SCAC)
- **DANTE** ranks videos by max_t DP[N,t], the best ordered-path score. **MADTempo** ranks segments by α·EventScore + (1−α)·ContextScore. — [DANTE](https://arxiv.org/abs/2512.13169); [MADTempo](https://arxiv.org/abs/2512.12929)

### Inferences
- For C1, corpus ranking by the best DP path score (l) is already DANTE's design. The open question is **calibration across videos**: path scores grow with N and with video length (more chances), and neither DANTE nor the AIC papers normalise for this.
- CONQUER's result suggests a cheap improvement: multiply or add a video-level prior, such as the max single-event score or a first-stage rank, to the in-video path score. Its "general" rule beat the purely in-video "disjoint" rule by +0.58 R@1.
- SCAC's "mean-variance joint scoring" over event chains is the most conceptually similar training-free corpus-level method. It needs full-text review before any novelty claim about "training-free ordered-event VCMR".

### Gaps
- The SCAC full text was not accessible, so its event-chain scoring formula, whether it uses order or DP, and its datasets and numbers are unknown. This is **the most important unresolved item** for C1's novelty statement.
- No training-free VCMR ablation found comparing max, mean and path-score video aggregation.

## Q5. Span or boundary estimation from sparse keyframes

### Takeaway
The closest training-free rule is anchor-and-expand: take the argmax frame and expand while a similarity stays above an adaptive threshold. Examples are CVPRW'26 (visual self-similarity with a median-split threshold) and HiERO-StepG (fraction-of-peak thresholds of 0.6/0.5/0.4 with a 2 s minimum pad). Both were designed for dense frames or nodes. Nothing found targets irregular shot keyframes, where a natural span is the matched shot's [start, end] from the shot detector.

### Cited Findings
- **CVPRW'26**: expand from the anchor while s^im_j = v_k·v_j ≥ τ, with τ set by an iterative median split. Proposals merge when the gap < g seconds. — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/papers/Banditakkarakul_Towards_Robust_Zero-Shot_Video_Temporal_Grounding_CVPRW_2026_paper.pdf)
- **HiERO-StepG**: peak expansion to {0.6, 0.5, 0.4} × peak, padding to a minimum of 2 s, then NMS at IoU 0.65. Expansion plus NMS adds +7.5 R1@0.3 (41.00 → 48.51, LaViLa). — [arXiv 2605.31227](https://arxiv.org/pdf/2605.31227)
- **2504.08384**: walks up to 20 keyframes left and right of the pivot while text similarity stays above a threshold. Keyframes are TransNetV2 plus dedup. — [arXiv 2504.08384](https://arxiv.org/abs/2504.08384)
- **ABTS**: fixed windows of 10/15/20 s around the pivot, with a stability-weighted choice of the start and end frames. — [arXiv 2504.09298](https://arxiv.org/abs/2504.09298)
- **DANTE**: 4 keyframes per TransNetV2 scene. It outputs only matched keyframes, with no span. — [arXiv 2512.13169](https://arxiv.org/abs/2512.13169)
- **MADTempo's segment** is [t_{k₁}, t_{k̂_n}], the first to last matched keyframe. — [arXiv 2512.12929](https://arxiv.org/abs/2512.12929)
- **Keyframe or boundary-aware sparse sampling for MLLMs** (e.g. "Hybrid16", +26 mIoU over uniform sampling when boundaries are known) is sampling for trained MLLMs, not training-free span estimation. **Snippet only.** — [arXiv 2607.24570](https://arxiv.org/html/2607.24570)

### Inferences
- The median span of 2 s and the collapse to the pivot in 15/48 cases for the 2504.08384 walk (the project's own evidence) are consistent with its design. Its walk stops as soon as text similarity to a sub-query drops. With sparse shot keyframes, adjacent keyframes often belong to different shots, so the walk stops after 0–1 steps.
- For C1, a principled span would be [shot_start(first matched keyframe), shot_end(last matched keyframe)]. Per-event spans could come from CVPRW'26-style expansion over neighbouring keyframes' visual self-similarity, measured in seconds. This is not published for sparse keyframes, but each ingredient is.

### Gaps
- No paper found that evaluates span IoU when only shot keyframes, rather than frames, are available.

## Q6. LLM decomposition into ordered sub-events: benefit and failure modes

### Takeaway
Measured benefits are small. TFVTG gains +0.85 R@0.5 from decomposition alone and +1.8 with order/relation filtering (prior notes). TFVTG and TAG are also brittle across datasets.

Where the order constraint is enforced by an exact DP instead of post-hoc filtering, gains are large: HiERO-StepG's Viterbi roughly doubles R1@0.3 on Ego4D Goal-Step. There, the steps are given, not LLM-generated.

### Cited Findings
- **TFVTG**: 42.32 → 43.17 (LLM decomposition) → 44.12 (plus order and relation constraints) R@0.5 on Charades. The authors note that LLM order and relation reasoning is not always reliable. — [arXiv 2408.16219](https://arxiv.org/html/2408.16219) (via prior notes)
- **HiERO-StepG ablation**: +Viterbi ordering gives 13.98 → 33.95 (EgoVLP) and 15.70 → 35.28 (LaViLa) R1@0.3. The steps are human-given and ordered. — [arXiv 2605.31227](https://arxiv.org/pdf/2605.31227)
- **Cross-dataset fragility**: TFVTG loses 7.91 mIoU and TAG 6.73 when hyperparameters are not tuned on the test set. — [CVPRW'26](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/papers/Banditakkarakul_Towards_Robust_Zero-Shot_Video_Temporal_Grounding_CVPRW_2026_paper.pdf)
- **Other decomposition uses**:
  - MADTempo lets the user or an LLM compose the ordered events plus a context. — [arXiv 2512.12929](https://arxiv.org/abs/2512.12929)
  - 2512.12935 uses an agent (GPT-4o) for query expansion and decomposition. — [arXiv 2512.12935](https://arxiv.org/abs/2512.12935)
  - SCAC builds LLM "query event chains". — [DOI](https://doi.org/10.1109/tip.2026.3723243)

### Inferences
- The evidence separates two effects. Decomposition alone gives a small gain (TFVTG), while enforcing order jointly gives a large one (HiERO-StepG). C1's contribution should be measured on the DP, with oracle or human sub-queries (e.g. TRAKE's given events), separately from LLM splitting quality. Report both "oracle split" and "LLM split" conditions.
- **Failure modes to expect**:
  - The LLM produces more or fewer events than are visible. The drop state helps when there are too many.
  - The LLM gets the order wrong. The DP then fails hard, which argues for an ablation on order robustness.
  - Sub-events are not visually depictable (sound, intent). The caption or ASR term may help.

### Gaps
- No study found on how LLM split errors (wrong N, wrong order) propagate through an ordered DP in retrieval.

## Q7. What is already published vs open for C1 (element-by-element matrix of closest prior work)

### Takeaway
Every individual C1 element exists somewhere:
- exact ordered O(NT) DP: DANTE, HiERO-StepG;
- seconds-based penalty: vitrivr IDA, 2512.12935;
- hard max gap: MADTempo, VISIONE, vitrivr;
- drop state: Drop-DTW;
- no pivot: DANTE, MADTempo;
- sparse TransNetV2 keyframes: DANTE, MADTempo, 2504.08384;
- corpus ranking by path score: DANTE.

The unpublished part is the specific combination: an exact O(N·T) DP that uses a seconds-based additive gap penalty and a hard seconds cap on irregular shot keyframes, together with an event-drop state, a fused image + caption unary, and span output. The larger unpublished part is quantitative evaluation of any such sequence scorer against the pivot walk, beam and index-DP baselines. None of the AIC and VBS precedents report one.

### Cited Findings
The matrix below is compiled from the sources cited in Q2–Q5.

| Work (type) | (a) N ordered events | (b)(c) exact DP, O(NT) | (f) gap penalty unit | (g) max gap | (h) drop weak event | (e) caption term | (i) no user pivot | (k) sparse shot keyframes | (j) span output | (l) corpus ranking | Quantitative eval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DANTE 2512.13169 (training-free, AIC) | yes (N) | yes (running max) | linear, **keyframe index** | no | no | no (BEiT-3 only; OCR separate) | yes | yes (4 keyframes per TransNetV2 scene) | keyframes only | yes, max_t DP[N,t] | no (ratings) |
| MADTempo 2512.12929 (training-free plus LLM) | yes (n) | no (boundary pairs + beam) | none (hard only) | yes, (n−1)·τ, **timestamps** | no | captions via LLM ContextScore | yes | yes (TransNetV2) | first–last keyframe | yes (segments) | no |
| 2512.12935 (AIC) | yes (K) | no (beam B=8) | multiplicative e^{−αΔt}, **seconds**, α=0.01 | no (soft only) | no | BLIP-2 ITM gate (post-hoc) | yes | keyframes (type not stated) | not stated | yes | no |
| vitrivr IDA (Basel 2021) | yes (k containers) | no (incremental/greedy) | exp decay on deviation from **user-given seconds** | yes (max time) | no | no | yes | segments | sequence | yes | small internal eval (109 queries) |
| VISIONE 2023 | 2 only | no | none | 12 s window | no | no | yes | keyframes | pair | per video | no |
| VIRET/SOMHunter | 2 only | no | none | k frames | default score if q2 absent | no | yes | frames | no | yes | VBS logs |
| HiERO-StepG 2605.31227 (zero-shot, trained backbone) | yes (N) | yes (Viterbi; mask, naive O(N·T²) as written) | none (hard monotone, τ=1 s tolerance) | no | no | no | yes | graph nodes, not keyframes | yes (peak expansion) | no (single video) | yes (Ego4D) |
| Drop-DTW (NeurIPS'21) | yes (K steps) | yes, O(KN) | none | no | yes (frames; steps in supplement), percentile cost | no | yes | clips, dense | yes (step boundaries) | no | yes (CrossTask etc.) |
| TFVTG (ECCV'24) | 2–3 LLM sub-events | no (top-3 union/intersection) | none | no | no | no | yes | dense 3 fps | yes | no | yes |
| 2504.08384 / 2504.09298 | 2 (start/end) | no (pivot walk / windows) | none | gap_C / 10–20 s windows | no | no | **no (user pivot)** | yes (TransNetV2 + dedup) | yes | no | no (case studies) |
| SCAC (TIP'26, training-free VCMR) | event chains (LLM) | unknown | unknown | unknown | unknown | subtitles | yes | keyframe aggregation | yes | yes | yes (claimed) — **abstract only** |

### Inferences
- **Defensible novelty statement** (to be phrased as "not found in our survey"): an exact, linear-time ordered-event DP for corpus-scale retrieval over irregular shot keyframes. It combines:
  1. an additive gap penalty in seconds, which is separable, so DANTE's running-max O(NT) still holds when λ·(time(t) − time(τ)) replaces λ·(t − τ);
  2. a hard seconds cap, which stays O(NT) with a monotone-deque sliding-window maximum;
  3. an event-drop transition (DP[i,t] ← DP[i−1,t] − d_i, a Drop-DTW z-side drop with a per-video percentile cost);
  4. an image + caption unary;
  5. span output from shot boundaries.
- **Claims to avoid**:
  - "First DP for TRAKE": DANTE.
  - "First time-aware gap penalty": vitrivr IDA, 2512.12935.
  - "First without a pivot": DANTE, MADTempo.
  - "First drop/skip in ordered alignment": Drop-DTW.
- **Algorithmic observation**: 2512.12935's multiplicative decay s_i·e^{−αΔt} cannot use the running-max trick. It depends jointly on s_i and Δt, so an exact DP would cost O(N·T²) per video, which is why its authors needed beam search. C1's additive seconds penalty is what preserves exactness at O(NT). This is a concrete technical point in C1's favour over the closest seconds-based prior work.
- **Minimum baseline set for evaluation**:
  1. Pivot walk (2504.08384).
  2. DANTE (index-linear DP, λ ∈ {0.001, 0.01}).
  3. Beam with exponential decay (2512.12935, α = 0.01, B = 8).
  4. MADTempo-style boundary pair plus max gap.
  5. HiERO-style hard-monotone Viterbi (no penalty).
  6. C1 ablations: seconds vs index penalty, with/without cap, with/without drop, with/without caption term.
- The literature offers no benchmark for this, so the evaluation set (48 sequence queries) is itself part of the contribution.

### Gaps
- The SCAC full text is still unread. If it performs ordered event-chain DP alignment over keyframes, it becomes the closest training-free VCMR prior.
- The vitrivr-engine's current temporal scorer was not checked, and neither was whether any VBS 2025/2026 system (e.g. Exquisitor VBS 2026, "Temporal Queries …") uses an exact DP with time penalties. The Exquisitor VBS 2026 title was found, but its content was not read. — [Springer](https://link.springer.com/chapter/10.1007/978-981-95-6963-2_27)
- 2512.12935 does not state its keyframe extraction method or its Δt unit explicitly. "Seconds" is inferred from the text's examples (2 s, 10 s).
