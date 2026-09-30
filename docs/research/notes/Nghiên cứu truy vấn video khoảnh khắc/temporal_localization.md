# Training-free temporal localization and ordered multi-event matching over keyframe similarity sequences

Context assumed by these notes: about 384 keyframes per video at shot-based (irregular) timestamps, SigLIP 2 and BEiT-3 per-keyframe embeddings, AIC TRAKE (several ordered event moments in one video), and KIS queries of the form "begins with... then... ends with...".

## 1. Zero-shot / training-free moment localization from CLIP-style similarity curves

### Takeaway
Current training-free VTG methods all work the same basic way. They compute a per-frame query-similarity curve, smooth or pool it, and then either threshold it into spans or score every (start, end) proposal with an "inside mean minus outside mean" contrast. The strongest reported Charades-STA mIoU rose from about 38 (Luo et al.) to 44.5 (TFVTG, ECCV'24), 45.7 (TAG, 2025) and 51.3 (DSE-VTG, 2026). All of them are evaluated on single-moment, short-video benchmarks sampled densely (e.g. 3 FPS), not on sparse shot keyframes.

### Cited Findings
- **TFVTG (ECCV 2024)**: training-free. An LLM splits the query into sub-events and states their temporal relation, and a VLM localizes each sub-event — [arXiv 2408.16219](https://arxiv.org/abs/2408.16219); [code](https://github.com/minghangz/TFVTG).
  - *Dynamic score*: Gaussian-filter the frame-query similarity to get Ŝ. Take D_i = Ŝ_i − Ŝ_{i−1}. For a transition segment (i,k), S^dynamic_{i,k} = Σ_{l=i..k} D_l if D_l > δ for every l in [i,k], else 0. δ = 5×10⁻⁴ on all datasets — [arXiv HTML](https://arxiv.org/html/2408.16219).
  - *Static score*: S^static_{k,j} = mean_{l∈[k,j]} S_l − mean_{l∉[k,j]} S_l — [arXiv HTML](https://arxiv.org/html/2408.16219).
  - *Final score*: S^final_{i,j} = max_{k∈[i,j]} (S^dynamic_{i,k} + S^static_{k,j}). Enumerating proposals makes this O(N²) or worse per query — [arXiv HTML](https://arxiv.org/html/2408.16219).
  - *Multi-sub-event fusion*: "simultaneously" relation → intersection of sub-event predictions; "sequentially" → union. Combinations where an earlier event starts after a later one ends are filtered out. Top-k = 3 per sub-event, 3 FPS sampling — [arXiv HTML](https://arxiv.org/html/2408.16219).
  - *Results*: Charades-STA R@0.3/0.5/0.7/mIoU = 67.04/49.97/24.32/44.51. ActivityNet = 49.34/27.02/13.39/34.10 — [arXiv HTML](https://arxiv.org/html/2408.16219).
  - *Ablation (Charades R@0.5)*: neither score = 42.32 (mIoU 31.61); dynamic only = 47.63 (mIoU 41.68); static only = 45.48; both = 48.01 (mIoU 43.37) — [arXiv HTML](https://arxiv.org/html/2408.16219).
  - *VLM choice*: CLIP 42.68, BLIP-2 48.01, InternVideo 44.60 R@0.5 — [arXiv HTML](https://arxiv.org/html/2408.16219).
- **TAG (arXiv 2508.07925, 2025)**: LLM-free.
  - *Temporal pooling*: sliding-window mean of frame features, c_i = (1/w) Σ f_j with w = 21.
  - *Temporal coherence clustering*: window r = 7, used to find change points and enumerate segments.
  - *Box-Cox similarity adjustment*: a_i = ((f_i·q)^λ − 1)/λ, used to spread out the skewed similarity distribution.
  - *Proposal score*: S_p = mean(a inside) − mean(a outside).
  - *Results*: Charades 67.82/48.58/26.67/45.69; ActivityNet 51.88/28.91/15.07/36.55 — [arXiv HTML](https://arxiv.org/html/2508.07925).
- **DSE-VTG (arXiv 2609.08850, 2026)**:
  - *Multi-scale fusion*: frame-level and clip-level similarities are combined. Clip scores are projected onto frames with a triangular kernel K(d) = max(0, 1 − |d|/δ), then s_t^fus = (1−α)s_t^frm + α s̃_t^clip.
  - *Query-level test-time adaptation*: a query-embedding offset Δ is fitted to pseudo-labeled frames with BCE; the VLM stays frozen.
  - *Results*: Charades 75.65/58.58/32.23/51.30; ActivityNet 54.83/31.97/16.15/37.93; QVHighlights R@0.5/R@0.7/mAP = 67.29/47.16/38.64.
  - *QVHighlights baselines in its table*: TFVTG 64.45/40.19/33.15 — [arXiv HTML](https://arxiv.org/html/2609.08850).
  - The same table lists TAG on QVHighlights at R@0.5 = 23.29, which looks implausibly low next to TAG's Charades numbers. It may be a transcription or setting issue. Verify before citing.
- **Moment-GPT (AAAI 2025)**:
  - *Pipeline*: LLaMA-3 rewrites the query, MiniGPT-v2 scores frames, and Video-ChatGPT with a span scorer picks spans.
  - *Span generator SG(S^f; η, κ, τ)*: build an inverse cumulative histogram with η = 10 bins and take threshold γ as the left edge of the first bin (scanning from the top) that holds ≥ κ = 7 moments. A span opens at a moment above γ and closes when τ = 5 consecutive moments fall below γ.
  - *Results*: QVHighlights test R1@0.5 58.3, R1@0.7 37.7, mAP@0.5 55.1; Charades 58.2/38.4/21.6; ActivityNet 48.1/31.1/14.9 — [arXiv HTML 2501.07972](https://arxiv.org/html/2501.07972); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32971).
- **VTG-GPT (Applied Sciences 2024)**: uses GPT to debias and rewrite the query, then matches it against image captions. A proposal generator with top-k and a "continuity threshold" turns the scores into segments.
  - Charades 59.48/43.68/25.94/39.81 and ActivityNet 47.13/28.25/12.84/30.49, as reported in the TFVTG and DSE-VTG tables — [arXiv 2403.02076](https://arxiv.org/abs/2403.02076); [TFVTG HTML](https://arxiv.org/html/2408.16219).
- **Point-to-Span (P2S, arXiv 2512.10363)**: zero-shot moment retrieval in hour-long videos.
  - *Adaptive span generator*: adaptive ratio τ_r = 0.5 + 0.5·sigmoid(σ_s), where σ_s is the standard deviation of the similarity signal. It sets the smoothing window (w = fps·τ_r) and the peak-expansion thresholds.
  - *Query decomposition*: an LLM splits the query into start-state and end-state sub-queries, and their peaks give a rerank bonus s_rerank = s_base + β·s_bonus.
  - *Injection*: the span generator is re-run inside regions where the start-state and end-state candidates overlap.
  - *Results*: MAD average 14.5 vs supervised RevisionLLM 14.4; MomentSeeker 39.3 vs CLIP 14.8 — [arXiv HTML](https://arxiv.org/html/2512.10363).

### Inferences
- All of these methods reduce to operations on a 1-D similarity curve: smoothing (Gaussian, box or triangular kernel), contrast (inside minus outside), and adaptive thresholding. Each ports directly to about 384 keyframes per video. With irregular shot timestamps, the kernels should be defined in seconds (or weighted by shot duration), not keyframe index.
- The TFVTG ablation suggests the VLM matters more than the proposal logic: BLIP-2 vs CLIP is worth +5.3 R@0.5. The dynamic "rising similarity" term adds about 5 R@0.5 over the static contrast alone.
- The static "inside minus outside" score favors long moments when the rest of the video is uniformly irrelevant. In corpus search across videos, this contrast should be normalized per video.

### Gaps
- No training-free VTG paper found that evaluates on sparse shot keyframes. All sample densely (1–3 FPS).
- Could not verify TAG's QVHighlights numbers. The low value in the DSE-VTG table is suspicious.
- Did not find exact VTG-GPT QVHighlights numbers in a primary source.

## 2. Ordered multi-event matching (DP / Viterbi / DTW / beam search) with gap penalties

### Takeaway
There are two directly relevant precedents from AIC 2025. **DANTE** (arXiv 2512.13169) is a Viterbi-style DP with a linear gap penalty, solved in O(N·T) per video with a running max. **MADTempo** (arXiv 2512.12929) uses beam search with a max-gap constraint plus an LLM context score. The alignment literature adds Drop-DTW, which drops outlier frames and steps and is the principled version of "not every frame belongs to an event", and Bayesian/HMM filtering such as BaGLM (NeurIPS 2025). None of these adds a stability term, meaning a reward for a matched keyframe's neighbors also matching.

### Cited Findings
- **DANTE (AIC HCMC 2025, TRAKE)**:
  - *Recurrence*:
    - Base: DP[1,t] = S[1,t]
    - Step: DP[i,t] = S[i,t] + max_{τ∈[s_v, t−1]} (DP[i−1,τ] − λ(t−τ))
    - Video score: DANTE[v] = max_t DP[N,t]
    - Keyframe sequences are recovered by backtracking.
  - *Speed-up*: running_max = max(running_max, DP[i−1,t−1] + λ(t−1)); DP[i,t] = S[i,t] + running_max − λt. This gives O(N·(e_v − s_v + 1)) per video.
  - *Similarity*: S is BEiT-3 cosine similarity from Milvus.
  - *λ values*: tuned in 0.001–0.01. Use 0.001 when event keyframes are 3–15 indices apart and 0.01 when they are 1–3 apart. The penalty is on keyframe index distance, not seconds. No max-window constraint.
  - *Results*: no DANTE-specific metrics or ablations. Only system-level ratings: "Outstanding" on TRAKE and "Excellent" overall — [arXiv HTML](https://arxiv.org/html/2512.13169).
- **MADTempo (AIC 2025, team AIO_Trinh)**:
  - *Similarity*: SimScore_i(k) = sim(z_{E_i}, z_k) with CLIP embeddings.
  - *Constraints*: all events in the same video, and 0 < t_{k_n} − t_{k_1} ≤ (n−1)·τ.
  - *Scores*:
    - EventScore = max over ordered paths of Σ_j SimScore_j(k_j)
    - BoundaryScore = SimScore_1(k_1) + SimScore_n(k̂_n)
    - ContextScore = (1/100)·f_LLM(metadata, context, events)
    - FinalScore = α·EventScore + (1−α)·ContextScore
  - *Search*: beam search over the top-M candidate segments.
  - *Results*: none reported in the text fetched — [arXiv HTML](https://arxiv.org/html/2512.12929).
- **Enhanced Multimodal Video Retrieval System (arXiv 2512.06334, AIC 2025)**: rank-based (RRF-like) scoring for three-scene sequences, pivoted on f₁.
  - Formula: 1/(100+r₁) + 1/(100+r₂)·𝟙[0 < f₂−f₁ < w_d] + 1/(100+r₃)·𝟙[0 < f₂−f₁, f₃−f₂ < w_d], with w_d = 10 frames.
  - AIC 2025 moved from 2-scene (2024) to 4-scene TRAKE queries. No numeric results — [arXiv HTML](https://arxiv.org/html/2512.06334).
- **vitrivr (ICME 2020, "Multi-Stage Queries and Temporal Scoring")**: supports temporal queries where segment A must precede segment B, and builds sequences of segments that satisfy the order — [IEEE](https://ieeexplore.ieee.org/abstract/document/9105954/); [PDF](http://lucaro.ch/papers/ICME20_vitrivr.pdf).
  - vitrivr at VBS 2022 added a new temporal-context query approach — [Springer](https://link.springer.com/chapter/10.1007/978-3-030-98355-0_44).
  - Exquisitor revisited temporal queries at VBS 2026 — [Springer](https://link.springer.com/chapter/10.1007/978-981-95-6963-2_27).
- **Drop-DTW (NeurIPS 2021, Dvornik et al.)**: aligns the common signal between two sequences while automatically dropping outlier elements from either one. It serves both as an inference-time step localizer and as a training loss.
  - CrossTask: Recall/Acc/IoU 49.7/74.1/36.9 with learned drop costs, vs SmoothDTW 43.1/70.2/30.5 and MIL-NCE 39.1/66.9/20.9 — [NeurIPS PDF](https://proceedings.neurips.cc/paper/2021/file/729c68884bd359ade15d5f163166738a-Paper.pdf); [code](https://github.com/SamsungLabs/Drop-DTW).
- **BaGLM (NeurIPS 2025, "Training-free Online Video Step Grounding")**:
  - *Method*: Bayesian predict/update filtering over steps. An LLM builds a step-dependency matrix, adjusted by "readiness" (prerequisites done) and "validity" (successors not yet done) computed from LMM progress estimates.
  - *Results*: HT-Step R@1 57.4 vs NaSVA 53.1; CrossTask Avg.R@1 59.8 vs MPTVA 47.9; Ego4D Goal-Step 43.3 vs 29.1. The baselines are trained and offline — [arXiv HTML 2510.16989](https://arxiv.org/html/2510.16989).
- **TFVTG's sequential handling** is only a post-hoc filter plus a union of per-sub-event top-k proposals, not a joint DP. Its order constraint alone gives R@0.5 43.01, the relation constraint alone 43.97, and both 44.12 (from a no-LLM base of 42.32) — [arXiv HTML](https://arxiv.org/html/2408.16219).

### Inferences
- DANTE is the closest prior work for component (b). It is exactly "DP over N ordered sub-queries over keyframes with a linear gap penalty, O(NT) via running max". Novelty over DANTE has to come from somewhere else:
  - a gap penalty in seconds for irregular shot timestamps, since DANTE penalizes index distance;
  - a max-gap window (MADTempo and 2512.06334 use one, DANTE does not);
  - span outputs per event rather than a single keyframe per event;
  - a stability or local-support term.
- Adding a unary stability term (e.g. S[i,t] replaced by S[i,t] + μ·local_support(i,t)) keeps DANTE's O(NT) recurrence unchanged. A pairwise term between consecutive matched keyframes would too, if it depends only on (τ, t) through a separable form.
- Letting each event span [a_i, b_i] instead of one keyframe gives a semi-Markov DP. That is O(N·T·L) with max span length L, or O(N·T) if span scores use prefix sums with a separable length penalty.
- Drop-DTW-style drop costs map to a "background" state. A per-video percentile of S could serve as a training-free drop cost. I recall the Drop-DTW paper suggesting percentile-based drop costs in the zero-shot setting, but I did not verify that in the fetched text.
- For "begins with... then... ends with..." KIS queries, MADTempo's BoundaryScore (first plus last event) and P2S's start-state/end-state decomposition are the closest analogues.

### Gaps
- Could not extract Drop-DTW's exact recurrence or its zero-shot (fixed drop cost) CrossTask numbers. The PDF was not machine-readable via fetch.
- Could not extract vitrivr's exact temporal scoring formula (PDF unreadable).
- No quantitative TRAKE results (e.g. per-query accuracy) published for DANTE, MADTempo or 2512.06334. None compare DP against beam search against rank fusion head-to-head.
- VISIONE and VIRET temporal-query formulas were not retrieved in this pass.

## 3. LLM query decomposition into ordered sub-events

### Takeaway
LLM decomposition helps, but only modestly, in training-free VTG: about +1.8 R@0.5 on Charades in TFVTG. It also depends on LLM quality (GPT-4 Turbo > GPT-3.5 > Gemini-1.0-Pro), and the TFVTG authors list LLM unreliability as a limitation. For long videos, P2S's start/end decomposition with an evidence bonus is reported to help find the moment.

### Cited Findings
- TFVTG Charades R@0.5: no LLM 42.32 → LLM 43.17 → LLM plus order/relation filtering 44.12. By LLM: Gemini-1.0-Pro 48.97, GPT-3.5 Turbo 49.23, GPT-4 Turbo 49.97 — [arXiv HTML](https://arxiv.org/html/2408.16219).
- TFVTG notes that LLMs are not always reliable at reasoning about sub-event order and relations, which can hurt performance — [TFVTG summary via arXiv/ECCV](https://arxiv.org/abs/2408.16219).
- P2S uses an LLM to split queries into ordered start-state and end-state sub-queries as separate evidence channels, reranked with a bonus β·s_bonus. It reports 14.5 on MAD (training-free) — [arXiv HTML](https://arxiv.org/html/2512.10363).
- Moment-GPT (LLaMA-3) and VTG-GPT (GPT) use the LLM to rewrite or debias the query rather than decompose it — [Moment-GPT](https://arxiv.org/html/2501.07972); [VTG-GPT](https://arxiv.org/abs/2403.02076).
- BaGLM uses an LLM to produce a step-dependency or transition matrix — [arXiv HTML](https://arxiv.org/html/2510.16989).
- MADTempo uses an LLM ContextScore mixed in with weight (1−α) — [arXiv HTML](https://arxiv.org/html/2512.12929).
- AIC 2025 systems commonly use LLM query rewriting. VBS 2024 systems also used GPT-4 for query rephrasing and summarization — [arXiv 2512.06334](https://arxiv.org/html/2512.06334).

### Inferences
- For TRAKE, the BTC already gives the events separately, so decomposition is mainly useful for KIS "begins/then/ends" queries. A cheap rule-based split on connectives ("then", "after that", "finally") may capture much of the benefit before trying an LLM. This is untested.

### Gaps
- No ablation found that isolates LLM decomposition in an interactive KIS/TRAKE setting.

## 4. Boundary refinement, smoothing, edge-preserving filtering and change points

### Takeaway
The smoothing used in training-free VTG is always linear and temporal-only: Gaussian (TFVTG), box/mean pooling (TAG), triangular kernel (DSE-VTG), or adaptive window (P2S). Boundaries come from derivative thresholds (TFVTG), clustering-based change points (TAG) or adaptive histogram thresholds (Moment-GPT). No training-free VTG paper found uses a bilateral filter, i.e. neighbor weights that are a product of time distance and visual-embedding similarity.

### Cited Findings
- **TFVTG**: Gaussian filter on the similarity curve, then a first-difference D_i with threshold δ = 5×10⁻⁴ to find "rising" transition segments — [arXiv HTML](https://arxiv.org/html/2408.16219).
- **TAG**: sliding-window mean over features (w = 21), temporal coherence clustering (r = 7) that yields change points and hence segment boundaries, and a Box-Cox transform of the scores — [arXiv HTML](https://arxiv.org/html/2508.07925).
- **DSE-VTG**: triangular-kernel projection of clip-level scores onto frames — [arXiv HTML](https://arxiv.org/html/2609.08850).
- **P2S**: smoothing window and peak-expansion thresholds that adapt to the signal's standard deviation — [arXiv HTML](https://arxiv.org/html/2512.10363).
- **Moment-GPT**: adaptive threshold with a τ = 5 "consecutive below-threshold" hysteresis to close spans — [arXiv HTML](https://arxiv.org/html/2501.07972).
- **Temporal-neighbor score re-ranking (pre-deep-learning)**: Safadi & Quénot, CIKM 2011, "Re-ranking by local re-scoring for video indexing and retrieval". Each shot's score is re-evaluated from its temporal neighbors in the same video (a generalized-mean rule), exploiting within-video homogeneity. About +18% on the TRECVID 2010 semantic indexing task — [ResearchGate](https://www.researchgate.net/publication/221614860_Re-ranking_by_Local_Re-scoring_for_Video_Indexing_and_Retrieval); [summary](https://liner.com/review/reranking-by-local-rescoring-for-video-indexing-and-retrieval).
- **Visual-affinity score propagation**: "Video search re-ranking via multi-graph propagation" (ACM MM 2007, Microsoft Research). Graphs with video shots as vertices and conceptual or visual similarity as edges; a topic-sensitive PageRank propagates text relevance scores — [Microsoft Research](https://www.microsoft.com/en-us/research/publication/video-search-re-ranking-via-multi-graph-propagation/); [ACM DL](https://dl.acm.org/doi/10.1145/1291233.1291279).
- **Bilateral filters in video** appear in pixel-domain denoising and coding pre-filters (e.g. "A Temporal Pre-Filter For Video Coding Based On Bilateral Filtering"). No use on query-score time series was found — [ResearchGate](https://www.researchgate.net/publication/347627977_A_Temporal_Pre-Filter_For_Video_Coding_Based_On_Bilateral_Filtering).

### Inferences
- **Closest prior work for (a), bilateral (time × visual similarity) neighbor aggregation of query scores:**
  1. Safadi & Quénot 2011: temporal-only neighbor re-scoring within a video.
  2. Multi-graph propagation 2007: visual-similarity-only graph propagation of scores across shots.
  3. kNN or diffusion re-ranking in image retrieval (graph message passing over a kNN similarity graph, see [GCN re-ranking arXiv 2306.08792](https://arxiv.org/pdf/2306.08792)). These are not temporal.
  4. TAG's temporal coherence clustering: it groups temporally adjacent frames by feature similarity, but uses that for segmentation, not for weighting score aggregation.
  5. Classic bilateral filtering (Tomasi & Manduchi), applied to images and pixels.

  The combination (weights w_{tt'} = exp(−|t−t'|²/2σ_t²) · exp(−(1−cos(e_t, e_t'))/σ_v) applied to query similarity scores over keyframes) does not appear in the retrieved VTG, VCMR or VBS literature. A novelty claim should be phrased as "not found in our survey" and should cite items 1–5.
- A bilateral weight is effectively an edge-preserving smoother. It avoids the Gaussian's known failure of bleeding scores across shot cuts, which matters because keyframes here are shot-based. At N ≈ 384 per video, a dense N×N kernel is cheap (about 150k entries per video), so a windowed version is optional.
- Shot boundaries (e.g. from TransNetV2, used by DANTE's system) already act as change points. Embedding change-point detection beyond that adds value mainly for merging consecutive similar shots into "scenes" that serve as span candidates.

### Gaps
- No systematic comparison found of Gaussian vs median vs bilateral smoothing on similarity curves for VTG.
- Did not locate a paper using formal change-point detection (e.g. PELT or kernel CPD) on CLIP embedding sequences for retrieval. TAG's clustering is the nearest.

## 5. Video-level vs frame-level score aggregation for corpus moment retrieval (VCMR)

### Takeaway
The common practice in VCMR and interactive systems is to rank videos by max frame similarity, sometimes averaged across modalities. The retrieved sources have little rigorous ablation comparing max, mean top-k and softmax (log-sum-exp) pooling in a training-free setting. DANTE and MADTempo rank videos by the best DP or beam path score, which is a structured max.

### Cited Findings
- Existing VCMR methods typically rely on frame-aware video retrieval, ranking videos by maximum frame similarity to the query. Event-aware VCMR (EventFormer) argues for event-level instead of frame-level units — [arXiv 2402.13566](https://arxiv.org/pdf/2402.13566).
- In one multi-video moment ranking system, the video-level score is the maximum clip-query cosine similarity, averaged across modalities — [arXiv 2301.13606](https://arxiv.org/pdf/2301.13606) (from search snippet; not verified in full text).
- The TVR baseline max-pools appearance features every 1.5 s to get clip features — [TVR ECCV 2020](https://arxiv.org/pdf/2001.09099).
- DANTE: video score = max_t DP[N,t], and the top-k videos by that score are returned with backtracked keyframes — [arXiv HTML](https://arxiv.org/html/2512.13169).
- MADTempo: FinalScore mixes EventScore (sum over ordered keyframes) with an LLM context score — [arXiv HTML](https://arxiv.org/html/2512.12929).
- Zero-shot VMR with off-the-shelf models (Diwan et al., PMLR 2023) computes frame-level query similarities from CLIP or captioning models — [PMLR](https://proceedings.mlr.press/v203/diwan23a/diwan23a.pdf).

### Inferences
- Max pooling is sensitive to single-keyframe false positives, especially with about 384 keyframes per video, where many chances inflate the max for long videos. Mean top-k or temperature-softmax pooling, or the inside-minus-outside contrast from TFVTG/TAG applied per video, are natural fixes. They should be benchmarked rather than assumed.
- For ordered queries, the DP path score is a structured aggregation. It should be normalized by N, and possibly by per-video score statistics (z-score), to compare across videos of different length.

### Gaps
- No primary-source ablation found comparing max vs mean-top-k vs softmax pooling for training-free VCMR. The EventFormer PDF could not be parsed to check its aggregation ablation.
