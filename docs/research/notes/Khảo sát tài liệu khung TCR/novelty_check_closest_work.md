# Novelty check against the closest prior work for C1 (ordered-event DP) and C2 (bilateral score propagation)

Scope: this note adds to `c1_temporal_alignment.md` and `c2_context_reranking.md` in the same folder and does not repeat them. Where a paper was already read in full there (DANTE, MADTempo, 2512.12935, 2504.08384/2504.09298, TFVTG, HiERO-StepG, BaGLM, Robust ZS-VTG, Moment-GPT, VTG-GPT, Drop-DTW), this note only restates what is needed for the C1/C2 comparison and cites the primary source.

Each finding is labelled with the **source level** it rests on:
- **[full text]**: the paper PDF or HTML was read.
- **[source code]**: the published code was read.
- **[abstract]**: only the abstract or publisher metadata was available.
- **[snippet]**: only a search-engine snippet was available.

Reference definitions:
- **C1**: training-free.
  - The English query is split into N ordered sub-queries.
  - For each video, an exact DP over sparse, irregular shot keyframes aligns the sub-queries in order.
  - Unary = fused SigLIP 2 + BEiT-3 image similarity, optionally + β·caption similarity.
  - Additive gap penalty in seconds, plus a hard maximum-gap cap in seconds (sliding-window max keeps the DP at O(N·T)).
  - Drop-DTW-style event drop with a percentile cost.
  - No user pivot. Videos are ranked by path score. Spans are taken from shot boundaries.
- **C2**: a one-step propagation of text→keyframe relevance over neighbours in the same video.
  - Weights are joint-bilateral: a time-distance kernel × an embedding-similarity kernel.
  - It includes a self term and runs on irregular shot timestamps.

---

## Q1. SCAC (IEEE TIP 2026): event-chain construction, mean-variance joint scoring, calibration, ordering/DP, datasets, numbers

### Takeaway
The SCAC full text is **not obtainable**. The paper is closed access (IEEE), with no arXiv or repository copy, and the GitHub repo omits the core retrieval and localization modules.

What can be confirmed:
- SCAC is training-free VCMR on ActivityNet Captions, Charades-STA and DiDeMo, **not TVR**.
- An LLM rewrites each query, splits it into atomic sub-queries and **sorts them chronologically**.
- Videos become "event chains" built from BLIP captions of keyframes.
- Chains are matched with BLIP ITM features and "mean-variance joint scoring" over the top-200 videos.
- Localization is a stride-based window search with a "profit-setback" calibration.

It is therefore the closest competitor on "training-free, ordered sub-queries, corpus-level". There is **no evidence** that it uses an exact order-preserving DP, a seconds-based gap penalty, a max-gap cap, an event-drop state, or any bilateral score smoothing. None of these can be ruled out without the full text.

### Cited Findings
- **Bibliographic record** [abstract]:
  - Authors: Jialong Zhao, Huafeng Li, Yafei Zhang, Wen Wang, Changchun Hua.
  - IEEE Transactions on Image Processing, vol. 35, pp. 8952–8965 (2026), DOI 10.1109/TIP.2026.3723243, IEEE document 11658824.
  - OpenAlex lists it as `oa_status: closed` with no repository full text, and Semantic Scholar's openAccessPdf is "CLOSED". — [DOI](https://doi.org/10.1109/tip.2026.3723243); [IEEE Xplore](https://ieeexplore.ieee.org/document/11658824/); [OpenAlex API](https://api.openalex.org/works/doi:10.1109/tip.2026.3723243); [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TIP.2026.3723243?fields=title,abstract,openAccessPdf)
- **Abstract content** [abstract]:
  - A "Query Event Chain Generation module … leverages large language models to transform complex textual queries into structured event chains".
  - A "Video Event Chain Generation module represents videos as semantically coherent event chains through subtitle segmentation and keyframe aggregation".
  - "Event-Chain-Based Cross-Modal Retrieval with mean-variance joint scoring to suppress local mismatches and reinforce global consistency".
  - "During localization, a Synergy-Calibration Mechanism dynamically refines temporal boundaries via profit-setback feedback".
  - It claims results "comparable or superior … to supervised counterparts under training-free conditions". — [Semantic Scholar record](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TIP.2026.3723243?fields=title,abstract)
- **Query event chain prompt** [source code]. The repo's `prompts/query_event_chain.txt` asks the LLM (config: `llama3.1:8b-instruct-fp16`, temperature 0.7, via Ollama) to:
  - (1) rewrite the query into several paraphrases;
  - (2) decompose it into unique "atomic queries", each "a single, simple action or piece of information";
  - (3) "infer the likely sequence of events … Sort the atomic queries into a chronologically ordered list".
  - The output is a plain list of strings, e.g. 4 atomic events for an ActivityNet caption. So SCAC does produce **N ordered sub-queries**, as C1 does. — [SCAC prompt](https://github.com/cyanlll/SCAC/blob/HEAD/SCAC/prompts/query_event_chain.txt); [config](https://github.com/cyanlll/SCAC/blob/HEAD/SCAC/configs/activitynet.yaml)
- **Pipeline stages and features** [source code]. `main.py` declares these stages: query-chain, format-query-chain, caption, extract-text, extract-video, extract-video-chain, retrieve, evaluate.
  - Captions come from `blip-image-captioning-large` run on `keyframes/` and are written to `video_captions.jsonl`.
  - Video features are BLIP ITM (`blip_image_text_matching`, large, "legacy_qformer" text encoder) at `fps: 3` for both full videos and `video_event_clips`.
  - Retrieval keeps `top_k: 200` videos per query.
  - Localization parameters: `stride` (ActivityNet 50, Charades 23, DiDeMo 21), `max_stride_factor` (1.0 / 0.5 / 0.5), `calibration_step: 1`, `overlap_threshold: 0.3`, `cc: 1`.
  - Evaluation is standard VCMR: R@10 and R@100 at IoU 0.3/0.5/0.7, counted as correct only if the target video is in the top-n **and** the localized IoU ≥ threshold.
  - The code also runs an "oracle localization" evaluation. — [main.py](https://github.com/cyanlll/SCAC/blob/HEAD/SCAC/main.py); [evaluation.py](https://github.com/cyanlll/SCAC/blob/HEAD/SCAC/scac/evaluation.py); [config.py](https://github.com/cyanlll/SCAC/blob/HEAD/SCAC/scac/config.py)
- **Unreleased modules** [source code]. The modules that implement the scoring and localization are **not in the repository**: `scac/core/retrieval.py`, `scac/core/localization.py` and `scac/preprocess/*`.
  - The git history (27 commits, 2026-08-08 to 2026-08-11) only ever contained `.gitkeep` placeholders for `scac/core/` and `preprocess/`.
  - The README only covers data preparation, including "refer to … RefCap" for the VLM. — [SCAC repo](https://github.com/cyanlll/SCAC)
- **Datasets** [source code]. `config.py` accepts only `act` (ActivityNet Captions), `cha` (Charades-STA) and `didemo`. No TVR config exists. — [config.py](https://github.com/cyanlll/SCAC/blob/HEAD/SCAC/scac/config.py)

### Inferences
- **C1 elements SCAC does have**: training-free; LLM decomposition into N chronologically ordered sub-queries; corpus-level ranking (top-200 videos, then localization); caption use, but as the *video-side representation* (keyframe captions → event chains) rather than as a β-weighted unary term.
- **C1 elements SCAC seems not to have**, inferred from the config rather than confirmed:
  - The config works on **dense 3 fps BLIP features** and a **stride-based window** localizer, not on sparse shot keyframes.
  - The config has no gap-penalty, max-gap or drop parameter.
  - "Mean-variance joint scoring" reads like chain-level aggregation of per-event similarities (mean rewarded, variance penalised: "suppress local mismatches, reinforce global consistency"). That would be an **order-agnostic set score**, not an order-preserving alignment. This is plausible but **unconfirmed**.
- **C2**: nothing in the abstract or code suggests same-video bilateral score propagation.
- **Risk level**: SCAC is the paper a reviewer would most likely cite against "training-free ordered-event corpus retrieval". C1's claim should be narrowed to the **exact order-constrained DP with seconds-based gap penalty, hard cap and drop state over sparse shot keyframes**. SCAC should be cited as the training-free LLM-event-chain VCMR baseline, and ideally compared against on ActivityNet or Charades if the full code is released.

### Gaps
- SCAC's mean-variance formula, any order/DP handling, the "profit-setback" calibration rule and all reported numbers are unknown. The full text is paywalled (IEEE Xplore returned nothing to scripted access; PubMed blocked), and the core code is unreleased.
- Whether "subtitle segmentation" means ASR subtitles or generated captions is ambiguous. The config only shows generated BLIP captions.

---

## Q2. Exquisitor VBS 2026 temporal queries and the current vitrivr-engine temporal scorer

### Takeaway
- **Exquisitor VBS 2026 ("Temporal Queries Revisited")** proposes a "sequence-chain method combined with reciprocal rank fusion (RRF)". Only the abstract is accessible: it is a closed Springer chapter with no public code for the 2026 version. Its formula is unknown, but RRF implies **rank-based** rather than score-based fusion.
- **vitrivr-engine (current `main`, commit d22860e, 2026-08-17)** implements temporal queries in `TemporalSequenceAggregator`. It is a **greedy** chainer with:
  - a hard **10-second** window after the previous stage's end;
  - **no gap penalty**;
  - free skipping of stages (at least 2 stages must match);
  - sequence score = **max** of the stage scores.

It shares only "ordered stages + seconds + hard max gap" with C1.

### Cited Findings
- **Exquisitor at VBS 2026: Temporal Queries Revisited** [abstract]:
  - Authors: Khan, O.S.; Sharma, U.; Marcelino, G.; Rudinac, S.; Jónsson, B.Þ. MMM 2026, LNCS pp. 245–251.
  - Abstract: "tasks with temporal components highlighted the need for improved temporal querying. To address this, we propose a novel sequence-chain method combined with reciprocal rank fusion (RRF)". It also adds in-video search for QA tasks.
  - The page states "This is a preview of subscription content", and OpenAlex marks it closed. — [Springer](https://link.springer.com/chapter/10.1007/978-981-95-6963-2_27)
- **Exquisitor code** [source code]. The public ExquisitorServer repo (last push January 2024, author Ok2610 = O.S. Khan) contains only logging fields `temporalVal`/`temporalOption` and no temporal scoring code. No 2025–2026 Exquisitor repo was found under Ok2610 or ITU-DASYALab. — [ExquisitorServer](https://github.com/Ok2610/ExquisitorServer); [Ok2610 repos](https://github.com/Ok2610?tab=repositories)
- **vitrivr-engine `TemporalSequenceAggregator.kt`** [source code, HEAD d22860e, 2026-08-17]:
  - *Constants*: `PADDING_TIME = 1_000_000_000` (1 s in ns) and `MAX_TIME_BETWEEN_STAGES = 10_000_000_000` (10 s in ns).
  - *Per stage (one per sub-query)*: for each source video, the retrieved segments are sorted by start time and merged into "ContinuousSequence"s. Each sequence has start, end, score = **max** of its segments' scores, and a stage index.
  - *Chain construction*: for every start stage s and every sequence of that stage, iterate over the next stages s+1…N−1. For each, take `filter { it.start in last.start..(last.end + 10 s) }.maxByOrNull { it.score }`, i.e. the **highest-scoring** candidate starting between the previous element's start and 10 s after its end.
  - If a stage has no candidate it is silently **skipped**. Chains may start at any stage.
  - Chains of length ≥ 2 that are not a subset of an existing chain are kept.
  - *Chain score*: `sequence.maxOf { it.score }`, i.e. the **max, not a sum**. It is emitted as a `temporalSequence` retrievable.
  - *Example config*: 3 CLIP text queries ("Giraffe", "Lion", "Elephant") fed into the aggregator. — [TemporalSequenceAggregator.kt](https://github.com/vitrivr/vitrivr-engine/blob/HEAD/vitrivr-engine-query/src/main/kotlin/org/vitrivr/engine/query/aggregate/TemporalSequenceAggregator.kt); [temporal.json](https://github.com/vitrivr/vitrivr-engine/blob/HEAD/example-configs/advanced/retrieval/temporal.json)
  - *Code observation*: the segment-merging test `if ((lastEndTime + PADDING_TIME) > startTime)` starts a **new** sequence when the gap is *smaller* than 1 s. The comment above it says the opposite ("if gap … is larger than padding, start new sequence"), so the condition appears inverted. This is my reading of the source, not a documented bug. — [same file](https://github.com/vitrivr/vitrivr-engine/blob/HEAD/vitrivr-engine-query/src/main/kotlin/org/vitrivr/engine/query/aggregate/TemporalSequenceAggregator.kt)
- **Related open-access system (UIT, Multimedia Systems 2026)**: "Towards scalable and context-aware multimodal interactive video retrieval" (Tran, Le, Ngo).
  - It describes a "multi-stage sequential retrieval paradigm" with "coarse-to-fine expansion and context-aware alignment". Users define stages that can be "added or removed".
  - Its sequential-query benchmark uses frame pairs ≤ 30 s apart.
  - Only the abstract and supplementary appendix are visible; the main method is paywalled. [abstract + appendix] — [Springer](https://link.springer.com/article/10.1007/s00530-026-02459-8)

### Inferences
- **vitrivr-engine vs C1**:
  - Shared: ordered stages, seconds as the unit, a hard max-gap cap (10 s, fixed, not configurable), implicit skipping of stages, corpus-level output, no user pivot.
  - Missing: an exact DP (it is greedy, taking the best next candidate per stage), any additive gap penalty, a drop *cost* (skips are free, so a 2-of-5 match can outscore a full match), additive path scoring (it uses the max, so the chain score does not reward matching more events), and any caption term.
  - Because the input is the top-k retrieved segments per stage, alignment runs over retrieved candidates, not over all keyframes of a video.
- **vitrivr's 2021 Cineast IDA** (prior notes) already had seconds-based exponential decay and a max-time cap. So "seconds + cap" is established in VBS practice. C1's distinguishing combination remains an **exact DP + additive seconds penalty + cap + priced drop**.
- **Exquisitor 2026**: its "sequence chain + RRF" is at least an ordered multi-stage query. Since RRF fuses ranks, it cannot be an additive score DP with seconds-based penalties unless the paper also has a score stage. **This cannot be verified.**

### Gaps
- The Exquisitor 2026 formula (window size, chain construction, RRF k) is unknown: paywalled, no code.
- It is unknown whether any vitrivr VBS 2026 deployment used a different scorer from the `main` branch.

---

## Q3. arXiv 2304.12570 and arXiv 2303.17703: what they do and their results

### Takeaway
Neither paper applies diffusion or k-reciprocal reranking to **CLIP text→image scores in video**, and neither uses a temporal kernel.
- **2304.12570 (LeaPRR, SIGIR 2023)** is a **trained** GCN reranker for image–text retrieval (Flickr30K, MS-COCO). Its baseline table does show that classic QE, DBA, diffusion (DFS) and FSR often *hurt* cross-modal retrieval, while k-reciprocal KRNN helps.
- **2303.17703 (ICFRR, 2023)** is a **training-free**, iterative, rank-based gallery–gallery reranker for zero-shot sketch→image retrieval.

Both are general neighbour-context rerankers. They support C2's framing ("neighbour-context reranking of cross-modal scores is known") but do not anticipate C2's same-video, time × embedding bilateral weights.

### Cited Findings
- **LeaPRR (Qu, Liu, Wang, Zheng, Nie, Chua; SIGIR 2023; arXiv 2304.12570)** [full text]:
  - *Problem*: rerank the top-K items of the initial image–text similarity φ(q,d).
  - *Pillars*: the top-L intra- and inter-modal neighbours act as "pillars". A sample becomes a 2L-dim vector of similarities to the pillars: v = [⊕φ(I,T_i), ⊕π(I,I_j)] and t_k = [⊕π(T_k,T_i), ⊕φ(I_j,T_k)] (Eq. 1).
  - *Graph reasoning*: a "neighbor-aware graph reasoning" GCN (2 layers) runs over neighbour- and learning-based affinities.
  - *Training*: contrastive + local triplet + structure-alignment losses (Eq. 11), SGD with batch 512.
  - *Settings*: L = 64 (128-dim pillar space), K = 32 for I2T and K = 8 for T2I, sparse factor 0.8. — [arXiv PDF](https://arxiv.org/pdf/2304.12570)
  - *Results (Table 1, VSE∞ base)*:
    - Flickr30K T2I R@1: 61.4 → **66.6** with LeaPRR. KRNN [ACMMM19] gives 64.6. α-QE leaves R@1 unchanged (61.4) and drops R@5 from 85.9 to 83.4. DFS diffusion 61.4/86.1. FSR drops R@10 from 91.5 to 87.6.
    - MS-COCO 5K rSum: 434.2 → **451.5** (LeaPRR) vs 445.9 (KRNN).
    - With DIME∗: DFS and FSR *reduce* rSum (520.0 → 476.4 and 460.5), while LeaPRR raises it to 535.6. — [arXiv PDF](https://arxiv.org/pdf/2304.12570)
  - The authors argue that existing rerankers fail for image–text retrieval because of "generalization, flexibility, sparsity, and asymmetry". — [arXiv PDF](https://arxiv.org/pdf/2304.12570)
- **ICFRR ("If At First You Don't Succeed", Hudson & Smith, arXiv 2303.17703)** [full text]:
  - *Initial score*: s⁽⁰⁾ = −Euclidean distance between query and gallery. Gallery–gallery ranks r_{i,j} are precomputed, with self-matches excluded (r_{i,i} = ∞).
  - *Rank score*: α[r] = 1 − (r−1)/(G−1) if r ≤ K_g, else 0 (Eq. 5).
  - *Update*: Δ_i⁽ᵗ⁾ = (1/K_q)·Σ_{p=1}^{K_q} α[r_{I_p⁽ᵗ⁾, i}] (Eq. 6), the mean rank-score of gallery item i in the neighbour lists of the current top-K_q items. It is added with weight β (Eq. 3) and iterated until ranks stop changing (≈10 iterations are best in the supplement).
  - *Rule of thumb*: K_q ≈ K_g ≈ 0.5 × the expected number of positives, β = 0.5.
  - *Results*: ViT-x base → +ICFRR on TU-Berlin mAP@all 0.695 → 0.754, Sketchy 0.777 → 0.836, QuickDraw 0.204 → 0.270. Office-Home Art→Product mAP 0.741 → 0.855.
  - *Compared rerankers*: it beats SuperGlobal, AQE, DBA, ECN, EDRM (diffusion) and CSA on the same backbone. It is image↔image (sketch/cartoon→photo), not text→image. — [arXiv PDF](https://arxiv.org/pdf/2303.17703)

### Inferences
- **For C2, LeaPRR's Table 1 is useful negative evidence**: naive feature-space QE, DBA and diffusion *degrade* cross-modal text→image retrieval. This supports C2's design choice to propagate **scores**, not to expand features, and to restrict neighbours to the same video with a time kernel. It is also a warning that unrestricted neighbour smoothing can hurt, which argues for including the self term.
- ICFRR is the rank-based, training-free analogue of kNN score propagation. Like k-reciprocal/ECN it uses gallery–gallery neighbourhoods with no notion of time. C2's time kernel and same-video restriction are absent from both papers.
- Neither paper touches C1.

### Gaps
- No paper found (here or in the prior notes) that runs diffusion or k-reciprocal reranking on CLIP/SigLIP **text→keyframe** scores inside a video corpus and reports numbers.

---

## Q4. Prior use of time × feature-similarity (bilateral / edge-aware) weighting on per-frame relevance scores

### Takeaway
No 2024–2026 paper was found that multiplies a temporal-distance kernel by an embedding-similarity kernel to smooth or propagate **text→frame relevance scores**. The closest items each have only one factor:
- **time only**: Gaussian, box or triangular smoothing in TFVTG, TAG, DSE-VTG and P2S; temporal re-scoring in Safadi & Quénot 2011; temporal reranking in Yang & Hauptmann 2006;
- **feature only**: multi-graph propagation (ACM MM 2007); LeaPRR/ICFRR/k-reciprocal;
- **a time window + feature statistics used as a *confidence*, not as propagation weights**: the ABTS stability term (2504.09298);
- **a feature-similarity curve *replacing* the text curve**: the anchor self-similarity in Robust ZS-VTG (CVPRW 2026).

The bilateral combination itself remains unanticipated as far as these searches reach. Its ingredients are classic (bilateral filter, graph propagation), so C2 should be framed as a specific instantiation, not a new principle.

### Cited Findings
- **TAG (BMVC 2025, arXiv 2508.07925)** [full text via arXiv HTML]:
  - *Temporal pooling*: uniform window average of frame features, w = 21.
  - *Clustering*: temporal-coherence k-means (k = 9, window r = 7) penalising different clusters for adjacent frames.
  - *Similarity adjustment*: a scalar Box-Cox transform a_i = ((f_i·q)^λ − 1)/λ.
  - *Proposal score*: mean inside minus mean outside.
  - No step weights scores by both time and feature similarity. Charades mIoU 45.69, ActivityNet 36.55. — [arXiv HTML](https://arxiv.org/html/2508.07925)
- **Robust ZS-VTG (CVPRW 2026)** [full text, prior notes]: replaces the noisy text→frame curve with visual self-similarity to the argmax anchor (s^im_j = v_k·v_j). This is a feature-similarity re-scoring around one anchor, without a time kernel and not applied to all frames. — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/papers/Banditakkarakul_Towards_Robust_Zero-Shot_Video_Temporal_Grounding_CVPRW_2026_paper.pdf)
- **ABTS (2504.09298)** [full text, prior notes]:
  - Per-frame confidence c_i = λ_s·s_i + λ_t·t_i with stability t_i = 1 − min(1, 2·σ({e_j·e_i : j ∈ N_i})) over a temporal neighbourhood N_i.
  - This combines a time window with embedding similarity, but as an additive confidence bonus on frame i. It does **not** propagate neighbours' text scores. — [arXiv 2504.09298](https://arxiv.org/abs/2504.09298)
- **Prompts to Summaries (arXiv 2506.10807, zero-shot video summarization)** [full text via arXiv HTML]:
  - Scene scores are propagated to frames by cosine interpolation between scene midpoints (time only).
  - The result is multiplied by a frame weight = σ·Consistency + (1−σ)·Uniqueness, computed from K-means over CLIP embeddings.
  - Training-free. SumMe 56.73 F1, TVSum 62.21 F1.
  - The time and feature terms are applied as separate factors per frame, not as pairwise bilateral weights. — [arXiv HTML](https://arxiv.org/html/2506.10807v1)
- **Yang & Hauptmann, "Exploring temporal consistency for video analysis and retrieval" (MIR 2006)** [abstract]: "temporally adjacent video shots usually share similar visual and semantic content". It proposes "a temporal reranking method … for improving the efficiency of interactive video search" on TRECVID. The weighting form is unknown. — [OpenAlex record](https://api.openalex.org/works?search=Exploring%20temporal%20consistency%20for%20video%20analysis%20and%20retrieval)
- **Video search re-ranking via multi-graph propagation (ACM MM 2007)** [abstract]: shots are vertices, and "conceptual and visual similarity" are "hyperlinks". A modified topic-sensitive PageRank propagates relevance. There is no time kernel. — [Microsoft Research](https://www.microsoft.com/en-us/research/publication/video-search-re-ranking-via-multi-graph-propagation/)
- **Event-Anchored Frame Selection (arXiv 2603.00983)** [abstract]: DINO-embedding event segmentation, the most query-relevant frame per event as anchor, then adaptive MMR. This is frame selection for MLLMs, not score propagation. — [arXiv](https://arxiv.org/abs/2603.00983)
- **Search outcome**: queries on "bilateral filter similarity curve video grounding", "temporal smoothing frame relevance weighted by visual similarity", "graph propagation relevance scores temporal adjacency visual similarity" and "refine query-frame similarity using frame-frame similarity matrix temporal Gaussian" returned only the items above plus trained models. One search summary mentioned a "hybrid adjacency combining temporal proximity with learned feature similarity" in trained VTG graph models. [snippet; not traced to a specific paper] — search results including [ST-GCVA](https://www.sciencedirect.com/science/article/pii/S2667305326000517) and [Sparse-Dense Side-Tuner](https://arxiv.org/pdf/2507.07744)

### Inferences
- The strongest honest claim for C2 is: "a training-free, one-step, joint-bilateral (time × embedding) propagation of cross-modal relevance scores over irregular shot timestamps within a video, with a self term".
- Trained VTG graph models may already use temporal × learned-feature adjacency *inside the network* (snippet-level). The novelty therefore rests on its **training-free, post-hoc, score-level** nature and on handling irregular timestamps, not on the idea of combining time and feature affinities.
- TAG and Robust ZS-VTG show that the zero-shot community currently fixes noisy text curves by temporal pooling or by switching to visual self-similarity. C2 sits between the two, and both are natural ablation baselines: time-only kernel, feature-only kernel (anchor self-similarity), bilateral.

### Gaps
- The trained VTG models with "temporal + feature" hybrid adjacency were not verified at full text. If they exist, they should be cited as learned analogues.
- The exact weighting in Yang & Hauptmann (2006) was not verified: the full text was blocked (ResearchGate and Academia 403).

---

## Q5. Other training-free VCMR / multi-event retrieval papers 2025–2026, and a consolidated comparison

### Takeaway
Beyond SCAC, searches found **no new training-free VCMR paper** for 2025–2026. New multi-event items are trained:
- **GenSpan** (arXiv 2603.22121, multi-verb VCMR on TVR/ActivityNet, bidirectional SSM);
- **MELON** (arXiv 2609.01654, dataset plus a multi-event-aware training loss).

The listed training-free works (DANTE, MADTempo, 2512.12935, TFVTG, BaGLM, HiERO-StepG, Robust ZS-VTG, Moment-GPT, VTG-GPT) were checked at full text in the prior notes. **DANTE** remains the closest C1 overlap: exact O(NT) ordered DP, sparse scene keyframes, corpus ranking by max_t DP[N,t]. It differs in:
- penalising **keyframe-index** gaps;
- having no max gap, no drop, no caption term and no span.

### Cited Findings
- **GenSpan (Sun et al., arXiv 2603.22121, v2 June 2026)** [abstract]: "constructs short auxiliary videos from LLM-selected subtitle cues and decomposed sub-events, using these as temporal priors". It uses a bidirectional state-space model. Evaluated on TVR and ActivityNet-Captions. Trained. — [arXiv](https://arxiv.org/abs/2603.22121)
- **MELON (Hur et al., arXiv 2609.01654, 31 August 2026)** [abstract]: multi-event text-to-long-video retrieval dataset with "multiple event intervals per video" and "a multi-event aware loss". Trained. — [arXiv](https://arxiv.org/abs/2609.01654)
- **REZE (arXiv 2608.04480)** [snippet]: training-free single-query MR that converts an MLLM clip-confidence curve to spans. It raises the training-free mAP on QVHighlights from 38.23 to 40.32. It is single-video and single-event. — [arXiv](https://arxiv.org/abs/2608.04480)
- **DANTE (arXiv 2512.13169)** [full text, prior notes]:
  - DP[i,t] = S[i,t] + max_{τ<t}(DP[i−1,τ] − λ(t−τ)), with λ acting on keyframe index.
  - Running max gives O(N·T). Video score = max_t DP[N,t]. BEiT-3 unary only.
  - No max gap, drop, caption or span. — [arXiv](https://arxiv.org/abs/2512.13169)
- **MADTempo (arXiv 2512.12929)** [full text, prior notes]: boundary-pair candidates within (n−1)·τ in timestamp units, beam search for intermediate events, and an LLM ContextScore on boundary-keyframe captions plus ASR. — [arXiv](https://arxiv.org/abs/2512.12929)
- **2512.12935** [full text, prior notes]: beam search (B = 8); additive SS = Σ s_i·e^{−α(t_i−t_{i−1})} with α = 0.01 in seconds; BLIP-2 ITM post-hoc gate; BEiT-3 + SigLIP. — [arXiv](https://arxiv.org/abs/2512.12935)
- **HiERO-StepG (arXiv 2605.31227)** [full text, prior notes]:
  - Exact Viterbi over N ordered steps, with a hard monotonicity tolerance τ = 1 s. No gap penalty, no drop.
  - Ordered decoding raises val R1@0.3 from 15.70 to 35.28.
  - Its backbone is trained on EgoClip. — [arXiv](https://arxiv.org/pdf/2605.31227)
- **TFVTG, BaGLM, Robust ZS-VTG, Moment-GPT, VTG-GPT** [full text, prior notes]: all single-video VTG on dense frames.
  - TFVTG combines 2–3 LLM sub-events by proposal intersection or union with order filtering.
  - BaGLM is online Bayesian filtering with an LLM transition prior.
  - The other three are single-query span extractors. — [TFVTG](https://arxiv.org/html/2408.16219); [BaGLM](https://arxiv.org/abs/2510.16989); [Robust ZS-VTG](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/papers/Banditakkarakul_Towards_Robust_Zero-Shot_Video_Temporal_Grounding_CVPRW_2026_paper.pdf); [Moment-GPT](https://arxiv.org/html/2501.07972); [VTG-GPT](https://arxiv.org/abs/2403.02076)
- **Drop-DTW (NeurIPS 2021)** [full text, prior notes]: exact DP with a percentile drop cost, index-based, no time gap and no cap. — [arXiv 2108.11996](https://arxiv.org/abs/2108.11996)

### Inferences
**C1 element-by-element risk:**
- **Already present in prior work**:
  - N ordered sub-queries: SCAC, DANTE, MADTempo, 2512.12935, vitrivr, HiERO.
  - Exact O(NT) DP with running max: DANTE, HiERO Viterbi.
  - Seconds-based penalty: 2512.12935 (beam) and vitrivr IDA (heuristic).
  - Hard max gap in seconds: vitrivr-engine (10 s), MADTempo, VISIONE.
  - Percentile drop cost: Drop-DTW.
  - No pivot: DANTE, MADTempo, 2512.12935, vitrivr-engine, SCAC.
  - Corpus ranking by path score: DANTE.
- **Combination not found in any single work**:
  - exact DP + additive seconds penalty + hard seconds cap enforced via sliding-window max;
  - an event-drop state with percentile cost inside that DP;
  - a fused multi-encoder image unary + β·caption term inside the DP unary;
  - span output from shot boundaries.
- **The open threat is SCAC**. Its scoring is unknown and it already has LLM ordered event chains + training-free + corpus ranking + captions. The C1 claim must stay on the alignment mechanics.

**C2**: no paper found with a time × embedding bilateral kernel on text→keyframe scores, with or without a self term. Single-factor precedents should be cited:
- temporal-only: TFVTG, TAG, Safadi & Quénot, Yang & Hauptmann;
- feature-only graph: ACM MM 2007 multi-graph, LeaPRR, ICFRR, k-reciprocal;
- anchor self-similarity: Robust ZS-VTG;
- time-window embedding-stability confidence: ABTS.

### Summary table

Yes/no refer to what the method does as published. "unk" means unknown (not obtainable). Source level is in brackets.

| Paper [source level] | N ordered events | Exact DP vs beam/greedy | Gap unit | Max gap | Drop/skip | Caption term | Needs pivot | Sparse keyframes | Corpus-level ranking | Bilateral score smoothing |
|---|---|---|---|---|---|---|---|---|---|---|
| SCAC, TIP 2026 [abstract + partial code] | yes (LLM, chronologically sorted) | unk (core code unreleased) | unk | unk | unk | yes, as video-side event chains from BLIP captions; unk as a score term | no | no/unk (3 fps BLIP features, stride windows) | yes (top-200 videos, VCMR R@10/100) | unk (no evidence) |
| Exquisitor VBS 2026 [abstract] | yes ("sequence chain") | unk (RRF rank fusion) | unk | unk | unk | unk | unk | unk | yes (VBS system) | unk |
| vitrivr-engine TemporalSequenceAggregator 2026 [code] | yes (stages) | greedy | seconds (ns) | yes, 10 s fixed | skip yes, free (no cost) | no | no | yes (shot segments, top-k per stage) | yes (score = max over stages) | no |
| vitrivr/Cineast IDA 2021 [full text, prior] | yes | heuristic | seconds | yes (max time) | no (normalise by #containers) | no | no, but needs user-given distances | yes (segments) | yes | no |
| DANTE 2512.13169 [full text, prior] | yes | exact DP O(NT) | keyframe index | no | no | no | no | yes (4 per scene) | yes (max_t DP[N,t]) | no |
| MADTempo 2512.12929 [full text, prior] | yes | beam | timestamp | yes (τ, (n−1)τ) | no | partial (LLM ContextScore on captions, not unary) | no | yes | yes | no |
| 2512.12935 [full text, prior] | yes | beam (B = 8) | seconds (exp decay) | no | no | partial (BLIP-2 ITM post-hoc gate) | no | yes | yes | no |
| 2504.09298 ABTS [full text, prior] | 2 (start/end) | independent argmax | seconds window | yes (W = 10–20 s) | no | no | yes | yes | yes (global rerank) | no (stability confidence, not propagation) |
| HiERO-StepG 2605.31227 [full text, prior] | yes | exact Viterbi | seconds (hard monotone tolerance only) | no | no | no | no | no (dense features) | no (single video) | no |
| Drop-DTW NeurIPS'21 [full text, prior] | yes | exact DP | index | no | yes (percentile cost) | no | no | no | no | no |
| TFVTG 2408.16219 [full text, prior] | 2–3 LLM sub-events | proposal combination + order filter | frames | no | no | no | no | no | no | no (Gaussian, time-only) |
| BaGLM 2510.16989 [full text, prior] | yes (LLM prior) | online Bayesian filter | no | no | no | no | no | no | no | no |
| Robust ZS-VTG CVPRW'26 [full text, prior] | no (single query) | anchor + expand | seconds (merge gap g) | no | no | no | auto anchor (argmax), no user pivot | no | no | no (anchor self-similarity replaces curve) |
| Moment-GPT / VTG-GPT [full text, prior] | no | threshold span generator | frames | no | no | VTG-GPT yes (caption matching); Moment-GPT no | no | no | no | no |
| TAG 2508.07925 [full text] | no | contrast proposals | frames | no | no | no | no | no | no | no (box pooling, time-only) |
| LeaPRR 2304.12570 [full text] | n/a | n/a | n/a | n/a | n/a | n/a | no | n/a (image–text) | yes (top-K rerank) | no (trained GCN neighbour graph, no time) |
| ICFRR 2303.17703 [full text] | n/a | n/a | n/a | n/a | n/a | n/a | no | n/a (image–image) | yes | no (rank-based kNN update, no time) |
| Multi-graph propagation MM'07 [abstract] | n/a | n/a | n/a | n/a | n/a | text relevance as prior | no | shots | yes | no (visual/concept graph, no time) |
| Yang & Hauptmann MIR'06 [abstract] | n/a | n/a | n/a | n/a | n/a | n/a | interactive | shots | yes | unk (temporal reranking) |
| GenSpan 2603.22121 [abstract] | yes (sub-events) | learned SSM | unk | unk | unk | subtitles | no | no | yes (TVR/ANet VCMR) | no; trained |
| MELON 2609.01654 [abstract] | multi-event (unordered intervals) | trained loss | n/a | n/a | n/a | n/a | no | no | yes | no; trained |
| **C1 + C2 (ours)** | yes | exact DP O(N·T) | seconds | yes (sliding-window max) | yes (percentile cost) | yes (β·caption in unary) | no | yes (irregular shots) | yes (path score) | yes (C2) |

### Gaps
- SCAC and Exquisitor 2026 formulas could not be verified. Both are closed access, and SCAC's core code is unreleased. These two rows are the main residual novelty risk.
- There was no systematic search of IEEE TIP, TMM or TCSVT 2026 issues beyond keyword search. Another journal-only training-free VCMR paper could exist behind a paywall.
