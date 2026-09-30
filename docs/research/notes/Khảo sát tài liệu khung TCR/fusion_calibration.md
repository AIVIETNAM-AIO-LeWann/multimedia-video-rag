# Score fusion and calibration for two dual encoders (SigLIP 2 so400m + BEiT-3 large COCO) in text-to-image keyframe retrieval

Scope and provenance. This note builds on `docs/research/notes/Nghiên cứu truy vấn video khoảnh khắc/fusion_and_captions.md` and does not repeat it. That note already covers Bruch et al. TOIS'23 (convex combination vs RRF, NDCG tables, sample efficiency), the NNN headline numbers, SigLIP 2 / BEiT-3 benchmark tables, and GenSearch query paraphrasing.

Every finding below is tagged by where the evidence comes from:
- **[IR]**: text information retrieval.
- **[ITR]**: image-text retrieval.
- **[VTR]**: video-text retrieval.

Claims seen only in a search snippet are marked **(snippet-only)**.

Project observation, restated for the reader: switching between RRF and max-normalized score fusion changes the top-1 video on about 40% of the 79 labeled queries.

---

## 1. Hubness reduction for cross-modal retrieval: formulas, reference sets, gains, applicability

### Takeaway
All the main methods share one idea: subtract or divide out a per-gallery-item "hubness" term that is estimated from a bank of reference queries. The methods are inverted softmax (IS), CSLS, QB-Norm's Dynamic Inverted Softmax (DIS), DBNorm (DualIS/DualDIS) and NNN.

- Typical text→image R@1 gains are +3 points for strong zero-shot models (SigLIP, BEiT-3) and +4 to +7 points for weaker CLIP ViT-B/32 on COCO/Flickr.
- Gains on already-strong finetuned models are about 0.5 to 1.8 points.
- IS and CSLS can **hurt** badly when the query bank does not cover the gallery. DIS and NNN with a small α are the "do no harm" options.
- NNN is the only method with published SigLIP and BEiT-3 numbers. It is the natural first choice here, and its bias term is additive, so it fits a cosine-score fusion pipeline directly.

### Cited Findings

**Formulas**

Notation: s_q(j) is the raw similarity of query q to gallery item j; p_j is the vector of similarities between gallery item j and all querybank queries.

- **Inverted Softmax (IS)** [originally bilingual word translation; applied to VTR/ITR in QB-Norm]. The query bank is built from a random subsample of possible queries. The normalized score is η_q(j) = exp(β·s_q(j)) / Σ_i exp(β·p_j[i]), where β is an inverse temperature. Performance "varies smoothly with inverse temperature, peaking at a value of 20" for TT-CE+ on MSR-VTT — [QB-Norm, arXiv 2112.12777 PDF](https://arxiv.org/pdf/2112.12777)
- **CSLS** (as used in QB-Norm): η_q(j) = 2·s_q(j) − (1/K)·1ᵀŝ_q − (1/K)·1ᵀp̂_j. Here ŝ_q is restricted to the K gallery items most similar to q, and p̂_j to the K querybank queries most similar to j — [arXiv 2112.12777 PDF](https://arxiv.org/pdf/2112.12777)
- **Dynamic Inverted Softmax (DIS, QB-Norm)**, by Bogolin, Croitoru, Jin, Liu and Albanie at CVPR 2022. First precompute an activation set A = the top-k gallery items retrieved by each querybank query, i.e. "potential hubs". Then η_q(j) = IS score if argmax_l s_q(l) ∈ A, and s_q(j) otherwise. In words, normalization is applied only when the query's top-1 hit is a known hub. The authors "observed that choosing k = 1 offers a good trade-off." Motivation: "if the querybank does not effectively cover the space containing the gallery, performance is severely degraded such that it falls below the performance of unnormalised similarities" — [arXiv 2112.12777](https://arxiv.org/abs/2112.12777); [PDF](https://arxiv.org/pdf/2112.12777)
- **DBNorm** (Wang et al., EMNLP 2023) adds a *gallery bank* alongside the query bank.
  - DualIS: ŝ = ŝ^g · ŝ^q, where ŝ^g = exp(β₁ s_{q,g_i}) / Σ_j exp(β₁ s_{g_i,ĝ_j}) over the gallery bank, and ŝ^q = exp(β₂ s_{q,g_i}) / Σ_j exp(β₂ s_{g_i,q̂_j}) over the query bank.
  - DualDIS applies each term only for items in the top-k activation sets.
  - The authors argue that "a hub will exhibit high similarity with any point from 𝒳 or 𝒴", so both banks are needed.
  - Sources: [arXiv 2310.11612](https://arxiv.org/html/2310.11612); [ACL Anthology](https://aclanthology.org/2023.emnlp-main.652/); [code](https://github.com/yimuwangcs/Better_Cross_Modal_Retrieval)
- **Nearest Neighbor Normalization (NNN)** (EMNLP 2024). The bias is b(r) = α·(1/k)·Σ_{j∈top-k reference queries of r} s(q_j, r), and the debiased score is s_D(q, r) = s(q, r) − b(r).
  - Grid: α ∈ {0.25, …, 1.5}, k ∈ {1, 2, 4, …, 512}.
  - "For out-of-distribution reference query databases, smaller α (0.25 to 0.5) and k (8 to 16) are optimal, and for in-distribution reference query sets, larger α (0.75) are optimal."
  - Source: [arXiv 2410.24114](https://arxiv.org/html/2410.24114v1)

**Reported gains**

- **QB-Norm [VTR]**, TT-CE+ on MSR-VTT full split (R@1):

  | Querybank | R@1 |
  |---|---|
  | No querybank | 14.9 |
  | Training-set querybank, 60k | 17.3 |
  | Val set, 10k | 16.6 |
  | Test set, 60k | 17.5 |

  "strong results can be obtained with a querybank of just a few thousand random training samples" — [arXiv 2112.12777 PDF](https://arxiv.org/pdf/2112.12777)
- **QB-Norm robustness to querybank domain [VTR]**, same model (base R@1 14.9):

  | Querybank | GC | CSLS | IS | DIS |
  |---|---|---|---|---|
  | In-domain (MSR-VTT) | 15.8 | 16.8 | 17.1 | 17.0 |
  | Far domain (LSMDC) | 14.8 | 13.4 | 11.6 | 14.9 |
  | "Adversarial" low-coverage MSR-VTT bank | 14.5 | 14.4 | 12.3 | 14.9 |

  With the far-domain bank IS loses 3.3 R@1. The authors traced the failure to LSMDC queries that "retrieve only a small subset of videos" — [arXiv 2112.12777 PDF](https://arxiv.org/pdf/2112.12777)
- **QB-Norm, other results**:
  - [ITR] MSCOCO 5k T→I with CLIP ViT-B/32: 30.3 → 34.8 R@1. MMT-OSCAR: 52.2 → 53.9.
  - [VTR] CLIP2Video on MSR-VTT 1k-A: 45.6 → 47.2. On MSVD: 47.0 → 47.6.
  - Source: [arXiv 2112.12777 PDF](https://arxiv.org/pdf/2112.12777)
- **DBNorm [ITR/VTR]**, R@1:

  | Setting | Baseline | IS | DIS | DualIS | DualDIS |
  |---|---|---|---|---|---|
  | COCO 5k T→I, CLIP | 30.31 | 35.15 | 35.16 | 37.93 | 37.92 |
  | Flickr30k T→I, Oscar | 71.60 | 72.26 | 72.26 | 73.00 | 73.00 |
  | MSR-VTT, CLIP4Clip | 44.10 | 44.20 | 44.20 | 45.00 | 45.00 |

  Source: [arXiv 2310.11612](https://arxiv.org/html/2310.11612)
- **NNN [ITR]**:
  - COCO T→I R@1: CLIP 30.43 → 37.53; BLIP ft-COCO 62.68 → 64.44; SigLIP 47.15 → 50.24; BEiT-3 47.62 → 50.64.
  - Flickr30k T→I R@1 (original / DBNorm / NNN): CLIP 58.82 / 65.26 / 64.60; BLIP ft-Flickr 83.58 / 83.12 / 84.32; SigLIP 74.62 / 76.02 / 76.54. DBNorm slightly *hurts* the finetuned BLIP, and NNN does not.
  - Reference set size: with 20% of the reference data, CLIP COCO still reaches 37.53 vs 37.74 with the full data.
  - OOD reference set: COCO references used on Flickr give 64.42 vs 65.52 with in-distribution references.
  - Stated limitation: shown only for contrastive dual encoders, not cross-attention models.
  - Source: [arXiv 2410.24114](https://arxiv.org/html/2410.24114v1)
- **Newer work (snippet-only)**: NeighborRetr (CVPR 2025) addresses hub centrality during *training*, not as post-hoc normalization — [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_NeighborRetr_Balancing_Hub_Centrality_in_Cross-Modal_Retrieval_CVPR_2025_paper.pdf)

### Inferences
- **Applicability to this project.** NNN's b(r) is one scalar per keyframe per encoder, precomputed once (335k × 2 floats). At query time it is a single subtraction on the full-corpus cosine scores that are already available, so it is practically free.
- **Reference queries.** Two sources are plausible here:
  - (a) Synthetic in-domain queries, e.g. captions or LLM-written queries for a random sample of the project's own keyframes.
  - (b) COCO/Flickr captions, which are OOD for news/cooking footage. Following NNN's guidance, use a small α (0.25–0.5) and k of 8–16 with these.
  - Never use the 79 labeled test queries themselves.
- **Hub frames.** Studio-anchor frames repeated across many videos are exactly the "retrieved by many queries" items these methods demote. However, a query that *wants* an anchor shot, such as "a news anchor in a blue suit", is then penalized. DIS's gating (normalize only when the top-1 is a known hub) and NNN with a small α limit this risk. This effect must be measured on the labeled set.
- **Order of operations.** Calibrate each encoder separately with its own bias, then fuse. NNN's additive form keeps scores on the cosine scale, so theoretical min-max normalization (min = −1) and convex combination still apply afterwards. IS/DBNorm produce exponential-ratio scores on a different scale, which would change the fusion normalization.
- **Expected size of the effect.** For SigLIP and BEiT-3 on COCO, NNN gives about +3 R@1 each. That is smaller than the swing seen between RRF and max-norm fusion, so hubness calibration is likely second-order compared with the choice of fusion rule. It may still matter for the hub-heavy news subset specifically.

### Gaps
- **Dual softmax.** Dual softmax / "DSL" (CAMoE, used in CLIP4Clip-era VTR) was not fetched in this pass. It normalizes over the *test query batch*, which is transductive and not usable for single online queries, but I did not verify its exact formulas or numbers.
- **Checkpoints and versions.** I found no hubness-reduction numbers for SigLIP 2 specifically, or for the BEiT-3 *large COCO-finetuned* checkpoint. NNN's BEiT-3 (47.6 COCO R@1) looks like a non-finetuned or base variant.
- **Video corpora.** I found no study of hubness methods on video-keyframe corpora with heavy near-duplicate frames (as opposed to video-level MSR-VTT retrieval).

---

## 2. Score normalization and fusion of multiple retrievers (incl. CLIP-family ensembles)

### Takeaway
[IR] evidence (Bruch et al. TOIS'23, summarized in the earlier note) favors a tuned convex combination of normalized scores over RRF. The normalization choice matters little *once the weight is tuned*, while RRF is sensitive to k.

The project's observation (about 40% of top-1 videos change between RRF and max-norm) shows that in this setting the fusion rule is **not** a minor detail. The two encoders disagree often enough that rank-only and score-aware fusion diverge.

- I found no peer-reviewed [ITR] study of score-level fusion of two strong CLIP-family encoders with COCO/Flickr numbers. The only one located is a minor-venue paper without extractable numbers.
- The fusion step therefore has to be decided empirically on the 79 labeled queries.
- The candidate set should be small: CC over theoretical-min-max or z-scored cosines with a single α, against RRF(k=60) as the zero-label baseline.

### Cited Findings
- **Standard fixed hybrid rule [IR].** R(q, d) = α·S̃_dense + (1−α)·S̃_BM25, with α "determined via offline tuning on validation data." In DAT's replication, a grid search over α ∈ [0, 1] with step 0.1 picked α* = 0.6 on both SQuAD and DRCD — [DAT, arXiv 2503.23013](https://arxiv.org/pdf/2503.23013)
- **How much fixed fusion helps [IR].** The fixed hybrid beats the best single retriever by large margins:
  - SQuAD P@1: 0.8461 vs 0.7594 (BM25) / 0.7396 (dense).
  - DRCD P@1: 0.8113 vs 0.7630 / 0.5743.
  - Source: [arXiv 2503.23013](https://arxiv.org/pdf/2503.23013)
- **Normalizations and RRF sensitivity [IR].** Bruch et al. compare min-max, theoretical min-max (−1 floor for cosine) and z-score. They find the learned CC "generally agnostic" to the normalization, and RRF "sensitive to its parameters", with in-domain-tuned RRF failing out of domain — [arXiv 2210.11934](https://arxiv.org/abs/2210.11934) (details in the earlier note)
- **CLIP ensembles [ITR], weak evidence.** Combining OpenAI ViT-L-14-336 and Apple ViT-L-14 with per-model confidence normalization "can match or surpass" ViT-H-14. No COCO/Flickr numbers were extractable — [ACM ICIIT 2025](https://dl.acm.org/doi/10.1145/3731763.3731800) (snippet-only)
- **Multi-model fusion in composed image retrieval (snippet-only).** DAFM (2025) reports that adding heterogeneous models gives "monotonically increasing" gains through complementarity — [arXiv 2511.05020](https://arxiv.org/pdf/2511.05020). This is a different task (composed image retrieval, CIR), and I did not verify it.

### Inferences
- **Why RRF and max-norm diverge here.**
  - Max-norm (s/max s per query per encoder) preserves score *gaps* but pins each encoder's top-1 to 1.0. An encoder with a flat top (e.g. SigLIP's sigmoid-trained cosines, which cluster tightly) then gets the same influence as one with a peaked top.
  - RRF ignores gaps entirely.
  - When the encoders' top-1 hits differ, which top-1 wins depends on the tail shape under max-norm and only on rank position under RRF. That is consistent with the 40% flip rate.
  - A per-encoder z-score over the full corpus (μ, σ of all 335k cosines per query) is well defined here because full-corpus scores are available. Unlike top-K min-max, it is not sensitive to the candidate depth.
- **Recommended candidate grid.** Keep it small to protect 79 queries (see §3):
  1. RRF, k = 60, no tuning.
  2. CC over theoretical-min-max cosines, one α.
  3. CC over full-corpus z-scores, one α.
  4. Option 3 after NNN calibration.

  Report video-level R@1/R@5/MRR with bootstrap confidence intervals, not just point estimates.
- **CombSUM and CombMNZ.** CombSUM is the unweighted sum of normalized scores (CC with α = 0.5). CombMNZ multiplies that sum by the number of lists containing the item. With only two encoders and full-corpus scores (every item is "in" both lists unless top-K truncation is used), CombMNZ reduces to CombSUM, or to a hard 2× agreement bonus under top-K truncation. The definitions are standard (Fox & Shaw 1994) but were not re-verified in this pass.

### Gaps
- I found no study reporting COCO/Flickr numbers for score fusion of SigLIP/SigLIP 2 with EVA-CLIP, BLIP or BEiT-3, and nothing comparing RRF vs CC for VLM + VLM fusion.
- I did not retrieve TOIS'23 follow-ups that specifically re-test RRF sensitivity (2024–2026).

---

## 3. Query-adaptive fusion weights (QPP, agreement-based weighting) and overfitting risk

### Takeaway
Classical and neural QPP predictors correlate only weakly with per-query effectiveness of dense retrievers [IR]. The only text-to-image QPP benchmark found (PQPP, CVPR 2025) reports Pearson correlations of about 0.2 for reciprocal rank [ITR]. Weights driven by such weak predictors are unlikely to beat a well-tuned fixed α by much.

The clearest measured gain for query-adaptive weights (DAT) uses an **LLM judge of each retriever's top-1**, not a cheap score statistic. It adds +2 to +3.3 P@1 overall and +4.6 to +7.5 on "hybrid-sensitive" queries over a tuned fixed α [IR].

For contribution ③ (agreement between image-embedding and caption-embedding rankings as a confidence signal), the closest prior art is:
- Reference-list QPP (Shtok/Kurland/Carmel TOIS 2016; Roitman SIGIR 2017).
- Dense-QPP, which measures ranking stability under perturbation.

I found no published image-retrieval study that uses inter-retriever agreement to set fusion weights, so this is a plausible novelty gap. It needs a careful protocol, given only about 40 dev queries.

### Cited Findings
- **QPP for neural IR [IR].** QPP "tends to fall short" for dense IR. On Robust '04, pre-retrieval predictors average 15.9% correlation and post-retrieval predictors 30.2%, "with dense approaches showing the worst QPP performance" — [Faggioli et al., ECIR 2023, arXiv 2302.09947](https://arxiv.org/pdf/2302.09947) (numbers via search snippet; snippet-only)
- **Score-distribution predictors [IR].** NQC is the standard deviation of the top-k retrieval scores, normalized by the corpus score. WIG is the mean divergence of the top-k scores from the corpus score — [search summary](https://arxiv.org/html/2404.01012v1/) (snippet-only)
- **Dense-QPP [IR]** (Arabzadeh et al., 2023) "injects noise [into the] neural representation of the given query, and then measures the similarity between ranked lists for the original query and perturbed query representations" — [Springer MLJ 2024 on robust dense QPP](https://link.springer.com/article/10.1007/s10994-024-06659-z) (snippet-only). This is a rank-agreement signal of the same kind as ③.
- **Reference-list QPP [IR].** Shtok, Kurland & Carmel, "Query performance prediction using reference lists", TOIS 2016, predicts effectiveness from similarity to pseudo-effective and pseudo-ineffective reference rankings. Roitman (SIGIR 2017) found reference-list selection "works better with NQC" — [Semantic Scholar](https://www.semanticscholar.org/paper/Query-Performance-Prediction-Using-Reference-Lists-Shtok-Kurland/4fdcd47fb92bff059d8b5bbcfd5e56a4c9ec22bb) (snippet-only)
- **QPP for text-to-image retrieval [ITR].** PQPP (Poesina et al., CVPR 2025) has 10,200 queries and 1.39M+ human judgments, with retrieval by CLIP ViT-B/32 and BLIP-2. Test-set correlations:

  | Predictor | P@10 Pearson | P@10 Kendall | RR Pearson | RR Kendall |
  |---|---|---|---|---|
  | Fine-tuned BERT | 0.451 | 0.277 | 0.221 | 0.176 |
  | Fine-tuned CLIP | 0.473 | 0.299 | 0.200 | 0.149 |
  | Correlation CNN | 0.270 | 0.186 | 0.189 | 0.162 |

  "the pre-retrieval fine-tuned BERT is a worthy competitor for the post-retrieval predictors." NQC/WIG/clarity were not evaluated on the retrieval side — [arXiv 2406.04746](https://arxiv.org/html/2406.04746v2); [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Poesina_PQPP_A_Joint_Benchmark_for_Text-to-Image_Prompt_and_Query_Performance_CVPR_2025_paper.html). iQPP (2023) is an earlier *image-query* (content-based image retrieval) QPP benchmark — [arXiv 2302.10126](https://arxiv.org/pdf/2302.10126)
- **DAT, query-adaptive α [IR]** (Hsu et al., 2025 preprint).
  - Method: an LLM scores the top-1 of each retriever on a 0–5 scale. Then α(q) = 0.5 if both scores are 0; 1.0 if S_dense = 5 and S_bm25 ≠ 5; 0.0 in the reverse case; otherwise S_v/(S_v + S_b), rounded to 0.1.
  - Full-set P@1:
    - SQuAD: fixed α = 0.6 gives 0.8461. DAT gives 0.8663 with Qwen-14B, 0.8676 with GPT-4o-mini and 0.8740 with GPT-4o.
    - DRCD: 0.8113 → 0.8440 with GPT-4o.
  - Hybrid-sensitive subsets (1111 / 1523 queries), P@1:
    - SQuAD: 0.6229 → 0.6976.
    - DRCD: 0.6507 → 0.7150.
  - Source: [arXiv 2503.23013](https://arxiv.org/pdf/2503.23013)
- **Other signals (snippet-only).** An entropy-based dynamic hybrid weighting method exists on OpenReview, but I did not verify it — [OpenReview](https://openreview.net/forum?id=bwGaZOVo0c)
- **Sample efficiency of fixed CC [IR].** Bruch et al. report that α converges "with less than 5% of the training data". Note that 5% of MS MARCO training queries is still thousands, far more than 40 — [arXiv 2210.11934](https://arxiv.org/pdf/2210.11934)

### Inferences
- **What ③ can realistically gain.** DAT's gain over a tuned fixed α comes from a strong external judge (an LLM reading the top-1 document). A cheap agreement statistic will be a much weaker predictor, likely closer to PQPP-level correlation (about 0.2–0.45).
  - Realistic expectation: a few points of R@1 at best, concentrated on "fusion-sensitive" queries. The project's 40% top-1 flip rate means that subset is large, which is favorable for ③.
  - Evaluate ③ on that subset separately, as DAT does.
- **Agreement features cheap enough to test,** all computable from full-corpus scores of the two lists (image-embedding list vs caption-embedding list, or SigLIP 2 vs BEiT-3):
  - Jaccard or RBO of the top-k.
  - Whether the two top-1 hits are in the same video.
  - Kendall τ on the union of the top-k.
  - Per-list NQC and the top-1/top-2 margin.

  A natural rule: α(q) = σ(a + b·feature), with only 2 parameters, or a gated rule of the form "if the lists agree use CC, else trust the list with the larger top-1 margin".
- **Overfitting with about 40 dev queries.** One fixed α has one degree of freedom and is safe. An adaptive rule with 2–3 parameters is borderline.
  - Use leave-one-out or repeated 2-fold cross-validation over all 79 queries.
  - Report paired bootstrap confidence intervals on the difference from fixed-α CC.
  - Pre-register the feature set; do not search over many agreement features on the same 79 queries.
  - With 79 queries, a 1-query change is about 1.3 R@1 points, so claimed gains under about 5 points are hard to distinguish from noise.
- **Research-contribution framing.** No prior [ITR] work was found that uses inter-encoder or image-vs-caption rank agreement as a fusion weight. The closest precedents are Dense-QPP (perturbation stability) and reference-list QPP, both [IR]. ③ could be framed as "reference-list QPP where the reference is the other modality's ranking."

### Gaps
- I found no study of query-adaptive fusion between two image-text encoders, or between image and caption embeddings, with measured gains over a tuned fixed weight.
- The exact correlation tables of Faggioli et al. and Dense-QPP were not fetched; the numbers above are snippet-only.
- There is no evidence on the minimum labeled-query count at which adaptive weighting reliably beats fixed α.

---

## 4. SigLIP 2 vs BEiT-3 complementarity and known weaknesses

### Takeaway
Published evidence on complementarity is thin. Both are bag-of-words-ish dual encoders with documented compositional-binding failures. SigLIP-family models (so400m and larger) are the best contrastive models on attribute binding, but they still fail "confusion" tests near chance.

BEiT-3's COCO finetuning favors literal COCO-style descriptions. Nothing found measures BEiT-3 on modern compositionality benchmarks alongside SigLIP 2. Fixed weights should therefore come from the project's own labeled data, not from priors.

### Cited Findings
- **SigLIP2-So400M on Auto-Comp [ITR]** (Sbrolli et al., 2026):
  - Color, N = 2: 72.0% Swap / 50.5% Confusion (minimal); 75.5% / 44.0% (contextual).
  - Position, N = 2: 58.1% Swap / 57.7% Confusion.

  So confusion-type accuracy is near or below chance (about 50%). BEiT-3 was not evaluated — [arXiv 2602.02043](https://arxiv.org/html/2602.02043)
- **Snippet-level claims [ITR].** "SigLIP and SigLIP2 models generally outperform CLIP and OpenCLIP models, particularly on attribute-binding tasks"; the larger SO400M/Giant models are best but "not sufficient to close the Confusion gap" — [arXiv 2602.02043](https://arxiv.org/html/2602.02043) and [The Limits of Binding in Dual Encoders, arXiv 2608.15971](https://arxiv.org/html/2608.15971) (snippet-only; attribution between the two uncertain)
- **Benchmark numbers from the earlier note.** SigLIP 2 so400m zero-shot COCO T→I 55.8 R@1; BEiT-3-large COCO-finetuned 63.4 — see the earlier note (SigLIP 2 [arXiv 2502.14786](https://arxiv.org/pdf/2502.14786); [BEiT-3 README](https://github.com/microsoft/unilm/tree/master/beit3)).
- **Hubness bias is similar in size** for SigLIP and BEiT-3 under NNN (+3.1 / +3.0 COCO R@1) — [arXiv 2410.24114](https://arxiv.org/html/2410.24114v1)

### Inferences
- Weaknesses shared by both models (binding, relations, negation, counting) will not be fixed by fusing them. Fusion helps mainly where their *training-data* priors differ:
  - BEiT-3 is COCO-finetuned: literal scene descriptions, common objects.
  - SigLIP 2 is trained on WebLI: long-tail entities, text-in-image, broader domains such as documentary footage and on-screen graphics.
- Since BEiT-3 is 7–8 points better in-domain on COCO, but the corpus is news/cooking/documentary (OOD for COCO), a prior of α ≈ 0.5 is defensible. The tuned α should be inspected per video genre if the labels allow it.

### Gaps
- There is no public comparison of SigLIP 2 and BEiT-3 on SugarCrepe, Winoground or text-in-image (OCR-heavy) retrieval.
- There is no measurement of their error overlap (e.g. oracle-of-two R@1) on any benchmark. The project should compute this oracle on the 79 queries to bound the achievable fusion gain.

---

## 5. Query-side ensembles (paraphrases, prompt templates, multilingual)

### Takeaway
Beyond GenSearch (in the earlier note: +4.9% relative xinfAP on TRECVid AVS, with a noisy branch that had to be down-weighted), this pass found no new controlled [ITR] evidence that LLM paraphrase ensembles help CLIP-style text-to-image retrieval on COCO/Flickr.

Averaging paraphrase embeddings, as GenSearch does per transformation, is the cheap option: one extra LLM call plus N text-encoder passes, with no extra index lookups. Its gain for literal English queries is unproven.

### Cited Findings
- GenSearch "perform[s] an average ensemble of all valid queries with equal weight across each transformation" — [arXiv 2407.12341](https://arxiv.org/html/2407.12341v2)
- **Related claims (snippet-only, unverified).** "Text-to-image retrieval exhibits substantially larger gains from embedding calibration, with improvements reaching more than +19%". The source among the results was not identified, and a candidate is [arXiv 2608.11343](https://arxiv.org/pdf/2608.11343). Treat it as unverified.

### Inferences
- For English queries against SigLIP 2 and BEiT-3, test paraphrase averaging as a *separate* ablation after the fusion rule is fixed. Otherwise the 79-query budget is spread across too many choices.
- Paraphrase averaging may also act as a cheap stabilizer, and the variance of scores across paraphrases could itself serve as a QPP feature for ③.

### Gaps
- I found no [ITR] study measuring LLM-paraphrase ensembles or prompt-template ensembles for CLIP/SigLIP text-to-image retrieval on COCO/Flickr with numbers.
- I found no evidence on multilingual query ensembles for SigLIP 2 beyond what is in the earlier note.
