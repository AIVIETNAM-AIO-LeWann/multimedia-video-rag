# C2 — Context-aware reranking of keyframe scores (graph diffusion, query expansion, temporal smoothing, edge-aware filtering): closest prior art and novelty audit

Scope note: this builds on `docs/research/notes/Nghiên cứu truy vấn video khoảnh khắc/reranking_and_attributes.md` (Q2, Q3) and `temporal_localization.md` (§4). Items already there (SuperGlobal headline numbers, TAG/TFVTG smoothing kernels, the summary of Safadi & Quénot) are only referenced, not repeated. New in this pass: exact formulations from primary PDFs (Iscen et al. CVPR'17, FSR CVPR'18, k-reciprocal CVPR'17, SuperGlobal ICCV'23, 2504.08384 Alg. 2, 2504.09298 ABTS/SuperGlobal use), Quaero TRECVID 2011 confirmation of the Safadi–Quénot gains, CLIP/SigLIP PRF evidence, hubness evidence, and a formal reduction of C2 to one-step diffusion.

Notation for C2 (ours): s̃_t = (1−μ)s_t + μ·Σ_{t'∈N(t)} w_tt' s_t' / Σ_{t'∈N(t)} w_tt', with w_tt' = exp(−Δtime²/2σ_t²)·exp(−(1−cos(e_t,e_t'))/σ_v), N(t) = same-video neighbours within a time window, top ~8 per keyframe, precomputed.

## Q1. Graph-based reranking and diffusion (Iscen diffusion, FSR, k-reciprocal, manifold ranking, score propagation): formulations, gains, cost, cross-modal applicability, noisy top-k

### Takeaway
C2 is, mathematically, one Jacobi step of Zhou/Iscen-style diffusion, f₁ = (1−α)y + αSy with f₀ = y, where S is a row-normalised (random-walk) affinity. The only differences are the affinity (a product of a temporal Gaussian and a visual-similarity kernel, restricted to the same video) and the signal y (cross-modal text→keyframe scores rather than image→image similarities). So C2's novelty is in how the graph is built and where it is applied, not in the propagation operator. No primary source found applies diffusion or k-reciprocal reranking to text→image scores over a temporal+visual keyframe graph.

### Cited Findings
- **Iscen et al., "Efficient Diffusion on Region Manifolds", CVPR 2017** — [arXiv 1611.05113](https://arxiv.org/abs/1611.05113) (formulas from the full PDF):
  - Iteration (their Eq. 3), re-derived as the Jacobi solver: f_t = αS f_{t−1} + (1−α)y. For 0<α<1 it converges to f* = (1−α)(I−αS)⁻¹y (Eq. 4), with L_α = I−αS positive definite.
  - The quadratic cost being minimised is Σ_ij a_ij (f_i − f_j)², i.e. a smoothness prior on scores over the graph.
  - The graph keeps only reciprocal (mutual) k-NN edges, "to handle noise and outliers": s_k(x,z) = min{s_k(x|z), s_k(z|x)} (Eqs. 5–6).
  - Parameters: k = 50 (global) / 200 (regional). Mutual-kNN leaves about 25 edges per image (global) and 75 per region (regional) on INSTRE. The unseen query vector y is built from its k nearest neighbours, with k = 10 for global diffusion.
  - Cost: conjugate gradient converges in about 20 iterations, versus 110 for the plain iteration. Average Oxford5k query time is 0.001 s (global baseline), 0.02 s (global diffusion), 0.321 s (regional baseline) and 0.664 s (regional diffusion). The exhaustive kNN graph for Oxford105k (2.2M regions) takes 96 h; Dong's approximate kNN takes 45 min, with a slight loss.
  - Gains (mAP; columns INSTRE / Oxf5k / Oxf105k / Par6k / Par106k):
    - R-MAC-2048 + AQE: 70.5 / 89.6 / 88.3 / 95.3 / 92.7
    - Global diffusion (2048-D): 80.5 / 87.1 / 87.4 / 96.5 / 95.4. It beats AQE on INSTRE and Paris but is slightly worse on Oxford.
    - Regional diffusion (21×2048): 89.6 / 95.8 / 94.2 / 96.9 / 95.3
  - The paper says the biggest gains are for small objects. Global diffusion "performs well on Paris because query objects almost fully cover the image".
- **Fast Spectral Ranking (Iscen et al., CVPR 2018)** — [arXiv 1703.06935](https://arxiv.org/abs/1703.06935):
  - Treats ranking as graph low-pass filtering, x = h_α(A)y with h_α(A) = β(I−αA)⁻¹ = βΣ(αA)^t.
  - Precomputes a low-rank approximation of A offline, so online search is a sequence of sparse matrix-vector products. The authors say manifold search "runs almost as fast as Euclidean search", so dataset truncation is no longer needed.
  - The query observation y is made sparse by keeping only the k largest entries.
  - Oxford105k: mAP 94.4 uncompressed, falling to 94.2 / 91.1 with 256 / 64-byte PQ codes.
  - The authors "avoid the term diffusion"; they frame it as a random walk / graph filter.
- **k-reciprocal re-ranking (Zhong et al., CVPR 2017, person re-ID)** — [arXiv 1701.08398](https://arxiv.org/abs/1701.08398):
  - Encodes each image's k-reciprocal neighbours as a vector and computes a Jaccard distance d_J. Final distance (Eq. 12): d*(p,g) = (1−λ)d_J(p,g) + λ d(p,g).
  - The paper sweeps k1 and λ with k2 fixed at 6 and λ fixed at 0.3 (Fig. 4).
  - Cost is O(N²) distances plus O(N² log N) ranking over the gallery.
  - Structurally close to C2 in that it convex-mixes the original score with a neighbourhood-derived score. It is purely visual and image→image.
- **SuperGlobal (Shao et al., ICCV 2023)** — [arXiv 2308.06954](https://arxiv.org/abs/2308.06954):
  - Refinement uses GeM pooling over neighbours: f_k = ((1/|X_k|) Σ_{x∈X_k} x^{p_k})^{1/p_k}. The p=1 case (average) is used for database-side refinement; p→∞ (max) is used for query expansion.
  - Final score: S = (S1 + S2)/2, where S1 is the original query vs refined database descriptors, and S2 is the expanded query vs original descriptors.
  - Complexity is O(M²) over the top M. Headline numbers are in the earlier notes.
- **GRAB (arXiv 2504.09298, AIC-style system, local PDF)** puts SuperGlobal (Eqs. 4–5) into a moment-retrieval system, but describes it for "a given query image". It reports no ablation of SuperGlobal for text queries — `D:\Projects\Ref\Paper\2504.09298v1.pdf`; [arXiv 2504.09298](https://arxiv.org/abs/2504.09298).
- **Learnable Pillar-based Re-ranking (arXiv 2304.12570)** and **test-time re-ranking for zero-shot cross-domain retrieval (arXiv 2303.17703)** exist for image-text / cross-domain retrieval. **Snippet-only**: I did not read either paper, so I cannot confirm whether they use neighbourhood graphs or report gains — [2304.12570](https://arxiv.org/pdf/2304.12570), [2303.17703](https://arxiv.org/pdf/2303.17703).

### Inferences
- **Formal reduction of C2 (my derivation from Iscen's Eq. 3/16).**
  - Let W be C2's sparse bilateral affinity with zero diagonal, D = diag(ΣW), and S_rw = D⁻¹W. Then C2 is s̃ = (1−μ)s + μ S_rw s, which is exactly f₁ of the diffusion iteration with α = μ and f₀ = y = s.
  - So in the paper, C2 should be positioned as a "one-step, same-video, bilateral-affinity diffusion of cross-modal scores". It should not be claimed as a new propagation rule.
  - Running more steps (or the closed form (I−μS)⁻¹) turns C2 into full diffusion. That is a natural ablation: does 1 step beat k steps on short scenes?
- Diffusion is usually applied to image→image similarities. Nothing in the maths requires y to be image→image, so any per-keyframe score vector can be diffused, including text→keyframe scores. The prerequisite is that the graph encodes "same-relevance" structure. For a text query, visual similarity is only a proxy for "would also match this text", and it may be a poor proxy for attribute or OCR queries.
- **Behaviour with noisy top-k.**
  - Diffusion/QE amplify whatever the initial y ranks highly, so a wrong top-1 spreads its score to its neighbours. Iscen's mitigations are mutual-kNN edges and a sparse y (top-k only).
  - C2 restricts propagation to within-video time windows. This limits drift to nearby keyframes of the same video, which is much more conservative than global diffusion. A wrong peak can raise only its own neighbourhood.
- **Cost.** At ~384 keyframes/video and ~8 neighbours, C2 is ~3k multiply-adds per video per query, which is negligible. The global-diffusion cost results (CG, FSR) are irrelevant at this scale.

### Gaps
- I found no primary paper that runs Iscen diffusion, FSR or k-reciprocal reranking on CLIP/SigLIP text→image scores and reports numbers. The unverified candidates above are 2304.12570 and 2303.17703.
- The k-reciprocal Market-1501 gain numbers were not extracted (only the formula and the parameter sweep were read).
- FSR's full comparison table was not extracted.

## Q2. Query expansion and database-side augmentation (AQE, αQE, DBA, SuperGlobal) and evidence for text queries (PRF in embedding space)

### Takeaway
For CLIP/SigLIP text→image retrieval, the strongest recent evidence is that classic embedding-space PRF (Rocchio over the top-K image embeddings) gives essentially no gain. This supports C2's choice to smooth scores locally within a video rather than rewrite the query globally.

### Cited Findings
- **"A Little More Like This: Text-to-Image Retrieval with VLMs" (arXiv 2511.17255)**:
  - PRF is written as a softmax-weighted Rocchio update: z'_q = α z_q + β Σ w_i^p z_i − γ Σ w_i^n z_i, with w_{q,i} = exp(s_{q,i}/τ) / Σ_c exp(s_{q,c}/τ).
  - Settings: α=0.8, β=0.1, γ=0.1, τ=0.05, K=5. Tested on Flickr30k and COCO with CLIP-B/32, CLIP-L/14, SigLIP and BLIP-2.
  - Result: PRF "does not show noteworthy improvements". CLIP-B on Flickr30k goes from 0.671 to 0.669 Hits@1; COCO stays at 0.295 → 0.295.
  - Their generative feedback (GRF) and attentive feedback summariser (AFS) give 1–5% MRR@5. GRF drifts from the third feedback round onwards.
  - (Read through a fetch summary of the HTML, not the PDF; the numbers are as reported by that tool.) — [arXiv 2511.17255](https://arxiv.org/html/2511.17255)
- **SuperGlobal's own framing.** Earlier QE/DBA methods are "much more costly" at billion scale; SuperGlobal's reranking is O(M²) — [arXiv 2308.06954](https://arxiv.org/abs/2308.06954).
- **Iscen et al. contrast QE with diffusion.** QE "only explores the neighborhood of very similar images". Recursive QE crawls the manifold at higher query cost, whereas diffusion uses an offline graph — [arXiv 1611.05113](https://arxiv.org/abs/1611.05113).
- AQE/αQE/DBA definitions (standard: AQE is the mean of the query and its top-k; αQE weights each neighbour by sim^α; DBA applies the same operation to database vectors offline) are given in Radenović et al., "Fine-tuning CNN Image Retrieval with No Human Annotation" (TPAMI 2019). **Not re-fetched this pass** — [arXiv 1711.02512](https://arxiv.org/abs/1711.02512).

### Inferences
- **DBA relative to C2.**
  - DBA replaces features with neighbour means, independent of the query. C2 smooths query-dependent scores.
  - With linear scores, "time-local DBA then score" would equal "score then smooth with the same weights": s(q, Σw e′/Σw) = Σw s′/Σw for a dot product.
  - So C2 with fixed μ is equivalent to a same-video, bilateral-weighted DBA of image embeddings, provided the fused SigLIP 2 + BEiT-3 score is linear in each embedding (cosine on normalised vectors, fused by a weighted sum).
  - This equivalence breaks if the fusion is nonlinear: RRF, max-normalisation per query, or softmax calibration.
  - The equivalence matters for novelty claims, and also means C2 could be precomputed as a feature-side augmentation. **This is my derivation, not from a source.**
- The null PRF result for CLIP/SigLIP suggests that global query rewriting from a noisy top-5 does not help text→image retrieval. That is indirect support for local, structure-restricted aggregation, but not evidence that C2 helps.

### Gaps
- No published αQE/DBA/SuperGlobal ablation for text queries in a video-keyframe KIS setting was found. The earlier notes had the same gap.

## Q3. Temporal propagation / smoothing in video retrieval and concept detection (Safadi & Quénot, TRECVID re-scoring, multi-graph propagation, moment-retrieval rerankers)

### Takeaway
Temporal neighbour re-scoring of per-shot scores within the same video is well-established prior art (2007–2011 TRECVID era). It includes Gaussian temporal windows and gains of about 11–18%, so C2's temporal Gaussian term on its own is not new. Graph frameworks that combine a temporal-consistency graph with visual-similarity graphs for shot annotation also exist (OMG-SSL, ACM MM 2007 / TCSVT 2009). They combine the graphs additively in a learned regulariser, rather than as a per-edge product applied to query scores.

### Cited Findings
- **Safadi & Quénot, CIKM 2011, "Re-ranking by local re-scoring for video indexing and retrieval"**:
  - Confirmed in UJF-LIG/Quaero's own TRECVID 2011 notebook: re-ranking "re-evaluat[es] the scores of the shots by the homogeneity and the nature of the video they belong to" and "improve[s] the system performance by about 18% in average on the TRECVID 2010 semantic indexing task… For TRECVID 2008… non-homogeneous contents… about 11-13%" — [Quaero TRECVID 2011 notebook](https://www-nlpir.nist.gov/projects/tvpubs/tv11.papers/quaero.pdf).
  - The pipeline places re-ranking as the last stage, after higher-level fusion, "using the fact that videos statistically have an homogeneous content, at least locally" — [same](https://www-nlpir.nist.gov/projects/tvpubs/tv11.papers/quaero.pdf).
  - **Snippet-only (search summary / Liner review, not the paper)**: shot scores are combined with neighbours by a generalized-mean rule, using Rectangular or Gaussian windowing, and α=2 (RMS) works best — [ResearchGate](https://www.researchgate.net/publication/221614860_Re-ranking_by_Local_Re-scoring_for_Video_Indexing_and_Retrieval); [Liner summary](https://liner.com/review/reranking-by-local-rescoring-for-video-indexing-and-retrieval).
- **OMG-SSL (Wang, Hua et al., ACM MM 2007) / "Unified Video Annotation via Multigraph Learning" (IEEE TCSVT 2009)**. **Snippet-only (abstracts)**:
  - Multiple modalities, multiple distance metrics and *temporal consistency* are each represented as graphs, and the graphs are integrated into one regularisation/optimisation framework for semi-supervised annotation on TRECVID.
  - Exact temporal-graph weights were not retrieved (IEEE is closed access) — [ResearchGate](https://www.researchgate.net/publication/221571327_Optimizing_multi-graph_learning_Towards_a_unified_video_annotation_scheme); [IEEE](https://ieeexplore.ieee.org/document/4801611/).
- **Yang & Hauptmann, "Exploring temporal consistency for video analysis and retrieval" (MIR 2006)**. **Snippet-only**: temporally adjacent shots share visual/semantic content, and exploiting temporal consistency w.r.t. concepts and query topics gives "considerable improvement" in concept detection and retrieval — [ResearchGate](https://www.researchgate.net/publication/221318620_Exploring_temporal_consistency_for_video_analysis_and_retrieval).
- **Video search re-ranking via multi-graph propagation (ACM MM 2007)**. Shots are vertices and visual/conceptual similarities are edges; text-search relevance is propagated with a topic-sensitive PageRank. This entry is from the earlier notes — [Microsoft Research](https://www.microsoft.com/en-us/research/publication/video-search-re-ranking-via-multi-graph-propagation/).
- **arXiv 2504.08384 (AIC system; local PDF, Alg. 2 "Neighbor Score Aggregation")**:
  - For each candidate index, GETNEIGHBORS(key) is called; each neighbour's score is computed with COMPUTE SCORE(N, Q), and non-None scores are summed: `total_score ← total_score + score`. Candidates are then sorted by the aggregated score.
  - There is no weighting, no normalisation by neighbour count, and no mixing with the candidate's own score.
  - The text claims a "conditional score check" handles "sudden changes in the scene". In the pseudocode this appears only as a `score ≠ None` test.
  - The motivation given: "areas surrounding a keyframe are most likely to possess visual and semantic features in common".
  - Sources: `D:\Projects\Ref\Paper\2504.08384v1.pdf`, [arXiv 2504.08384](https://arxiv.org/abs/2504.08384).
- **arXiv 2504.09298 (GRAB, ABTS)**:
  - Per-frame confidence is c_i = λ_s·sim(e_q, e_i) + λ_t·Stability(N_i, e_i).
  - Stability is defined from the standard deviation of similarities between the frame and its neighbours. Low variance means stable; the paper says unstable frames occur "at abrupt cuts".
  - Windows are 10/15/20 s around a user-chosen pivot. The method is used for boundary selection, not first-stage ranking.
  - Sources: `D:\Projects\Ref\Paper\2504.09298v1.pdf`, [arXiv 2504.09298](https://arxiv.org/abs/2504.09298).

### Inferences
- **Sum aggregation in 2504.08384 favours long shots.**
  - A keyframe with more high-scoring neighbours accumulates more, and its own score is not in the sum.
  - This mechanically explains the observed failure: a 1-shot close-up dropped from rank 1 to 3.
  - C2 fixes two of the three causes: normalising by Σw removes the count bias, and the (1−μ)s_t term keeps the own score. The third cause, similarity-blind weights, is addressed by the visual kernel.
- GRAB's stability term uses neighbour-embedding similarity as a feature added to the score, which is a different use of the same signal. C2 uses it as an edge weight. Both treat low neighbour similarity as a sign of a cut.

### Gaps
- The exact Safadi–Quénot equation (whether it includes the shot's own score, and how σ and window size are set) was not verified from the paper PDF.
- OMG-SSL's exact temporal-graph weight function was not retrieved.
- No quantitative ablation of 2504.08384's neighbour aggregation exists in the paper; its evidence is a qualitative case study.

## Q4. Edge-aware / bilateral / guided filtering of 1-D score sequences

### Takeaway
I found no prior work that applies a bilateral/joint-bilateral kernel (time distance × embedding similarity) to query-relevance scores of video keyframes. Its ingredients are classic:
- the bilateral filter (Tomasi & Manduchi);
- joint/cross bilateral filtering, where a guide signal (here, embeddings) sets the range weights for another signal (here, scores);
- Gaussian temporal re-scoring (Safadi & Quénot).

The honest framing is "a joint-bilateral filter over the time axis, guided by embeddings, applied to cross-modal relevance scores".

### Cited Findings
- Searches for bilateral filtering of detection or retrieval score sequences returned only pixel-domain uses: video coding pre-filters, demoiréing and relighting — [Temporal pre-filter for video coding based on bilateral filtering](https://www.researchgate.net/publication/347627977_A_Temporal_Pre-Filter_For_Video_Coding_Based_On_Bilateral_Filtering); [Direction-aware video demoiréing with temporal-guided bilateral learning, arXiv 2308.13388](https://arxiv.org/pdf/2308.13388).
- The survey of edge-preserving smoothing (arXiv 1503.07297) is image-domain — [arXiv 1503.07297](https://arxiv.org/pdf/1503.07297).
- The earlier pass found that training-free VTG smoothing is always linear and temporal-only (Gaussian in TFVTG, box in TAG, triangular in DSE-VTG) — see `temporal_localization.md` §4.
- The closest learned analogue is CVPR 2025 "High Temporal Consistency through Semantic Similarity Propagation". **Snippet-only**: it linearly combines model outputs on frames with interpolation weights based on semantic similarity between frames, for video *segmentation* (pixel-level, not retrieval) — [arXiv 2503.15676](https://arxiv.org/abs/2503.15676).

### Inferences
- **Important design flaw to check in C2 (my analysis).**
  - The classic bilateral filter includes the centre sample (weight 1) in its normaliser. That is what makes it edge-preserving: an isolated sample whose neighbours are all dissimilar stays close to its own value, because Σw ≈ 1.
  - C2 as specified normalises over neighbours only (t' ≠ t) and then mixes with a *fixed* μ. If all neighbours of a 1-shot close-up are visually dissimilar, the w_tt' are tiny, but dividing by their sum re-inflates them. The keyframe is still pulled by the full μ toward the mean of its dissimilar neighbours.
  - That is exactly the short-scene failure C2 is meant to fix. On a true cut, C2 reduces to a temporal-only weighting among neighbours, still with weight μ.
  - Fixes:
    - (a) include the self term: s̃_t = (s_t + Σ w s') / (1 + Σ w), the true bilateral form;
    - (b) use an effective μ_t = μ·Σw / (Σw + κ);
    - (c) do not renormalise, i.e. use the random-walk matrix with a self-loop.
  - Any of these makes the visual kernel act as an edge stop. This is worth an explicit ablation, and it also distinguishes C2 more clearly from 2504.08384.
- The guided-filter family (He et al.) would be an alternative edge-aware 1-D smoother with an embedding guide. I found no use of it on retrieval scores. Not researched further.

### Gaps
- Dense-CRF or temporal-CRF smoothing of action-localisation scores was not surveyed in this pass (tool budget).
- He et al.'s guided filter was not fetched.

## Q5. Joint temporal–visual graphs over video frames/shots; failure modes (over-smoothing, hub frames, repeated shots) and mitigations

### Takeaway
Joint temporal+visual graphs over shots exist for annotation and scene segmentation (OMG-SSL; TAG's temporal coherence clustering), but not as a query-time score smoother for text KIS. Known mitigations for failure modes come from:
- mutual-kNN graphs (noise and outliers);
- hubness normalisation such as QB-Norm (hub items in cross-modal retrieval);
- keeping the original score in the mix (k-reciprocal's λ, diffusion's (1−α)y term, C2's (1−μ)).

### Cited Findings
- **TAG (arXiv 2508.07925)** clusters temporally adjacent frames by feature similarity (window r = 7) to get change points for segmentation, not score weighting. From the earlier notes — [arXiv 2508.07925](https://arxiv.org/html/2508.07925).
- **Mutual-kNN edges.** Iscen et al. keep only reciprocal kNN edges "to handle noise and outliers" — [arXiv 1611.05113](https://arxiv.org/abs/1611.05113).
- **Hubness in cross-modal retrieval (QB-Norm, Bogolin et al., CVPR 2022)**:
  - Joint embeddings suffer from hubness: a few gallery items are nearest neighbours of many queries.
  - QB-Norm renormalises similarities using a querybank and reduces hubness (skewness of the k-occurrence distribution).
  - **Snippet-level numbers**: MSR-VTT R@1 rises from 29.6 to 33.3 for TT-CE+ and from 45.6 to 47.2 for CLIP2Video.
  - Sources: [arXiv 2112.12777](https://arxiv.org/abs/2112.12777); [CVF](https://openaccess.thecvf.com/content/CVPR2022/papers/Bogolin_Cross_Modal_Retrieval_With_Querybank_Normalisation_CVPR_2022_paper.pdf); [code](https://github.com/ioanacroi/qb-norm).
- **Over-smoothing / drift.** GRF query drift appears from round 3 in text→image feedback — [arXiv 2511.17255](https://arxiv.org/html/2511.17255). Diffusion needs α<1 and uses the (1−α)y anchor — [arXiv 1611.05113](https://arxiv.org/abs/1611.05113).
- **Homogeneity assumption.** Safadi–Quénot gains are larger on homogeneous collections (≈18%, TRECVID 2010) than on non-homogeneous ones (11–13%, TRECVID 2008) — [Quaero TRECVID 2011](https://www-nlpir.nist.gov/projects/tvpubs/tv11.papers/quaero.pdf).

### Inferences
- **Studio-anchor hubs.**
  - Within one news video, anchor shots recur many times and are visually near-identical. If C2's window spans two anchor shots separated by a short story shot, the visual kernel links the anchor shots to each other and down-weights the story shot. That behaviour is correct.
  - The risk is a query about an anchor: many anchor keyframes all rise together, which is harmless for KIS.
  - Hubness across videos (QB-Norm's concern) does not affect C2, because the graph is same-video only.
  - Hubness should be handled before C2, in the first-stage fusion.
- **Repeated shots across stories** (the same footage re-aired later in a bulletin) are outside a short time window and are not linked. That is a property of C2 worth stating, versus global DBA or diffusion, which would link them.
- **Over-smoothing** is bounded because C2 is one step and window-limited. Multi-step diffusion would spread relevance across a whole story segment.

### Gaps
- Temporal CRF / Dense-CRF score smoothing in temporal action localisation was not covered.
- Shot-graph scene segmentation (e.g., Rasheed & Shah scene graphs, shot clustering) was not revisited this pass.

---

### Closest prior works vs C2 elements (summary matrix)

| Work | Temporal weight | Visual-sim weight | Cross-modal text query | Sparse shot keyframes | Precomputed sparse graph | Applied to query scores (vs features) |
|---|---|---|---|---|---|---|
| 2504.08384 Alg. 2 (neighbour sum) | fixed window, uniform | no | yes (CLIP+BEiT-3) | yes (TransNetV2 + dedup) | implicit (index neighbours) | scores (sum, no self term) |
| Safadi & Quénot CIKM'11 | yes (rect./Gaussian window; snippet) | no | no (concept classifier scores) | yes (TRECVID shots) | no (temporal adjacency) | scores |
| Multi-graph propagation MM'07 | no | yes | text-search scores (ASR/text) | shots | yes | scores (PageRank) |
| OMG-SSL MM'07 / TCSVT'09 | yes (temporal-consistency graph; snippet) | yes (separate graphs, summed) | no (concept labels) | shots | yes | labels (learned SSL) |
| Iscen diffusion CVPR'17 / FSR CVPR'18 | no | yes (mutual kNN, s^γ) | no (image→image) | no | yes | scores (y → f) |
| k-reciprocal CVPR'17 | no | yes (Jaccard of neighbour sets) | no | no | per query | distances |
| SuperGlobal ICCV'23 / GRAB 2504.09298 | no | yes (top-M) | GRAB uses it in a text system, but described for image queries | yes (GRAB) | no | features (GeM) |
| GRAB ABTS | window around pivot | yes (as a std "stability" feature) | yes | yes | no | additive feature, boundary selection |
| TAG 2508.07925 | yes (window) | yes (clustering) | yes | no (dense 1–3 fps) | no | segmentation, not scores |
| Bilateral / joint bilateral filter | spatial/temporal Gaussian | range kernel on guide | n/a | n/a | n/a | signal filtering (pixels) |

**What is new vs already published (precise statement):**
- **Already published:**
  - the propagation operator (one-step diffusion / random-walk smoothing, Zhou & Iscen);
  - convex mixing of the original and neighbourhood scores (k-reciprocal λ, diffusion (1−α));
  - same-video temporal Gaussian re-scoring of shot scores (Safadi & Quénot);
  - combining temporal and visual graphs over shots (OMG-SSL, for annotation);
  - fixed-window neighbour score aggregation for text KIS on shot keyframes (2504.08384).
- **Not found in prior work:**
  - a *per-edge product* kernel of time distance and embedding similarity (bilateral / joint-bilateral), applied to *cross-modal text→keyframe relevance scores*;
  - with the graph restricted to the same video and built on irregular shot-keyframe timestamps in seconds;
  - precomputed as a sparse (~8-NN) graph and applied as a query-time score filter in a KIS pipeline.
- The contribution is best described as an **edge-aware adaptation of temporal score re-scoring for sparse shot keyframes**, with an explicit failure analysis against 2504.08384. It should not be described as a new reranking algorithm.
- If fusion is linear, C2 is equivalent to a time-local, bilateral-weighted DBA, which should be acknowledged.
- The self-weight issue (Q4 Inferences) should be fixed before claiming edge-preservation.
