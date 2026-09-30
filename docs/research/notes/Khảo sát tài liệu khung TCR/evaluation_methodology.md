# Evaluation methodology for text-to-video moment retrieval / VCMR with a small curated query set

Scope: English queries; 873-video corpus, 335k sparse shot-based keyframes (irregular timestamps); SigLIP 2 so400m + BEiT-3 large + BLIP-2 captions per keyframe; training-free fusion, temporal reranking, ordered multi-event DP alignment. Query set: 79 queries (31 single-scene, 43 ordered sequences, 5 multi-event with 3–4 events).

Verification legend: **[fetched]** = read on a primary page in this session; **[snippet]** = seen only in a search-result snippet; **[bg]** = from background knowledge, URL given but not re-opened in this session — verify before quoting in a paper.

## 1. Standard benchmarks: size, video length, query style, splits, metrics, typical zero-shot/training-free scores

### Takeaway
The single-video grounding benchmarks (Charades-STA, ActivityNet Captions, DiDeMo, QVHighlights) are small enough to evaluate on cheaply and all report R@1 at IoU thresholds plus mIoU (QVHighlights also reports mAP). The best training-free reference is TFVTG (ECCV 2024): 49.97 R@0.5 on Charades-STA and 27.02 on ActivityNet Captions. Corpus-level VCMR benchmarks (TVR, and in practice MAD and Ego4D-NLQ for long video) report R@K at IoU 0.5/0.7 (or 0.1/0.3/0.5 for long video), and absolute scores there are low.

### Cited Findings
**TVR (ECCV 2020): the canonical VCMR benchmark**
- 109K queries over 21.8K videos from 6 TV shows. Videos average 76.2 s, queries 13.4 words, and moments 9.1 s, with about 5 annotations per video. Each query is labeled video-, subtitle- or both-related **[snippet]**. — [arXiv 2001.09099](https://arxiv.org/abs/2001.09099); [ECCV PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660443.pdf)
- Metrics: VR uses R@k with k ∈ {1,5,10,100}. SVMR and VCMR use "R@k, IoU=μ", k ∈ {1,10,100}, μ ∈ {0.5,0.7}: the share of queries with at least one top-k predicted moment (video + span) whose IoU with the GT is above μ **[snippet]**. — [ECCV PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660443.pdf)
- The data are needed together with subtitles, and the TVQA video frames have to be requested from the TVQA authors (access-gated) **[bg]**. — [TVRetrieval GitHub](https://github.com/jayleicn/TVRetrieval)
- Corpus-level extensions of single-video datasets for VCMR exist, e.g. "Text-based Localization of Moments in a Video Corpus", which evaluates DiDeMo/Charades/ActivityNet as corpora **[snippet]**. — [arXiv 2008.08716](https://arxiv.org/pdf/2008.08716); [Finding Moments in Video Collections, arXiv 1907.12763](https://arxiv.org/pdf/1907.12763)

**DiDeMo (ICCV 2017)**
- 10,464 unedited personal Flickr videos of 25–30 s, with 40,543 moment–description pairs. Moments are annotated on a grid of 5-second segments (the snippet says "3-second"; the original paper uses 5-s segments, 6 per video, so treat the snippet figure as suspect) **[snippet]**. — [arXiv 1708.01641](https://arxiv.org/pdf/1708.01641)
- There are multiple human annotations per sentence (4). Scoring requires a prediction to agree with at least 2 of the 4 human judgements, which handles annotator disagreement explicitly **[snippet]**. — [arXiv 1708.01641](https://arxiv.org/pdf/1708.01641); [ResearchGate](https://www.researchgate.net/publication/318982035_Localizing_Moments_in_Video_with_Natural_Language)
- Metrics: R@1, R@5 and mIoU over segment predictions **[bg]**.

**Charades-STA (ICCV 2017, TALL)**
- 6,672 indoor-activity videos (about 30 s each **[bg]**) and 16,128 query–moment pairs, split 12,408 train / 3,720 test **[snippet]**. — [Shuffling/temporal bias, arXiv 2207.14698](https://arxiv.org/abs/2207.14698)
- Known temporal-location bias: train and test share similar moment-position distributions, so methods can exploit priors. Re-split variants (train/val/test-iid/test-ood; "Charades-CD", "ActivityNet-CD") were proposed to counter this **[snippet]**. — [arXiv 2207.14698](https://arxiv.org/html/2207.14698); [Closer look at debiased TSGV, ACM TOMM](https://dl.acm.org/doi/full/10.1145/3565573)
- Metrics: R@1 at IoU {0.3, 0.5, 0.7} and mIoU (standard in TFVTG and similar work).

**ActivityNet Captions (ICCV 2017)**
- About 20K YouTube videos (about 2 min on average, about 849 h in total) with about 100K temporally localized sentences. Each video has a *paragraph of ordered sentences*, which makes this the most-used dataset whose annotations naturally contain ordered multi-event descriptions. Grounding work usually tests on val_2 (or val_1) **[bg]**. — [arXiv 1705.00754](https://arxiv.org/abs/1705.00754)
- Metrics: R@1 at IoU {0.3, 0.5, 0.7} and mIoU.

**QVHighlights (NeurIPS 2021)**
- More than 10,000 YouTube lifestyle-vlog and news videos. Each video has a free-form query, relevant moments (possibly several disjoint ones per query) and five-point saliency scores per clip **[fetched]**. — [arXiv 2107.09609](https://arxiv.org/abs/2107.09609)
- Commonly cited figures are 10,148 videos, 150-s clips, 10,310 queries and 18,367 moments. Metrics are R1@0.5 and R1@0.7, mAP@0.5, mAP@0.75 and average mAP (IoU 0.5:0.05:0.95) for moment retrieval, plus mAP and HIT@1 for highlights. Test labels are hidden behind a CodaLab server **[bg]**. — [Moment-DETR GitHub](https://github.com/jayleicn/moment_detr)
- License: code MIT, annotations **CC BY-NC-SA 4.0**. Moment-DETR trains in under 4 h on one RTX 2080Ti with SlowFast + CLIP features **[fetched]**. — [Moment-DETR GitHub](https://github.com/jayleicn/moment_detr)

**MAD (CVPR 2022): long-form, needle-in-a-haystack**
- More than 384K sentences grounded in more than 1,200 h of full-length movies, about 110.8 min each. Average target moments are only 4.1 s **[snippet]**. — [arXiv 2112.00431](https://arxiv.org/pdf/2112.00431); [GitHub Soldelli/MAD](https://github.com/Soldelli/MAD)
- Metrics: R@K (K = 1, 5, 10, 50, 100) at IoU {0.1, 0.3, 0.5}. The zero-shot CLIP baseline reaches about 32–33 R@50 at IoU 0.1 (MAD-v1/v2). Reported R@1 at IoU 0.1 is only about 3.6–9.3 across methods **[snippet]**. — [RGNet arXiv 2312.06729](https://arxiv.org/pdf/2312.06729); [ReVisionLLM arXiv 2411.14901](https://arxiv.org/pdf/2411.14901)
- Access: copyrighted movies. The raw videos are not distributed; only precomputed CLIP features (about 5 fps) are provided, under an access agreement **[bg]**. As a result, **SigLIP 2 / BEiT-3 embeddings cannot be recomputed on MAD**. — [GitHub Soldelli/MAD](https://github.com/Soldelli/MAD)

**Ego4D Episodic Memory – NLQ**
- 11.3k / 3.9k / 4k queries over 136 / 45 / 46 h of train / val / test video. Clips average 8.2 min and GT responses average 10.5 s **[snippet]**. — [NaQ, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Ramakrishnan_NaQ_Leveraging_Narrations_As_Queries_To_Supervise_Episodic_Memory_CVPR_2023_paper.pdf); [arXiv 2301.00746](https://arxiv.org/pdf/2301.00746)
- Metrics: R@1 and R@5 at IoU {0.3, 0.5}. Supervised challenge entries reach only about 25–28.5 R@1@0.3 and 18–20 R@1@0.5 **[snippet]**. — [ObjectNLQ 2024](https://arxiv.org/html/2406.15778v2); [OSGNet 2025](https://arxiv.org/html/2506.03710v1)
- Access requires signing the Ego4D license agreement **[bg]**.

**Typical training-free / zero-shot numbers**
- TFVTG (ECCV 2024; LLM decomposes the query into sub-events and their order, a VLM scores frames, and the predictions are integrated using that order): **Charades-STA R@0.3/0.5/0.7 = 67.04/49.97/24.32, mIoU 44.51; ActivityNet Captions 49.34/27.02/13.39, mIoU 34.10** **[fetched]**. — [TFVTG GitHub](https://github.com/minghangz/TFVTG); [arXiv 2408.16219](https://arxiv.org/pdf/2408.16219)
- TFVTG explicitly models sub-events and their temporal order, which is conceptually close to our ordered multi-event DP. It also reports cross-dataset and OOD generalization **[snippet]**. — [ECCV poster](https://eccv.ecva.net/virtual/2024/poster/729)
- Other zero-shot lines: language-free training (arXiv 2210.12977), ChatVTG (CVPR-W 2024), and 2025–2026 MLLM training-free grounders (e.g. "Your VLM Already Knows When", arXiv 2608.08315) **[snippet; numbers not extracted]**. — [arXiv 2210.12977](https://arxiv.org/pdf/2210.12977); [ChatVTG](https://openaccess.thecvf.com/content/CVPR2024W/PVUW/papers/Qu_ChatVTG_Video_Temporal_Grounding_via_Chat_with_Video_Dialogue_Large_CVPRW_2024_paper.pdf); [arXiv 2608.08315](https://arxiv.org/pdf/2608.08315)

**Multi-event / ordered-query benchmarks**
- **SynopGround** (ACM MM 2024): more than 2,800 h of TV dramas with human-written synopses. It defines Multi-Paragraph Video Grounding, where each paragraph in a sequence is grounded to an interval **[snippet]**. — [arXiv 2408.01669](https://arxiv.org/abs/2408.01669)
- **CoMET-Bench** (2026): 2,789 queries over 600 videos averaging 33.8 min, covering conditional multi-event grounding with temporal and spatial conditions plus a negative-query subset **[snippet]**. — [arXiv 2606.15320](https://arxiv.org/html/2606.15320)
- **HCMC AI Challenge TRAKE** task: retrieve a video containing an entire described sequence of events and align each event to its keyframe, requiring the order A-before-B **[snippet]**. AIC 2024 used 1,471 videos / 328 h, with KIS-text, KIS-visual, QA and TRAKE tasks **[snippet]**. — [AIC 2024 overview, Springer](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_1); [MADTempo arXiv 2512.12929](https://arxiv.org/html/2512.12929)
- MADTempo (an AIC 2025 system) decomposes a TRAKE query into a context C plus ordered events E1..En. It **does not disclose** the TRAKE scoring formula or keyframe tolerance **[fetched]**. — [arXiv 2512.12929](https://arxiv.org/html/2512.12929)
- Procedure-step datasets: CrossTask has about 4.7K videos and 83 tasks; COIN has 11,827 videos, 180 tasks and about 476 h. They have ordered step annotations, but the queries are short step labels, not free-form descriptions **[bg]**. — [CrossTask arXiv 1903.08225](https://arxiv.org/abs/1903.08225); [COIN arXiv 1903.02874](https://arxiv.org/abs/1903.02874)

### Inferences
- There is no public benchmark that combines all three of our properties: a corpus-level search space, free-form English ordered multi-event queries, and per-event GT. The closest pieces are ActivityNet Captions paragraphs (ordered, per-sentence timestamps, single video), TVR (corpus-level VCMR), and AIC TRAKE (corpus plus ordered events, but no public GT or scoring spec found).
- Scores below 30% R@1 on long-video benchmarks are normal. Our absolute numbers on an 873-video corpus should not be compared directly with single-video grounding scores.

### Gaps
- Exact QVHighlights / DiDeMo / ActivityNet statistics and licenses were not re-fetched (marked [bg]).
- VCMR training-free numbers on TVR (e.g. CLIP zero-shot VCMR) were not found in this session.
- The official AIC TRAKE scoring formula and tolerance were not found publicly.

## 2. Best-matching public benchmark and cost to evaluate our pipeline on it

### Takeaway
For a cheap external sanity check, the recommended choice is **ActivityNet Captions (val_2 or a sampled subset) run in a corpus setting**. It is the only widely used dataset with ordered multi-sentence per-video annotations. Charades-STA and QVHighlights val are the next options for single-scene grounding. MAD cannot be used with our encoders because raw video is not distributed. TVR and Ego4D are access-gated. On a 6 GB RTX 4050 or a Colab T4, embedding computation for these subsets is estimated at tens of minutes to a few hours. Video download and decoding are expected to dominate.

### Cited Findings
- ActivityNet Captions provides ordered sentences with timestamps per video (paragraph structure) **[bg]**. — [arXiv 1705.00754](https://arxiv.org/abs/1705.00754)
- TFVTG gives a directly comparable training-free protocol and numbers on Charades-STA and ActivityNet Captions (R@0.3/0.5/0.7, mIoU). Its code includes a feature-extraction script for custom videos **[fetched]**. — [TFVTG GitHub](https://github.com/minghangz/TFVTG)
- Charades-STA test: 3,720 query–moment pairs **[snippet]**. — [arXiv 2207.14698](https://arxiv.org/abs/2207.14698)
- QVHighlights annotations are CC BY-NC-SA 4.0 (non-commercial, so acceptable for research) **[fetched]**. — [Moment-DETR GitHub](https://github.com/jayleicn/moment_detr)
- MAD's copyright constraint means only precomputed features are released, so our embedding stack cannot be reproduced there **[bg]**. — [GitHub Soldelli/MAD](https://github.com/Soldelli/MAD)
- Ego4D NLQ val is 45 h of video, but access needs an Ego4D license **[snippet]**. — [NaQ](https://arxiv.org/pdf/2301.00746)

### Inferences (cost estimate; my own arithmetic, not sourced)
- Our corpus density is 335k keyframes / 873 videos ≈ 384 keyframes per video. Assuming a similar shot-based density of about 0.3–0.5 keyframes/s:
  - **Charades-STA test** (about 1.3K videos × about 30 s ≈ 11 h): about 12–20k keyframes.
  - **QVHighlights val** (about 1.5K videos × 150 s ≈ 65 h): about 70–120k keyframes.
  - **ActivityNet Captions val_2** (about 4.9K videos × about 2 min ≈ 160 h): about 170–290k keyframes, comparable to our own corpus.
- Assumed throughput for a ~400M-parameter ViT at 384 px in fp16 is roughly 100–200 images/s on a T4 and similar or somewhat less on a 6 GB RTX 4050 at batch sizes of 32–64. On that basis, SigLIP 2 plus BEiT-3 for 100k keyframes is about 20–60 min of GPU time. BLIP-2 captioning is far slower (roughly 5–15 captions/s), about 2–6 h per 100k frames. **Recommendation:** skip captions or subsample for the external benchmark. These throughput figures are estimates; benchmark 1,000 frames first.
- Shot detection and decoding (TransNetV2 or similar) plus YouTube download availability are likely the real bottleneck. Many ActivityNet and QVHighlights YouTube IDs have gone dead, and mirrors are needed.
- The protocol should state that a corpus setting over an entire split (every query searches all split videos) turns single-video grounding datasets into VCMR, as in [arXiv 2008.08716](https://arxiv.org/pdf/2008.08716) and [arXiv 1907.12763](https://arxiv.org/pdf/1907.12763). Report single-video grounding (for comparison with TFVTG) and corpus VCMR separately.
- Sparse keyframes put a ceiling on IoU. With keyframes about 2–3 s apart, 4–10 s moments cannot reach high IoU. An "oracle-keyframe upper bound" (the best achievable IoU given the keyframe grid) should be reported.

### Gaps
- Exact current download success rates for ActivityNet and QVHighlights were not found.
- No measured SigLIP 2 so400m or BEiT-3 throughput on a T4 or RTX 4050 was found.

## 3. Metrics for our own query set, and how interactive-retrieval (VBS/AIC/KIS) studies evaluate

### Takeaway
Report a layered set of metrics:
- **Video-level:** R@1/5/10 and MRR.
- **Frame-hit:** R@K@±τ, where a returned keyframe counts as a hit if its timestamp falls within [s−τ, e+τ]. Use several values of τ.
- **Span:** R@1@IoU{0.3,0.5,0.7}, mIoU, plus VCMR R@K@IoU requiring the correct video.
- **Multi-event:** the share of events hit, all-events-in-order accuracy, and mean per-event IoU.

VBS KIS judges a submitted frame or segment as correct if it falls inside the target segment (with tolerance). Its score combines time and wrong submissions, which is an interactive-only concern.

### Cited Findings
- VCMR R@k, IoU=μ counts a prediction as correct only if both video and moment match (IoU > μ) within the top k **[snippet]**. — [TVR ECCV PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660443.pdf)
- VBS KIS: the server checks the submitted frame number or segment for correctness. The score drops linearly with time and is penalized for wrong submissions to discourage trial and error. Task types are KIS-visual (5 min), KIS-textual (7 min) and KIS-conversational (7 min) **[snippet]**. — [VBS live evaluation](https://www.researchgate.net/publication/263659671_The_Video_Browser_Showdown_A_Live_Evaluation_of_Interactive_Video_Search_Tools); [VBS 2025 results arXiv 2509.12000](https://arxiv.org/pdf/2509.12000)
- VBS 2025 uses the V3C collection and judges submissions with tolerance thresholds. The exact tolerance was not extracted **[fetched, low detail]**. — [arXiv 2509.12000](https://arxiv.org/pdf/2509.12000)
- Remote vs. on-site interactive evaluation comparisons for VBS/LSC exist (distributed evaluation) **[snippet]**. — [Springer IJMIR 2021](https://link.springer.com/article/10.1007/s13735-021-00225-2)
- DiDeMo's handling of multiple annotations (a prediction must agree with at least 2 of 4 annotators) provides a template for tolerance to boundary ambiguity **[snippet]**. — [arXiv 1708.01641](https://arxiv.org/pdf/1708.01641)
- QVHighlights allows multiple GT moments per query and uses mAP averaged over IoU thresholds, which rewards retrieving all valid moments **[bg]**. — [arXiv 2107.09609](https://arxiv.org/abs/2107.09609)
- AIC TRAKE requires correct order and per-event keyframe alignment in a single video **[snippet]**. — [AIC 2024, Springer](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_1)

### Inferences (proposed metric definitions)
- **Video R@K / MRR:** rank of the first correct video. Use this as the primary metric for the retrieval stage.
- **Frame-hit R@K@τ:** at least one of the top-K (video, keyframe) results falls inside [s−τ, e+τ] of a GT moment. Suggested τ ∈ {0, 2 s, 5 s}; τ = 0 is strict. Because keyframes are shot-based and irregular, also report a **"reachable" flag** (whether any keyframe exists in [s−τ, e+τ]). Exclude unreachable queries or report them separately so that encoder quality is not confounded with the keyframe sampler.
- **Span metrics:** after temporal reranking and DP, compute the predicted span [ŝ, ê] from the first and last aligned keyframes. Optionally pad each end by half the local keyframe gap, and report both padded and unpadded. Report R@1@IoU{0.3, 0.5, 0.7} and mIoU (mIoU = 0 if the video is wrong) for VCMR, and the same conditioned on the correct video, which isolates localization.
- **Ordered sequences (43) and multi-event queries (5):**
  - (a) per-event hit rate: each event's aligned frame falls inside its GT interval ± τ;
  - (b) all-events-hit **and** order-correct (strict success);
  - (c) Kendall's τ between predicted and GT event order when events are not all hit;
  - (d) mean per-event IoU if spans are predicted.
  With 5 multi-event queries, report per-query descriptive results only, with no significance claims.
- Report every metric overall and per stratum (single / sequence / multi-event), with counts n = 31 / 43 / 5.

### Gaps
- The exact VBS 2025 and AIC 2024/2025 tolerance windows and scoring formulas were not retrieved. The VBS server (DRES) documentation would be the next source to check.

## 4. Statistics with ~79 queries: CIs, paired tests, effect sizes, multiple comparisons, topic counts, dev/test splitting

### Takeaway
With 79 queries, a binary metric near 50% has a 95% CI of about ±11 percentage points. Only large differences (roughly ≥10–15 pp, depending on how often paired outcomes disagree) are detectable. Recommended practice:
- Use **paired** tests: a paired randomization/permutation test or t-test on per-query scores, and exact McNemar for binary hits.
- Report bootstrap CIs of the paired difference.
- Control the family-wise error rate with Holm, or the false discovery rate with Benjamini–Hochberg, across ablations.
- Avoid Wilcoxon and sign tests, and treat most ablations as exploratory.
- Freeze hyperparameters on a dev split, or on a separately written query batch, before touching the test queries.

### Cited Findings
- Smucker, Allan & Carterette (CIKM 2007) on TREC runs: randomization, bootstrap and t-tests show little practical difference. Wilcoxon and sign tests detect significance poorly, can produce false detections, and "their use should be discontinued" for differences in means **[snippet]**. — [CIKM 2007 PDF](https://ciir-publications.cs.umass.edu/getpdf.php?id=744); [ACM DL](https://dl.acm.org/doi/10.1145/1321440.1321528)
- Follow-up work (SIGIR 2009) found that the tests disagree more as topic counts fall toward 10 **[snippet]**. — [ACM DL 1571941.1572050](https://dl.acm.org/doi/10.1145/1571941.1572050)
- Urbano, Lima & Hanjalic (SIGIR 2019, simulation study), on Type I/II/III errors:
  - The t-test and permutation test keep Type I error at α across measures and topic-set sizes.
  - The bootstrap-shift test is biased toward small p-values.
  - Wilcoxon and sign tests are unreliable.
  - Type III (wrong-direction) errors reach about 2% for P@10 and RR with small topic sets.
  - Behavior degrades at 25 topics, is robust at 50, and approaches ideal at 100.
  - The t-test is recommended for mean effectiveness **[fetched]**. — [arXiv 1905.11096](https://arxiv.org/html/1905.11096v2)
- A 2026 critique argues against the Wilcoxon test in IR ("Stop Using the Wilcoxon Test") **[snippet only]**. — [arXiv 2604.25349](https://arxiv.org/pdf/2604.25349)
- Sakai's topic set size design sets the number of topics from a desired power, or from a cap on CI width for the difference in means (paired t-test, CI-based, and ANOVA variants). It found that reasonable power often needs about 200 topics or more, versus the usual 50–100 **[snippet]**. — [Sakai, IRJ 2016](https://link.springer.com/article/10.1007/s10791-015-9273-z); [Paired/unpaired, ICTIR 2018](https://dl.acm.org/doi/10.1145/3234944.3234971); [Sakai book 2018](https://link.springer.com/book/10.1007/978-981-13-1199-4); [Fewer topics? IRJ 2019](https://link.springer.com/article/10.1007/s10791-019-09357-w)
- Sakai's SIGIR 2016 tutorial/paper "Statistical Significance, Power, and Sample Sizes" surveys under-powered IR experiments **[snippet]**. — [ACM DL](https://dl.acm.org/doi/10.1145/2911451.2911492)
- Different measures have very different per-topic variances, so the number of topics required depends on the metric. Binary or RR-type metrics need more topics **[snippet]**. — [Sakai IRJ 2016](https://link.springer.com/article/10.1007/s10791-015-9273-z)
- A multifaceted examination of significance tests (Sanderson et al., WSDM 2022) is a further reference **[snippet]**. — [WSDM 2022 PDF](https://marksanderson.org/files/papers/WSDM2022.pdf)
- Charades-STA's train/test temporal-bias problem shows how implicit tuning to a split's distribution inflates results. Separate OOD test splits were introduced for this reason **[snippet]**. — [arXiv 2207.14698](https://arxiv.org/abs/2207.14698)

### Inferences (concrete protocol; arithmetic is mine)
- **Precision of the estimates:** SE of a proportion p with n = 79 is √(p(1−p)/79), which gives about ±11 pp (95%) at p = 0.5 and about ±8.8 pp at p = 0.2 or 0.8. Per stratum it is much wider: n = 31 gives about ±18 pp, and n = 43 about ±15 pp. **Report 95% CIs** (percentile or BCa bootstrap over queries, 10k resamples; Wilson interval for proportions) on every headline number.
- **Paired comparisons of system A vs. B:**
  - For binary hit metrics, use the exact McNemar test (a binomial test on discordant pairs), or equivalently a paired permutation test (randomly sign-flip per-query differences, ≥10k or exact permutations).
  - For continuous metrics (RR, IoU), use a paired permutation test or paired t-test, per Urbano 2019.
  - Report the mean paired difference with a bootstrap CI as the effect size. Also report win/tie/loss counts per query, and optionally Cohen's d_z (mean difference / SD of differences).
- **Power intuition:** McNemar power depends on the number of discordant queries. If two variants disagree on only about 10 queries, even a 8–2 split is not significant at α = 0.05 (two-sided exact p ≈ 0.11). Stated plainly, most small-tweak ablations will be **inconclusive**, and the write-up should say so.
- **Multiple comparisons:**
  - Pre-register a small set of *confirmatory* comparisons, for example full pipeline vs. (i) SigLIP-only, (ii) no temporal reranking, (iii) no DP alignment. Control the FWER on those with Holm–Bonferroni.
  - Treat the remaining ablations (fusion weights, τ, top-k) as exploratory with Benjamini–Hochberg FDR, or report CIs only.
  - An alternative for many systems is Tukey HSD or a randomized Tukey HSD permutation test, as used by Carterette and Sakai in IR **[bg]**.
- **Dev/test discipline:**
  - Split the 79 queries in a stratified way (e.g. about 25 dev / about 54 test, keeping the multi-event 5 in test or reporting them descriptively). Tune fusion weights, τ and DP penalties on dev only, freeze the configuration with a git hash, then run test once.
  - Better still, write a second batch of held-out queries after freezing, since the current 79 were likely seen during development.
  - Use k-fold cross-validation over queries only for picking hyperparameters, and report the averaged held-out fold performance.
- **Topic count:** the IR literature suggests that 79 queries sits near the "50-topic" regime. That is adequate for Type I control with t or permutation tests (Urbano 2019), but underpowered for small effects (Sakai's ≥200 guidance). Target 150–200 queries if statistical claims about small differences matter.

### Gaps
- No video-retrieval-specific study of the number of queries needed was found. The guidance is transferred from text IR (TREC).
- No source was found on the power of McNemar with small n in retrieval specifically (standard statistics apply).

## 5. Annotation protocol for GT intervals: agreement and multiple valid moments

### Takeaway
Temporal-boundary ambiguity is substantial and well documented. Established datasets handle it in three ways:
- collecting multiple annotators and scoring against a consensus (DiDeMo: ≥2 of 4);
- allowing multiple disjoint GT moments per query and scoring with mAP (QVHighlights);
- measuring inter-annotator agreement as the temporal IoU between annotators.

For our set: double-annotate at least a subset, report the mean pairwise tIoU and the share of pairs with IoU ≥ 0.5, store all valid moments per query, and adjudicate disagreements.

### Cited Findings
- DiDeMo collected multiple human boundary annotations per description (4) and counts a prediction as correct when it matches at least 2 of them. A verification step made sure each description refers to a single distinct moment (a referring expression) **[snippet]**. — [arXiv 1708.01641](https://arxiv.org/pdf/1708.01641)
- TVR applied strict annotator qualification plus post-annotation quality verification, and labeled query type (video / subtitle / both) **[snippet]**. — [ECCV 2020 PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660443.pdf)
- QVHighlights annotates *all* query-relevant moments (possibly disjoint) plus per-clip 1–5 saliency from multiple annotators **[fetched: saliency and relevant moments; bg: multiple annotators]**. — [arXiv 2107.09609](https://arxiv.org/abs/2107.09609)
- Charades-STA labels were built semi-automatically from Charades activity segments plus sentence decomposition, and carry a strong location bias. It is a caution that GT construction choices bias evaluation **[bg; bias part snippet]**. — [arXiv 2207.14698](https://arxiv.org/abs/2207.14698)
- Otani et al. (BMVC 2020), "Uncovering Hidden Challenges in Query-Based Video Moment Retrieval", re-annotated Charades-STA/ActivityNet samples and reported notable annotator disagreement on boundaries, along with bias-only baselines that are competitive **[bg]**. — [arXiv 2009.00325](https://arxiv.org/abs/2009.00325)

### Inferences (proposed protocol)
- **Schema per query:** `query_id`, type, a list of valid `(video_id, start, end)` moments (allowing multiple), and for sequence and multi-event queries an ordered list of `events[{text, video_id, start, end}]`. Add `ambiguity_note` and an `annotator` field. Record boundaries in seconds on the source video timeline, not keyframe indices, and derive keyframe membership afterwards.
- **Guidelines:** the start is the first frame where the described content is visible and identifiable, and the end is the last such frame. For shot-based corpora, snap to shot cuts only if the description matches the whole shot. Write down in advance how to handle repeated occurrences (list all of them) and partial matches (exclude them, or mark them "partial").
- **Agreement:** a second annotator labels ≥20–30% of queries independently (all 5 multi-event queries plus a stratified sample). Report:
  - video-level agreement (%, Cohen's κ if applicable);
  - mean pairwise tIoU and the share with tIoU ≥ 0.5 / 0.7;
  - boundary deviation (median |Δstart|, |Δend| in seconds);
  - for events, order agreement.

  Use the observed median boundary deviation to justify the frame-hit tolerance τ.
- **Scoring against multiple annotators:** take the max IoU over all valid GT moments (the QVHighlights/TVR convention), or require agreement with ≥2 annotators (DiDeMo), and state which one is used.
- **Completeness:** because the 873-video corpus may contain unlabeled valid matches (the corpus-level false-negative problem), run a pooled post-hoc check. Review the top-10 of all systems for missed valid moments, as in TREC pooling. Report how many GT additions were made, and make them before the final comparison.

### Gaps
- Numeric inter-annotator tIoU values from the TVR, QVHighlights and Charades-STA papers were not extracted in this session. Otani et al. 2020 numbers are from background knowledge and should be verified.
