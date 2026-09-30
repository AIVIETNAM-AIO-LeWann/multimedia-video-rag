# Top interactive video retrieval systems (VBS 2022–2026, AIC HCMC 2023–2025), relevance feedback, and Vietnamese-query handling

Context for reader: our corpus is 873 Vietnamese videos / 335k keyframes, SigLIP 2 + BEiT-3 on CPU FAISS; tasks KIS, QA, TRAKE. Notes flag where a claim comes only from a team's own system description (self-reported) vs. organizer post-evaluation.

## Q1. Which systems won/ranked top at VBS 2022–2026 and what are their pipelines?

### Takeaway
Winners: vibro (2022, 2023), VISIONE (2024), NII-UIT (2025), PraK V4 (2026). All top systems are built around CLIP-family joint embeddings (OpenCLIP ViT-L/14/H-14, BEiT-3, CLIP2Video, ALADIN, SigLIP variants). They use score-normalized late fusion of several models, two-scene temporal queries, and a fast grid browser. Since 2024–2025 they add LLM query expansion and VQA. The model is no longer what separates teams. The difference comes from interaction design: temporal queries, browsing, and feedback.

### Cited Findings
- **Winners per year (organizer "Hall of Fame")**: VBS 2022 overall winner vibro, with Best AVS IVIST, Best KIS-Visual VISIONE, Best KIS-Textual Charles Univ. (CVHunter) and Best Newcomer AVSeeker. VBS 2023: Best System vibro ("Video Browsing with Semantic and Visual Image Embeddings"), runner-up VISIONE. VBS 2024: overall VISIONE 5.0, Best T-KIS/Experts vibro, Best QA/Novices diveXplore, Best V-KIS/Experts PraK. VBS 2025: overall + best expert NII-UIT, best novice VEAGLE (eye-gaze). VBS 2026: 1st PraK V4, 2nd NII-UIT ("Towards Effective VQA for Interactive and Multimodal Video Retrieval"), 3rd Exquisitor ("Temporal Queries Revisited"). — [VBS Hall of Fame](https://videobrowsershowdown.org/hall-of-fame/)
- **VBS 2024 ranking**: 1 VISIONE, 2 vibro, 3 PraK, 4 ViewsInsight, 5 diveXplore, with 12 teams. Tasks were KIS-V (5 min), KIS-T (7 min), AVS and QA (new in 2024). About 2,400+ h of video. Team members were scored individually, and expert and novice sessions were combined. — [Results of the 2024 VBS, arXiv 2502.15683](https://arxiv.org/html/2502.15683v1)
- **VBS 2025 ranking**: 1 NII-UIT, 2 PraK Tool V3, 3 diveXplore, 4 Exquisitor, 5 VIREO, 6 SnapSeek 2.0, 7 VERGE, 8 VEAGLE, 9 ViewsInsight 2.0, 10 IMSearch 2.0. There were 17 teams and 37 participants. Conversational KIS (KISC) was new. Data was the full V3C (3,800 h, 28,450 videos) plus MVK2 (marine) and LapGyn (medical, 100 h). — [Results of the 2025 VBS, arXiv 2509.12000](https://arxiv.org/html/2509.12000v1)
- **VISIONE (2024 winner)**: late fusion of three cross-modal extractors: OpenCLIP ViT-L/14 (LAION-2B), CLIP2Video and ALADIN. It also has object/color metadata search on a spatial canvas and temporal search. For VBS 2024 it added GPT-4-based query rephrasing/summarization (self-reported). — [VISIONE 5.0, MMM 2024](https://dl.acm.org/doi/10.1007/978-3-031-53302-0_29); summary via [search result/ResearchGate](https://researchgate.net/publication/377753034_VISIONE_50_Enhanced_User_Interface_and_AI_Models_for_VBS2024). At VBS 2022, about 83% of VISIONE's text queries were **temporal** (two-scene) queries. VISIONE evaluates queries "on the fly" at every UI interaction, which gave 5.4 queries/min (organizer analysis). — [Lokoč et al., Multimedia Systems 2023 (Zenodo OA)](https://zenodo.org/records/8321124)
- **NII-UIT (2025 winner, UIT-VNU-HCM + NII)**, self-reported pipeline:
  - Keyframes: BEiT-3 features on every 10th frame, keeping only significantly different frames. The approach is inspired by vibro.
  - Retrieval: multiple VLMs (BEiT-3, OpenCLIP H-14 and others). Scores are normalized per model, then fused by mean pooling.
  - Query expansion: GPT-4o proposes 5 paraphrases. These run in parallel, and results can be viewed separately or fused.
  - Visual queries: Stable Diffusion generates example images as queries.
  - Object filtering: Co-DETR COCO detections filter results after fusion.
  - Temporal: "dynamic temporal search", adapted from vitrivr. It scores shots both before and after the previous result, rather than enforcing a strict order, and reranks by aggregated scores for KIS-T. — [NII-UIT at VBS2025, MMM 2025](https://link.springer.com/chapter/10.1007/978-981-96-2074-6_38) (full text read locally)
- **PraK V4 (2026 winner, Charles Univ./Konstanz; formerly CVHunter/SOMHunter lineage)**: advances are spatial conjunction of localized queries (AND across regions), semi-automated UI adaptation for AVS via online learning, a single-video browser with within-video querying, a parallelized backend and a new keyframe layout. A search snippet says PraK uses a fine-tuned "MCIP-SigLIP" variant; I could not verify this from the full text. — [PraK V4 at VBS 2026](https://link.springer.com/chapter/10.1007/978-981-95-6963-2_25) (abstract-level). The PraK backend supports multi-query temporal search and Bayesian relevance feedback from keyframes. — [Jäckl et al., "Can Agents Win the VBS?", arXiv 2609.07311](https://arxiv.org/html/2609.07311)
- **diveXplore (top-5 2024, 3rd 2025)** integrates OpenCLIP (LAION-2B) and a query server that runs and merges parallel queries, including temporal ones. — [diveXplore at VBS 2024, arXiv 2508.20560](https://arxiv.org/pdf/2508.20560)
- **VBS 2022 tool landscape (older, 2022)**: nearly all top tools used CLIP-derived text encoders. VISIONE additionally used CLIP2Video; others used W2VV++. Temporal query forms varied:
  - vibro: two-query on-the-fly temporal fusion.
  - CVHunter: context-aware ranker.
  - VERGE: concept-only temporal queries.
  - vitrivr: dedicated temporal mode.
  - Exquisitor: two relevance-feedback models plus temporal constraints.
  
  — [Lokoč et al. 2023](https://zenodo.org/records/8321124)

### Inferences
- The recurring winning recipe is: (1) 2+ strong image-text embeddings fused late after per-model score normalization; (2) a two-scene temporal query with a tolerant time window; (3) LLM paraphrase expansion; (4) object/OCR/ASR filters as secondary modalities; (5) a fast, dense result grid with video-level grouping and a within-video browser. Our SigLIP 2 + BEiT-3 stack matches the model choice of the 2025 winner, which used BEiT-3 plus an OpenCLIP-family model.

### Gaps
- I could not access the full texts of VISIONE 5.0, PraK V4, vibro 2023 or the NII-UIT VBS2026 paper (Springer paywall). Exact model lists for 2026 (e.g., whether SigLIP 2 is used) are unverified.
- The VBS 2023 official results/analysis paper was not located in this pass.

## Q2. What do VBS post-evaluation analyses show correlates with success?

### Takeaway
Across 2022 and 2026, organizer log analyses show the same pattern. Success comes mostly from rapidly reformulating text queries against a strong CLIP-class model and inspecting the result grid. Advanced features (feedback, temporal, spatial) decide only about 10% of cases, mostly on homogeneous datasets. Browsing failures are a major loss source: the target was already ranked high but was missed.

### Cited Findings (VBS 2022 post-evaluation, Multimedia Systems 29(6), 2023; 16 teams) — [Lokoč et al. 2023, OA PDF](https://zenodo.org/records/8321124)
- The top 4 teams scored >230 points. vibro got 100/100/100 in KIS-V/KIS-T/AVS; CVHunter and VISIONE got similar KIS scores (KIS-T 100 and 90, KIS-V 96 and 100) but less in AVS (81 and 74). The top 3 "solved all visual KIS tasks and almost all textual KIS tasks". The conclusion confirms "the effectiveness and reign of joint-embedding approaches, where CLIP-based models demonstrate impressive performance".
- Share of KIS queries that were text: vibro 69.9%, CVHunter 82.2%, VISIONE 78.5%, VERGE 88.6%, vitrivr 82.1%, vitrivr-VR 96.9%. Image/relevance-feedback use was second: vibro 27.8%, CVHunter 17.8%.
- Hit rates per text query (target shot in top-K) were low for single queries:
  - vibro: 7.3% top-10, 29.3% top-100, and 43.1% outside the top-1000.
  - VISIONE: 9.7% top-10, 26.0% top-100.
  - VERGE: 13.4% top-10, 32.1% top-100.
  - vitrivr-VR: 0.0% top-10, 80% outside the top-1000.
  
  Success therefore came from multiple reformulations, not single queries.
- **Relevance feedback evidence**: CVHunter's Image (feedback) queries put the target in the top-100 43% of the time, vs 19% for its text queries. Authors caution that the feedback is "pre-conditioned" on prior text results.
- Across all 23 KIS tasks, the best user's best rank of a correct item was below 100 in about 87% of tasks for vibro and CVHunter, 78% for VISIONE, 68% for VERGE, 65% for vitrivr and 30% for vitrivr-VR. The second user was much worse (52% / 35% / 48% / 9%), so the user effect is large.
- **Query length**: for vibro, CVHunter and VERGE, queries that got the target into the top-1000 were significantly longer (t-test p ≤ 0.04). For VISIONE (mean 21.2 words, 104 chars) length made no difference. Mean words/query: vibro 8.1, CVHunter 9.1, VERGE 4.5.
- Text reformulation "may lead to some notable improvements" from the first to the last query. However, logs show "numerous browsing errors where correct shots were within top-10 or top-100, but queries were reformulated anyway". Examples include a correct video in the top-10 that was never submitted (vibro T2, VISIONE T11, VERGE V10, vitrivr T1/T8). Browsing failures were most frequent in textual KIS. In 5 cases the rank was below 25 but submission took more than 2 minutes.
- Query density fell from about 3.6 queries/min in the first minute to about 2.1 in the last. Teams shift from querying to browsing over time.
- AVS had no single dominant team. vibro, IVIST, VIREO and CVHunter were strongest.

### Cited Findings (VBS 2026 interaction-log analysis)
- The first VBS edition in which two systems had full interaction logging. Result: "a clear dominance of iterative, high-frequency text query reformulation with result set inspection, leveraging the power of modern CLIP-based models across most competition categories". "In around 10% of cases, users also relied on advanced system features … mostly on challenging homogeneous datasets." — ["What Drove Success at the 15th VBS?", ACM (abstract via search)](https://doi.org/10.1145/3805622.3810635); full text not accessible (403).

### Cited Findings (agentic VBS replay, 2026)
- A fully autonomous Qwen VLM multi-agent system on the PraK backend was run on 81 VBS 2024–2026 tasks:
  - t-KIS: 9/11 solved per run. On 8 of the 9 tasks that experts also solved, at least one agent run was faster than the fastest expert.
  - AVS: matched or beat the best expert in 8/13 tasks.
  - VQA: 11–12/16 solved vs 12–14 for experts.
  - Visual KIS on the homogeneous MVK/LHE collections: failed. In 94.8% of failed runs (146/154) the target never appeared in candidates.
  - When the browser found nothing relevant, the planner reformulated the query 78.2% of the time, vs 18.4% within-video queries and 3.4% Bayes updates.
  
  — [Jäckl, Vopálková, Keim, Lokoč, arXiv 2609.07311](https://arxiv.org/html/2609.07311)

### Inferences
- For our KIS/QA/TRAKE on news-like Vietnamese video (heterogeneous, OCR-rich), the evidence points to three priorities: (a) fast text query → dense grid loops; (b) grouping results by video with a temporal-context strip to cut browsing misses; (c) cheap query reformulation/expansion. Feedback and advanced tools act as a tail-case safety net and are not the main driver.
- A single query rarely puts the target in the top-10, so first-page recall numbers from offline benchmarks understate interactive success. Benchmarks should also track rank after 2–3 reformulations.

### Gaps
- The VBS 2022 extended post-evaluation (IJMIR 2024) and the VBS 2023 analysis were not retrieved. I found no per-feature ablation numbers for VBS 2024/2025 (those results papers focus on scores and methodology).

## Q3. Relevance feedback in interactive KIS: measured gains?

### Takeaway
Bayesian (PicHunter/SOMHunter-style) feedback on CLIP-class embeddings works in simulation. It lifts about 40–60% of targets to rank 1 within 7 rounds, but it assumes users judge similarity the way the model does. Robust variants help further. In live competition, feedback is a secondary tool, and agents rarely choose it.

### Cited Findings
- **Robust Relevance Feedback for Interactive KIS** (Ma & Ngo, ICMR 2025) evaluates Bayesian feedback on V3C1/V3C2 (about 1.08M and 1.43M clips) with a simulated user:
  - Classic PicHunter still boosts **40.05%** of targets initially ranked beyond 1,000 to top-1 within 7 rounds.
  - With a random user model, Recall@1 goes from about 0.02 to 0.23 after 7 steps.
  - PicHunter goes from 0.0187 to 0.49.
  - The proposed pairwise-judgment + multi-sub-perception model reaches 0.5467.
  - In the 10–50 initial-rank band: more than 60% of targets reach rank 1 (PicHunter about 0.58).
  - In the 1,000–5,000 band: more than 40%.
  - Search-space pruning raises Recall@1 from 0.547 to 0.638.
  - On 17 VBS textual-KIS queries over 3 rounds, R@1 was 10/14/16 of 17 vs PicHunter 9/11/14 and random 6/8/8. R@10 reached 17/17 by round 2.
  
  Caveat: this is simulation, not live users. — [arXiv 2505.15128](https://arxiv.org/pdf/2505.15128)
- At VBS 2022, CVHunter's feedback queries hit the top-100 43% of the time vs 19% for text (live logs, but conditioned on prior text). Other feedback designs: vibro used relevance feedback only for AVS; Exquisitor uses interactive learning (linear SVM-style classifiers over semantic features) as its primary mechanism. — [Lokoč et al. 2023](https://zenodo.org/records/8321124)
- Exquisitor: interactive learning over collections of more than 100M items at sub-second latency (system description). It placed 4th at VBS 2025 and 3rd at VBS 2026 with a temporal-query focus. — [Exquisitor LSC 2020 (older)](https://www.researchgate.net/publication/341908011_Exquisitor_at_the_Lifelog_Search_Challenge_2020); [Hall of Fame](https://videobrowsershowdown.org/hall-of-fame/)
- An earlier combination of VIRET and SOMHunter pairs a context-aware text ranker with Bayesian-like feedback (MMM 2022, older). — [Video Search with Context-Aware Ranker and Relevance Feedback](https://link.springer.com/chapter/10.1007/978-3-030-98355-0_46)
- In agent replays, Bayes updates were chosen in only 3.4% of recovery actions vs 78.2% query reformulation. — [arXiv 2609.07311](https://arxiv.org/html/2609.07311)

### Inferences
- Cheap to add on CPU FAISS: a Rocchio-style query-vector update (q' = αq + βmean(pos) − γmean(neg)) or a PicHunter-style Bayesian update restricted to the current top-N candidates (pruning helps). The expected payoff is highest when the target is already in the top 50–1,000 but not visible. Assume the gain is moderate and user-dependent.

### Gaps
- I found no live-user A/B measurement of Rocchio vs no-feedback in CLIP space at VBS/AIC. The Ma & Ngo paper also tests a Rocchio-like variant according to the fetch summary, but I did not verify separate Rocchio numbers.
- LSC-specific post-evaluation numbers on feedback were not retrieved.

## Q4. AIC HCMC 2023–2025 top-team systems

### Takeaway
AIC systems converge on the following:
- **Embeddings**: CLIP-family and/or BEiT-3 (often BEiT-3 + OpenCLIP/SigLIP ensembles) in Milvus/Qdrant/FAISS.
- **Vietnamese text**: OCR and ASR matched by Vietnamese text search (Elasticsearch), with translation to English for the visual encoders.
- **Query tooling**: LLM query rewriting and decomposition (GPT-4o/GPT-5/Gemini).
- **TRAKE**: dedicated ordered multi-event matching, using DP/alignment with a temporal penalty or pairwise windowing.

Team papers report competition ratings rather than controlled ablations.

### Cited Findings
- **Organizer overviews**: AIC 2023 "News Event Retrieval from Large Video Collection" (SoICT 2023) and AIC 2024 "Event Retrieval from Large Video Collection" (SoICT 2024, Springer CCIS). Both are modeled on VBS and LSC. — [AIC 2023 overview, ACM](https://dl.acm.org/doi/10.1145/3628797.3628940); [AIC 2024 overview, Springer](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_1) (full texts paywalled; stats not retrieved). The AIC 2025 theme was a "virtual assistant supporting information retrieval from large multimedia data warehouses", again in LSC/VBS format. — [Vietnam.vn news](https://www.vietnam.vn/en/gan-4-000-thi-sinh-tranh-tai-tai-vong-chung-ket-ai-challenge-tp-ho-chi-minh-2025)
- **SoICT 2024 AIC papers** (titles only): "A Hybrid Video Retrieval System Using CLIP and BEiT-3" and "MAVERICS: … CPU-Optimized Search". — [Hybrid CLIP+BEiT-3](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_17); [MAVERICS](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_12)
- **Unified moment retrieval (Tran, Nguyen-Nhu et al., arXiv 2504.08384; AIC-lineage team, AI VIET NAM)** combines coarse-grained OpenCLIP with fine-grained BEiT-3 as an ensemble. It removes near-duplicate frames with cosine >0.9 within a shot, reranks using neighbor-frame score aggregation, and runs a dual-query temporal search for start and end points. The ablation is qualitative only: BEiT-3 alone ranks the correct frame low, while ensemble + neighbor rerank "often propels correct frames into top-1 or top-5". — [arXiv 2504.08384](https://arxiv.org/abs/2504.08384)
- **Lightweight moment retrieval (arXiv 2504.09298, same group)** uses BEiT-3 for retrieval and duplicate detection, plus global reranking and "adaptive bidirectional temporal search" for start and end boundaries. — [arXiv 2504.09298](https://arxiv.org/abs/2504.09298)
- **Cascaded embedding-reranking (AIC 2025; arXiv 2512.12935)** has a first stage of BEiT-3 + SigLIP dual embeddings in Qdrant. It adds OCR from Gemini 2.0 Flash and ASR from Whisper. GPT-4o decomposes the query and predicts per-modality weights. A "Score-Reflected RRF" fuses the branches. BLIP-2 ITM reranks the top-100. The team says plain averaging or RRF can degrade results with noisy OCR/ASR. Qualifier score: 76.4/88 (86.8%) across 3 rounds. **Query expansion rule: the first expanded query is always a direct English translation "as English ensures better embedding performance from our models"** (self-reported, no ablation). — [arXiv 2512.12935](https://arxiv.org/abs/2512.12935)
- **LLandMark (AIC 2025; arXiv 2603.02888)** is a multi-agent system. CLIP ConvNeXt-XXLarge embeddings live in Milvus. Vietnamese queries are "translated into descriptive English to maximize alignment with the CLIP embedding space", while landmark names and specific Vietnamese terms stay in Vietnamese for ASR/OCR search. Landmark names are replaced by visual descriptions or by retrieved reference images. OCR is improved via PaddleOCR plus Vietnamese diacritic reconstruction. Qualifier score: 77.40/88. It ranked in the top 56 of more than 680 registered teams. — [arXiv 2603.02888](https://arxiv.org/abs/2603.02888)
- **QUEST + DANTE (AIC 2025 finalist; arXiv 2512.13169)** uses BEiT-3 embeddings in Milvus, Gemini OCR and an Elasticsearch Vietnamese analysis plugin for text metadata. An LLM enhancer rewrites queries, and exemplar images anchor unknown concepts. **DANTE for TRAKE** is DP alignment: DP[i,t] = S[i,t] + max over τ<t of (DP[i−1,τ] − λ(t−τ)). λ was tuned between 0.001 and 0.01: 0.001 worked best when event keyframe gaps were 3–15 indices, and 0.01 when gaps were 1–3. It is computed in O(T) with a running max. The final round rated them "Outstanding" on TRAKE, "Excellent" on textual KIS and "Very Good" on video KIS and QA (organizer rating as self-reported). — [arXiv 2512.13169](https://arxiv.org/abs/2512.13169)
- **MADTempo (AIC 2025; arXiv 2512.12929)** uses CLIP-LAION in Milvus with MongoDB metadata and PhoWhisper ASR. GPT-5 decomposes multi-event queries. Candidate segments are found from boundary events (first and last) and then scored by the LLM for contextual coherence. Intermediate events are aligned to maximize the total score. — [arXiv 2512.12929](https://arxiv.org/abs/2512.12929)
- **MERVIN (AIC; arXiv 2605.16120)** pairs a PE-Core-bigG-14-448 visual encoder with a Vietnamese sentence embedder (dangvantuan/vietnamese-embedding) for transcripts. Gemini Flash cleans and summarizes ASR. For TRAKE it prunes pairs where E2 comes before E1 or the gap exceeds 5 minutes, and scores videos as 10·S_pair + 5·(S̄1+S̄2). Qualifier score: 79/88. — [arXiv 2605.16120](https://arxiv.org/html/2605.16120)
- A 2024 AIC system combined CLIP ViT-L/14, TASK-former, transcripts and OCR for multilingual event queries (cited within MERVIN). — [arXiv 2605.16120](https://arxiv.org/html/2605.16120)

### Inferences
- Top AIC 2025 qualifier scores cluster at about 76–79/88. Architecture differences show up mostly in TRAKE handling and OCR/ASR quality, not in the choice of visual encoder.
- DANTE's DP with a small linear gap penalty is a directly portable TRAKE scorer for our FAISS setup. The penalty is sensitive to keyframe density, so λ must be retuned for our 335k-keyframe sampling rate.

### Gaps
- Official AIC 2023/2024/2025 final rankings and the winning teams' identities were not verified; the overview papers are paywalled. The papers found are mostly qualifier-top-N teams, not confirmed champions.
- No AIC paper found reports a controlled ablation (e.g., top-K accuracy with vs without an English translation, or BEiT-3 vs SigLIP).

## Q5. Vietnamese queries: translate to English vs multilingual encoders

### Takeaway
In practice, AIC teams translate Vietnamese to English, often with an LLM, for CLIP/BEiT-3/SigLIP visual search. They keep Vietnamese for OCR/ASR text search and proper nouns. There is no published head-to-head on AIC data. Benchmarks show that off-the-shelf multilingual encoders are weak on Vietnamese, e.g. mSigLIP-base at 28% average R@K vs about 62–65% for Vietnamese-fine-tuned CLIP/SigLIP baselines. Translation (or a Vietnamese-adapted encoder) is therefore the safer default.

### Cited Findings
- AIC 2025 teams made translation to English an explicit design choice. In one system, the first expansion is always a direct English translation "as English ensures better embedding performance". Another translates Vietnamese queries into descriptive English for CLIP but keeps Vietnamese keywords for ASR/OCR. — [arXiv 2512.12935](https://arxiv.org/abs/2512.12935); [arXiv 2603.02888](https://arxiv.org/abs/2603.02888)
- The alternative, used by MERVIN, is language-specific handling. It uses a Vietnamese text embedder for transcripts, with LLM cleanup of ASR errors, because "automatically generated transcripts are prone to recognition errors, especially in non-English languages". — [arXiv 2605.16120](https://arxiv.org/html/2605.16120)
- **ViCLIP-OT (2026)** reports average Recall@K on UIT-OpenViIC:
  - ViCLIP-OT 67.34%, ViSigLIP-OT 68.96%.
  - Baselines: CLIP 61.59%, SigLIP 64.77%.
  - Zero-shot multilingual encoders: Qwen3-VL-Embedding-2B 55.40%, Jina CLIP v2 53.91%, **mSigLIP-base 28.27%**.
  - Crossmodal-3600 (Vietnamese): ViCLIP-OT 56.85% vs CLIP 45.13%.
  
  The paper includes no translate-to-English baseline, but notes that translating introduces noise and loses language-specific meaning. — [ViCLIP-OT, arXiv 2602.22678](https://arxiv.org/html/2602.22678)
- SigLIP 2 is advertised as multilingual. — [SigLIP 2, arXiv 2502.14786](https://arxiv.org/pdf/2502.14786) (not checked for Vietnamese-specific retrieval numbers in this pass)

### Inferences
- For our SigLIP 2 + BEiT-3 stack: BEiT-3 is English-only, so it needs translation. SigLIP 2's multilingual text tower might accept Vietnamese directly, but no evidence shows it beats English translation on Vietnamese news imagery. We should run our own ablation on the evaluation set: Vietnamese raw, LLM/MT English, and fusing both. Proper nouns and on-screen text should go to OCR/ASR search in Vietnamese, not to the visual encoder.

### Gaps
- I found no study that measures translate-to-English vs multilingual CLIP/SigLIP 2 on Vietnamese video retrieval (AIC data). The ViCLIP-OT comparison is image-caption retrieval and lacks a translation baseline. SigLIP 2's Vietnamese performance was not verified.
