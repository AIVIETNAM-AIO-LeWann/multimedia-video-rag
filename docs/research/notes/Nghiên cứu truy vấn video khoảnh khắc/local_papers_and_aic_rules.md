# Local AIC/VBS papers and AIC training slides: primary evidence for an interactive text-to-video moment retrieval framework

Source keys (all local files; page = PDF page index as extracted with pypdf, not printed page number unless stated):

- [S26-1] `D:\Projects\Ref\Training\Tập huấn AIC 2026 - Buổi 1.pptx.pdf` (official BTC training 2026, SELab HCMUS, 50 pp.)
- [S26-2] `D:\Projects\Ref\Training\Tập huấn AIC 2026 - Buổi 2.pdf` (official BTC 2026, 45 pp.; pp. 30–40 image-only)
- [S26-3] `D:\Projects\Ref\Training\Tập huấn AIC 2026 - Buổi 3.pdf` (official BTC 2026, Agentic AI/LLM, 39 pp.)
- [AIO26] `D:\Projects\Ref\Training\Documents_2026-5_M01W04 - Buổi training cho AIC_HCMAI2026.pdf` (AI VIETNAM/AIO TA training for 2026, not the organizer; 24 pp.; architecture slides are images, which I rendered and read)
- [AIO25-1] `...\Training\HCMC AI Challenge (Buổi 1)_[Slide_v2]-HCMC_AI_2025.pdf` (AIO, 2025)
- [AIO25-2] `...\Training\HCMC AI Challenge (Buổi 2)-v4.pdf` (AIO, 2025)
- [AIO25-3] `...\Training\HCMC AI Challenge (Buổi 3)_[Slide]-HCMCAI_Part3.pdf` (AIO, 2025; the evaluation-metric slides pp. 8–19 are image-only, so I rendered and read them)
- [MAD] `D:\Projects\Ref\Paper\2512.12929v1.pdf` MADTempo (SOICT 2025, team AIO_Trình)
- [DANTE] `D:\Projects\Ref\Paper\2512.13169v1.pdf` Integrated Semantic and Temporal Alignment (SOICT 2025, team AIO_Owlgorithms)
- [CASC] `D:\Projects\Ref\Paper\2512.12935v1.pdf` Cascaded Embedding-Reranking and Temporal-Aware Score Fusion (AAAI 2026 workshop)
- [LLM] `D:\Projects\Ref\Paper\2603.02888v1.pdf` LLandMark (AAAI 2026 workshop)
- [NII] `D:\Projects\Ref\Paper\NII-UIT-at-VBS2025-Multimodal-Video-Retrieval-with-LLM-Integration-and-Dynamic-Temporal-Search.pdf` (MMM 2025 LNCS 15524, printed pp. 318–325 = PDF pp. 1–8)

I did not re-summarize 2504.08384 or 2504.09298. The notes below only add detail beyond `docs/research/video-retrieval-architecture-review.md`.

---

## Q1. What are the AIC 2026 task types, answer formats, scoring and time limits, and what do they imply for system design?

### Takeaway
The official 2026 BTC slides list **KIS (Video KIS + Textual KIS), AVS, VideoQA and the new Conversational KIS (KISC)**. They say nothing about TRAKE, submission format, scoring or time limits. The only detailed scoring and format rules in the local files come from the **AI VIETNAM (AIO) 2025** decks, which describe the 2025 preliminary round on Codabench. In that round each answer was `video, frame_idx` (TRAKE: `video, frame_1..frame_N`) with a binary or fractional R-Score, and the final score was the mean of best R-Score@{1,5,20,50,100}. Each query had a 5-minute limit, and scoring depended on time and on the rank of the target. Treat this as the most likely 2026 prior, not as a confirmed 2026 rule.

### Cited Findings

**Official 2026 BTC slides (task definitions)**
- The 2026 training covers context and problem, core tech (multimodal AI, CLIP, LVLM), the pipeline from indexing to retrieval, query tactics "Known-Item Search (KIS) and Ad-hoc Video Search (AVS)", data provided by the organizer, and tips — [S26-1 p.3].
- Main problem types: **KIS** ("find exactly one specific moment from a description or hint image"), **AVS** ("find all moments matching a semantic description"), **Video QA**, and **KISC (Conversational KIS)**, described as "a new technique of 2026" — [S26-1 p.7].
- KIS: one ground truth. The system "must return exactly the video segment or keyframe containing that moment" and needs "absolute accuracy at the first rank positions (Top-1, Top-5)". The example stresses tiny objects and actions lasting 1–2 s (dropping a pink teddy-bear keychain) — [S26-1 p.8].
- KIS has two sub-forms in the contest: **Video KIS** (a clip from the dataset is shown and teams must identify it) and **Textual KIS** (a text description is given) — [S26-1 p.9].
- Example with a strict co-occurrence rule: "ice cream + sea must appear together. Any beach moment or ice-cream moment not occurring together is not relevant" — [S26-1 pp.10–11]. This implies a conjunctive, single-frame co-occurrence requirement.
- AVS: return "a list of video segments sorted by semantic similarity, high to low". The example is "adult guiding children planting or watering flowers" and the slide shows Top1–Top4 results — [S26-1 pp.12–13].
- VQA: input is a long video plus a natural-language question, and the output is a short text answer that may need temporal reasoning and counting (candles blown, who gave the gift) — [S26-1 p.14].
- KISC: a multi-turn dialogue where the assistant asks clarifying questions (indoor/outdoor? male/female?). It then applies a metadata filter (last week), a space filter (outdoor cafe) and an object/colour filter (man, blue shirt) — [S26-1 p.15].
- 2026 data characteristics: "Sousveillance" or egocentric wearable/lifelog footage, shaky camera, changing light. The example is a 5-hour smart-glasses video of shopping and cooking — [S26-1 pp.4–6, 16]. Conflict: the "Data provided" slides still show HTV news metadata from 2022 (YouTube JSON with author, channel, description, keywords, length, publish_date, title, watch_url) and an HTV9 news frame — [S26-1 pp.42, 48]. The 2026 corpus domain is therefore ambiguous in the slides. Our local corpus is 873 Vietnamese videos, and the slides do not settle which domain applies.
- Provided data: Videos, Keyframes, Objects, CLIP features, YouTube metadata — [S26-1 p.41]. Objects come from **Faster R-CNN + InceptionResNetV2 trained on Open Images V4, up to 100 objects over 600 categories** — [S26-1 p.45].
- The keyframes provided are I-frames, and "the video segment to be queried will contain at least one Key-Frame" — [S26-1 p.44]. This is an important guarantee: at least one provided keyframe falls inside every GT segment.
- Technical challenges named by BTC: semantic gap; data sparsity (the KIS moment "often only lasts 2–3 seconds", and "a very fast coarse filter" is required because running big models on everything will time out); and **temporal logic constraints** (e.g., "takes off hat before entering room" vs "enters then takes off hat") — [S26-1 p.31].
- BTC reference pipeline: Video → keyframes → features (CLIP, Objects, OCR, ASR) → vector DB (FAISS, Milvus); query → LLM reasoner → query expansion → similarity search → re-ranking and display — [S26-1 p.36].
- BTC "best practice": narrow progressively in order of increasing certainty (time? place? object? scene?); try different concepts and phrasings; query expansion; image similarity; **timeline browsing** — [S26-1 p.40].
- BTC reference system modules (image slide): Query by Metadata, Query by Text Description, Filter by Date-Time, Filter by Location, Query Expansion with Visual Examples, Query Expansion with External Search Engine, Query Expansion with Sketch, Flexible Temporal Events Navigation, Map-based Visualization, Shot and Scene Clustering — [S26-1 p.33]. Preprocessing: Filtering → Normalization → Grouping (location-based clustering, image-sequence clustering, sequences of contiguous similar images) — [S26-1 p.34]. These are lifelog-style (LSC) components.
- BTC Buổi 2: the system is "retrieval models + display mechanism + user feedback", not only a strong feature extractor — [S26-2 pp.2–7]. Problems it lists: speed vs power vs cost at scale (V3C1 = 1000 h); non-unique text-to-video links, which call for more query turns, query-modality changes (e.g., sketch-generated image) and combined query types; linking frames when models are single-image — [S26-2 pp.9–13]. Display problems: near-duplicate frames within a video and too many relevant videos — [S26-2 p.15]. Feedback must balance exploration (show less related results to widen the search) against exploitation (show highly related results to separate near-identical videos), and should suggest concepts for both — [S26-2 pp.19–21]. Early fusion (GLIP, UNINEXT) needs a model run per query, so use it "on small data, in final steps". Late fusion (CLIP, OWL-ViT) suits large data but is "hard to control" — [S26-2 pp.35–43]. Prompt engineering/ensembling also appears — [S26-2 p.44].
- BTC Buổi 3 is generic agentic-AI material (reasoning CoT/ToT, episodic/semantic memory, HippoRAG, planning/world models). It cites STAR tool-augmented VideoQA (LLM planner alternating temporal tools such as clip/keyframe selection with spatial tools such as OD/OCR/zoom) and MemoriEase 2.0/3.0 (conversational lifelog search with CLIP/BLIP-2 vectors + Elasticsearch metadata, relevance-feedback re-weighting, rerank–reader RAG) as applications — [S26-3 pp.30–31]. No rules content.

**AIO 2026 deck (AI VIETNAM TA, not BTC)**
- Uses the VBS definitions: KIS-V, KIS-T, and **KIS-C where "further details are revealed after 60 seconds based on questions/chats from participants"**. VQA answers are "manually entered text", e.g., "How many nights do we see passing…" — [AIO26 pp.4–5]. The 60 s reveal is VBS's rule, cited from videobrowsershowdown.org. It is not confirmed as an AIC 2026 rule.
- The reference architecture (rendered image) is essentially the [CASC] system. Offline: keyframe extraction (TransNetV2, start/middle/end per shot) → SigLIP + BEiT-3 encoders → Qdrant; Gemini OCR + Whisper ASR → Elasticsearch. Online: text query → query decomposition → ASR text / OCR text / visual text; visual goes through SigLIP + BEiT-3 search → BLIP-2 rerank → adaptive score fusion → ranked result. A separate image query path uses the SigLIP encoder — [AIO26 pp.6–10]. The deck also stresses "with increasing data, filtering is becoming more and more important" — [AIO26 p.8].
- Reference list for 2026: the two CVPRW 2025 papers, LLandMark, the Cascaded paper, the DANTE paper and MADTempo — [AIO26 p.14]. Tips: textual query writing, a specific role per team member, "reducing search space or missing ground truth?", typing speed — [AIO26 pp.17–21].

**AIO 2025 decks (describe AIC 2025 rules)**
- AIC 2025 had two forms: traditional (interactive) and **automatic** (automatic exam between teams' assistants) — [AIO25-1 p.6]. Format: an online qualification round and an on-site final that is real-time (each query is timed), scored on effectiveness and checked for validity — [AIO25-1 p.8]. Rules: any pre-trained or commercial models are allowed; host keyframes/CLIP embeddings are allowed; laptops/PCs are allowed; phones are not — [AIO25-1 p.10].
- Automatic-track output: a "ranked list (typically top-K ≤ 100) of candidate moments [(video_id, start, end, score), …] with seconds or frame indices". Interactive track: the system returns one moment, the user gives binary or graded relevance plus optional free text, and the output is "one new candidate moment FOUND" — [AIO25-1 pp.11–12; AIO25-2 p.100].
- **"Each query (Visual KIS, Textual KIS, VQA) is given a time limit of 5 minutes. Host competition used metrics calculated by time and rank of target moment in returned topk submission."** — [AIO25-1 p.38]. The exact time-penalty formula is not given.
- Preliminary (Codabench) formats: Retrieval/KIS `[(video_id, frame_id), …]`; VQA `[(video_id, frame_id, answer), …]`; **TRAKE** `[(video_id, frame_id_1, …, frame_id_N), …]`, top-K ≤ 100 — [AIO25-3 pp.4–6].
- **Textual-KIS R-Score** = 𝕀(v = GT_v ∧ id ∈ [s,e]). Example: GT L01_V001 frames 500–510; answering 505 scores 1 and 600 scores 0 — [AIO25-3 pp.8–9].
- **VQA R-Score** = 𝕀(v = GT_v ∧ id ∈ [s,e] ∧ a = GT_a). All three must match; a wrong answer string or wrong video scores 0 — [AIO25-3 pp.10–11].
- **TRAKE R-Score** = (1/N) Σ_j 𝕀(id_j ∈ [s_j, e_j]) if v = GT_v, else 0. Partial credit is per moment. Each moment's tolerance is a GT frame ± epsilon. In the example epsilon = 5: GT frames 100/150/200/250, answer 101,156,203,251, so 3/4 match and R = 0.75 — [AIO25-3 pp.12–15].
- **Final score** = (1/5) Σ_{k∈{1,5,20,50,100}} max_{i≤k} R-Score(r_i). Example: R = 0.5 at rank 1 and 0.8 at rank 3 gives (0.5+0.8+0.8+0.8+0.8)/5 = 0.74. The slide adds that this "encourages you not only to find a correct answer but to place it at the first positions" — [AIO25-3 pp.16–20]. LLandMark restates the same Codabench protocol — [LLM p.5].
- 2025 qualifier query mix: Round 1 had 17 KIS / 3 QA / 3 TRAKE; Round 2 26/2/2; Round 3 29/4/2; **total 72 KIS / 9 QA / 7 TRAKE = 88 queries, max 88 points** — [CASC p.6 Table 2; LLM pp.5–6 Table 2]. The final round had four categories: TKIS, VKIS, TRAKE, QA — [MAD p.11].
- The AIO 2025 pipeline template is Query Enhancement → Query Search → Filtering → Reranking → (optional) Temporal Search → Submission. VQA uses "Interact & Answering" and TRAKE uses "N events" — [AIO25-3 pp.26, 35–37].
- The AIO 2025 temporal-search baseline takes top-K frames, decomposes Q into Q1 and Q2, gathers all frames of the same video, and scores `score = sim(Q1, F_i) + max(sim(Q2, F_{i+1 : i+T}))` with a fixed window T. The slide calls this heuristic and an "open question" — [AIO25-1 pp.56–65; AIO25-2 pp.30–33].
- Reranking options discussed: OD/OCR constraint filters (e.g., adding the OCR "NEW COFFEE" constraint); MLLM relevance scores in [0,1]; cross-encoder; region–phrase alignment. The deck judges model rerankers as costly and high-latency and recommends **SuperGlobal reranking**: kNN refinement with GeM pooling, final score 0.5·(cos(r_o_img, r_r_text) + cos(r_o_text, r_r_img)) — [AIO25-2 pp.56–78]. User-feedback formula: re-score using the cosine to the query plus the user's relevant and non-relevant frames — [AIO25-2 pp.80–81]. Filtering uses metadata (object location, quantity, colour, ASR, OCR) — [AIO25-2 p.85].

### Inferences
- **Answer granularity is frame-level**, both in the 2025 Codabench rules and in the BTC definition "return exactly the video segment or keyframe". A single frame index inside [s,e] scores full credit for KIS/QA, so interval (start, end) outputs are not needed for KIS/QA scoring. TRAKE needs N ordered frame indices, each within ±ε of its GT moment frame (ε = 5 frames in the example). **A moment solver should output one representative frame per event, not boundaries.** It should aim for the temporal centre of each event, because GT tolerance is narrow (±5 frames ≈ 0.2 s at 25 fps, if ε is in frames).
- Because the score is the mean of best-R@{1,5,20,50,100}, rank 1 is worth 5× a hit that appears only at rank 21–50. Diversity across the top-100 still earns 0.2–0.8 of the credit. Design implication: **fill all 100 slots**. Put the best candidate first, then diversify across videos. Do not spend all slots on neighbouring frames of one wrong video. Near-duplicate frames of the same correct video add nothing once one is inside [s,e]. For TRAKE, spend slots on alternative frame combinations within the best video(s), because partial credit (k/N) rewards getting more events right.
- VQA scoring is exact-match on the answer. So an LLM "answer synthesis" (as in LLandMark) must end with a human check of the normalized answer string. The frame must also be inside [s,e], which means retrieval matters even for QA.
- The 5-minute per-query limit, with scoring depending on time and rank, plus the BTC stress on "very fast coarse filtering", support a CPU-FAISS coarse stage (SigLIP 2 / BEiT-3) with heavier rerankers only on top-K. They also support fast browsing (timeline and neighbour frames).
- The guarantee that "at least one provided keyframe lies inside the GT segment" [S26-1 p.44] means keyframe-level retrieval over the provided keyframes can always hit KIS in principle. Our own 335,477 keyframes need to be mapped to original frame indices. The frame index is the scored unit.
- KISC (2026, new) and the BTC emphasis on exploration/exploitation feedback make a **stateful multi-turn session** (query refinement plus relevance feedback plus metadata/date/location filters) a first-class requirement. None of the local papers implements this well.
- AVS appears in the BTC 2026 slides [S26-1 p.12] but has no 2025 scoring slide locally. If AVS is scored like TRECVID/VBS (precision/recall over many submissions), the "fill 100 diversified slots" policy matters even more. This is unverified.

### Gaps
- No official 2026 document in the local files gives: the submission format, whether TRAKE continues in 2026, the R-Score/epsilon values, the per-query time limit, the time-penalty formula, the number of allowed submissions per query, or how KISC/AVS are scored. The question "how many submissions are allowed in the preliminary round" was asked in the AIO 2025 Q&A but the answer is not in the slide text — [AIO25-3 p.72].
- Whether epsilon is in frames or keyframe indices is not stated. The example uses frame numbers.
- Image-only slides that I did not render: [S26-1 pp.1, 5–6, 21–24, 42–47], [S26-2 pp.12, 26, 30–40 partly rendered: vitrivr UI, early/late fusion diagram, scene-graph grounding, GLIPv2, CLIP contrastive matrix, sketch→ControlNet query]. None of the rendered ones contained rules.

---

## Q2. For each local paper: architecture, models, fusion, reranking, temporal method, captions/OCR/ASR/OD, interactive features, quantitative evidence, and claimed novelty

### Takeaway
All five are **system/demo papers with no controlled ablations and no retrieval metrics (R@K, MRR)**. The only numbers are competition totals: AIC 2025 qualifier scores of 75.4 (MADTempo), 76.4/88 (Cascaded) and 77.40/88 (LLandMark), plus adjective final-round grades. Every component-level claim (reranking helps, fusion helps, DANTE helps, QUEST/landmark helps) rests on **one or two qualitative examples**. Useful concrete parameters that did appear: Cascaded top-100 → BLIP-2 ITM, 4 query variants, decay α = 0.01, beam 8; DANTE λ ∈ [0.001, 0.01]; MADTempo dedup cos > 0.965; LLandMark keyframe percentiles [0.15, 0.5, 0.85].

### Cited Findings

**MADTempo (2512.12929) — team AIO_Trình, AIC 2025**
- Claimed contributions: (1) an end-to-end pipeline with CLIP-LAION + Milvus + MongoDB and a GPT-5 query enhancer; (2) a **Google Image Search fallback for OOD queries**; (3) a **temporal search pipeline for multi-event retrieval (TRAKE)** — [MAD p.2].
- Preprocessing: TransNetV2 shots → representative keyframes → CLIP-LAION embeddings. **Two-stage dedup: pHash, then cosine > 0.965 on embeddings**. Milvus for kNN — [MAD pp.3–4].
- Metadata: **YOLOv8** OD, **Vintern-1B-v3.5** OCR, **Qwen2.5-VL captions**, **PhoWhisper ASR**, all in MongoDB with full-text search — [MAD p.4].
- Keyframe retrieval: v_q = CLIP_text(Q) or CLIP_image(user-selected Google image). MongoDB filters over ASR/OCR/OD are then applied: FinalSet = {k ∈ K | Filter(M(k), Q_m)}, "re-ranked by combined similarity and metadata relevance". The combination formula is not given — [MAD p.5]. Metadata acts as a **hard filter on visual candidates** here.
- Captions are used only inside TRAKE ContextScore (captions of boundary keyframes + ASR between them, fed to the LLM) — [MAD p.8].
- UI: query panel with metadata filters plus keyword/semantic search over objects/OCR/ASR/captions; temporal constraints; Include/Exclude IDs; Query Type; "Query Packs"; selection box/submit; keyframe viewer with temporal filmstrip, relevant-image search and frame exclusion; TRAKE UI with a context field and an ordered event list, each event with optional OCR/caption filters — [MAD pp.9–11].
- Results: preliminary official score **75.4** (max not stated in this paper). Final-round grades: **TKIS "Good", VKIS "Very good", TRAKE "Excellent", QA "Good"**, overall "Very good" — [MAD pp.10–11]. The authors attribute TRAKE to their algorithm and VKIS to Google Image Search without any ablation. "Early user evaluations show … significantly accelerate search", with no numbers — [MAD p.12].

**DANTE paper (2512.13169) — team AIO_Owlgorithms, AIC 2025**
- Claimed novel: **QUEST** (LLM rewriting + external image search for out-of-knowledge queries) and **DANTE** (DP for TRAKE) — [DANTE pp.1–2].
- Keyframes: TransNetV2 scenes, **4 frames per scene at a + ⌊i(b−a)/3⌋, i = 0..3** (start, two middles, end) — [DANTE pp.2–3].
- Single visual model: **BEiT-3** embeddings, L2-normalized, in Milvus; GPU dynamic batching — [DANTE p.3].
- OCR: **Gemini** per keyframe → Elasticsearch with the **Vietnamese analysis plugin (duydo) + coccoc-tokenizer** — [DANTE pp.3–4, refs 2, 4]. No ASR, no OD, no captions.
- Storage: Milvus {keyframe_id, vector}; ES {keyframe_id, ocr_text}; MongoDB {keyframe_id, image_path} plus a per-video {video_id, start index s_v, end index e_v}. **Keyframe IDs are globally contiguous per video**, which DANTE exploits. "Query-time hydration" resolves IDs to metadata — [DANTE pp.3–4].
- Modes: Semantic, OCR and DANTE are separate modes chosen by the user, with no automatic fusion. Also image-to-image "Find similar", image upload, a GPT-4o-mini "AI enhancer", Top-K, score threshold, penalty weight, video group/ID filters and a YouTube play button — [DANTE pp.4, 7–8].
- QUEST branch 1: an LLM (GPT-4 / Gemini; GPT-4o-mini in the UI) rewrites q0 into a visual description q_r. Branch 2: a Google exemplar image I* → BEiT-3 image-to-image search — [DANTE p.5].
- Evidence: the Semantic and OCR examples are Top-1 (three-tier cake; OCR "PHÚ XUÂN – GIA ĐỊNH – những dấu ấn lịch sử") — [DANTE p.8]. QUEST examples: "Labubu" fails directly and succeeds at Top-1 after rewriting; "Babythree" is found via a web image at Top-1 — [DANTE pp.8–9]. DANTE examples: a TRAKE-style assembly-line query from the preliminary round; a Video KIS clip described as 3 ordered cues (green rice field → golden rice bending → white rice grains close-up), Top-1; a Textual KIS clip with 2 cues (French cave carvings → researchers in white suits with headlamps), top position — [DANTE pp.9–10]. Final round: **TRAKE "Outstanding", Textual KIS "Excellent", Video KIS and QA "Very Good", overall "Excellent"** — [DANTE p.10]. No numeric metrics.
- New detail: DANTE was also used for **KIS**, by typing multiple ordered descriptions of a shown clip. This makes ordered multi-cue search a KIS tool, not only a TRAKE tool — [DANTE pp.9–10].

**Cascaded Embedding-Reranking (2512.12935) — AAAI 2026 workshop**
- Claimed innovations: (1) cascaded dual-embedding (BEiT-3 + SigLIP) → BLIP-2 rerank; (2) **temporal-aware scoring with exponential decay via beam search**; (3) GPT-4o agent query decomposition into visual/OCR/ASR sub-queries with adaptive weights — [CASC p.1].
- Offline: audio extraction; TransNetV2 with **3 keyframes per shot**; BEiT-3 + SigLIP (SigLIP v1, Zhai 2023 is cited; the variant is not given), both normalized and stored as **Qdrant named vectors**; OCR with **Gemini 2.0 Flash** via a JSON prompt (MLLM chosen over Tesseract/PaddleOCR); ASR with **Whisper Large-v3**, where timestamped segments are aligned to the nearest keyframes — [CASC p.2]. No OD, no captions in the index. Unified captions are listed as future work — [CASC p.7].
- Query expansion: GPT-4o generates **N = 4 variants**. **Variant 1 must be the direct English translation.** Variants may change angle, setting or style but "must not introduce new objects or actions". All variants are embedded with SigLIP and BEiT-3 and searched in parallel — [CASC p.5].
- Routing prompt heuristics: KIS/visual = what is *seen*; OCR = what is *read* (banners, numbers, jersey names); ASR = what is *heard* (speech keywords, lyrics, but not generic verbs like sing/speak). The agent outputs weights plus a short reason. Example: "Cristiano Ronaldo scoring a goal" gives visual high, OCR moderate, ASR low — [CASC p.5].
- Hyperparameters: QE 4 variants; first-stage top-100; BLIP-2 rerank; **decay coefficient 0.01, beam width 8** — [CASC p.6].
- Results: qualifier **76.4/88 (86.8%)**; R1 19.8/23, R2 26.6/30, R3 30/35 — [CASC p.6 Table 1]. Note: the text says "reaching the maximum score in Round 3", but Table 1 shows 30/35. This is an internal inconsistency. The final-round result is not reported.
- Qualitative only: without rerank the GT (blue bird) is missing from the top results, and with BLIP-2 rerank it reaches the top. Agent fusion with w_ocr ≈ 0.7 and w_vis ≈ 0.4 finds a "Program: Financial Support…" sign scene. Temporal example: a 3-event car-assembly sequence spanning 5.1 s scored 0.9234 versus a 34.1 s alternative with a lower score — [CASC pp.6–7]. Figure numbering in the text and captions is inconsistent (Fig 4/5/6) — [CASC pp.6–7].

**LLandMark (2603.02888) — AAAI 2026 workshop**
- Claimed contributions: a multi-agent architecture (Parsing & Planning Agent → Landmark Knowledge Agent → Orchestrator → Reranking & Answer Agent); **OCR with Gemini refinement** (PaddleOCR + Gemini 2.5 Flash via LlamaIndex); **LLM-assisted landmark image-to-image retrieval** — [LLM p.1].
- Keyframes: TransNetV2; 3 per shot at **percentiles [0.15, 0.5, 0.85]** (all frames if the shot is short). IDs are `group/video/frame` and live in MongoDB — [LLM p.2].
- Visual: **CLIP ConvNeXt-XXLarge (laion2B-s34B-b82K-augreg)** in Milvus. ASR: **WhisperX** → Elasticsearch BM25 with highlights. OD: **YOLOv9-e (COCO)** in one lazily loaded JSON, with AND/OR object filters ranked by matched object counts — [LLM p.2].
- OCR detail: PaddleOCR mangles Vietnamese diacritics, so the output is first normalized to **accent-free** text. Gemini 2.5 Flash then **restores diacritics**, fixes spelling and removes noise in structured batch prompts. The result is indexed in ES BM25 with group/video/frame/confidence — [LLM p.3].
- SearchPlan: a weighted set of steps over semantic/ASR/OCR/object. The semantic query is translated to descriptive English, while **ASR/OCR keep Vietnamese landmark names verbatim**. The Landmark agent replaces names with curated visual descriptions (e.g., St. Joseph's Cathedral → "Twin square bell towers, dark gray stone, Gothic architecture, neo-Gothic facade") — [LLM p.4].
- Fusion: weighted average Σ w_m s_m / Σ w_m. Scale normalization is not described. Top frames are grouped by video, and evidence (images, ASR, OCR, objects) goes to a multimodal LLM that writes a grounded answer — [LLM p.4].
- Landmark image-to-image: the LLM detects the landmark and writes a web-image query (e.g., "The Imperial City of Hue from above"). Google Custom Search returns images, which are encoded with CLIP image embeddings and searched in Milvus. Results are merged across reference images, **duplicates keep the max similarity**, and the set is reranked by confidence. UI knobs: Per-Reference Top-K, Max Landmarks, Images/Landmark — [LLM pp.4–5, 7].
- Results: qualifier **77.40/88** (R1 20.00/23, R2 28.20/30, R3 29.20/35); among >680 registered teams it was in the top 56 qualified — [LLM p.5 Table 1]. Qualitative: "Bach Dang Wharf at night" fails with embeddings and succeeds with LLandMark; "Ben Thanh Market" is read by CLIP as a generic market and succeeds via landmark image-to-image — [LLM p.6]. There is no ablation of the OCR refinement (no CER/WER before and after).
- UI: auto VI→EN translation for CLIP; mode switching; include/exclude; an **index-to-time calculator** mapping keyframe index to timestamp; panels showing the refined query, the per-modality weights of the search plan and the ASR transcript — [LLM pp.6–7].

**NII-UIT at VBS 2025**
- Keyframes: in a Vibro-inspired approach, **BEiT-3 features every 10th frame, keeping frames with significant differences**. The threshold is not given. Frames are stored as **WebP** — [NII p.3 (printed 320)].
- Retrieval: multiple VLMs (BEiT-3; OpenCLIP H-14 and InternVL-G are discussed), per-model normalization, then fusion — [NII pp.3–4]. Multi-modal search: scores from text, visual, query-expansion and SD-generated-image lists are **normalized, then mean-pooled**, then **Co-DETR (COCO) object filtering excludes shots lacking the objects**, then rerank — [NII p.5 (printed 322)].
- Query expansion: GPT-4o gives **5 paraphrases**. The user can pick one or fuse all in parallel — [NII p.4]. Visual query generation: Stable Diffusion images as queries, singly or fused — [NII p.4].
- Temporal: for KIS-T with unclear order, it explores shots **both before and after** the initial result, scores them against the new description and aggregates across stages. Stages are unlimited and iterative (vitrivr-inspired) — [NII p.5].
- Q/A: a question is converted into a description (e.g., the license-plate question becomes "A red Volkswagen Golf GTI with a license plate") and then searched temporally — [NII p.5].
- UI: two result modes (group nearby shots, or best frame per shot); Advanced Mode with model weights, paraphrasing and generation, hidden for novices — [NII pp.5–6].
- No quantitative results at all. Claims such as "surpasses traditional methods" are unsupported — [NII pp.1, 6].

### Inferences
- **Evidence strength ranking**: none of these papers gives component-level numbers. Qualifier totals are tightly clustered (75.4 / 76.4 / 77.4 on 88), so the 2025 preliminary did not separate these architectures: CLIP-LAION-only+metadata, BEiT-3+SigLIP+BLIP-2, and ConvNeXt-XXL CLIP+agents score about the same. **Our benchmark must decide component choices.** The papers can only suggest candidates.
- Recurring components that deserve benchmarking in a visual-first design: (a) a second visual encoder plus rank/score fusion; (b) a cross-encoder rerank on top-100; (c) English translation or expansion for the visual encoder while keeping Vietnamese literals for OCR/ASR; (d) web-image exemplar → image-to-image search for OOD entities and landmarks, which appears in three of five papers as a user-triggered fallback; (e) ordered multi-cue search, used even for KIS.
- Captions: only MADTempo uses captions (Qwen2.5-VL), and only as LLM context inside TRAKE reranking, not as a primary index. This is consistent with the team's "captions subordinate" plan. No paper shows captions helping retrieval quantitatively.
- OD: every use is a **hard filter** (MADTempo MongoDB filter, LLandMark AND/OR, NII Co-DETR exclusion). The fail-open caution in the earlier review still applies. The BTC-provided OD is Open Images V4 with 600 classes [S26-1 p.45], richer than COCO's 80.
- OCR engines used by 2025 teams have moved to MLLMs (Gemini ×2, Vintern-1B, Gemini-refined PaddleOCR). Nobody reports OCR accuracy.

### Gaps
- No paper reports R@K/MRR, latency, index size or hardware. There is no ablation of BEiT-3 vs SigLIP, of SRRF vs RRF, or of rerank on/off.
- MADTempo's preliminary max score is not stated. If it is also out of 88, the result is comparable to the others.
- The final-round grade scale ("Good / Very good / Excellent / Outstanding") is not defined in any local file.

---

## Q3. How do DANTE (2512.13169) and MADTempo (2512.12929) handle multi-event / ordered queries (algorithm, scoring, gap penalties, complexity)? Also LLandMark and NII for contrast.

### Takeaway
**DANTE** is an exact DP over all keyframes of every video. It uses a linear gap penalty λ·(t−τ) in keyframe-index units, a running-max recurrence giving O(N·T) total, and it returns top-k videos with one backtracked keyframe per event. **MADTempo** is a cascade: boundary pairs (E1, En) under a hard max-gap (n−1)·τ, then top-M segments, then an LLM ContextScore over captions + ASR, then beam search (width b) over intermediate events inside those segments, with FinalScore = α·Event + (1−α)·Context. MADTempo has only a hard gap constraint (no soft penalty). DANTE has only a soft penalty (no hard cap). **LLandMark** does not enforce order at all: it intersects the video sets and takes the min score. **NII** searches bidirectionally around a pivot.

### Cited Findings

**DANTE (exact algorithm)** — [DANTE pp.5–7]
- Input: N event query embeddings u_1..u_N (BEiT-3 text); per-video contiguous keyframe index ranges [s_v, e_v] covering 1..T; embeddings E[1..T]; penalty λ.
- Step 1: S[i,t] = cos(u_i, E[t]) for all i ≤ N and t ≤ T, obtained "via Milvus queries to reduce load" (Eq. 1). This is a dense N×T similarity matrix, i.e., event-vs-every-keyframe.
- Step 2, per video: DP[1,t] = S[1,t]; DP[i,t] = S[i,t] + max_{τ∈[s_v, t−1]} (DP[i−1,τ] − λ(t−τ)) (Eq. 2). Order is strict (τ < t), and there is no minimum or maximum gap.
- Running-max form: running_max = max(running_max, DP[i−1,t−1] + λ(t−1)); DP[i,t] = S[i,t] + running_max − λt (Eqs. 3–4). Complexity is **O(N·(e_v − s_v + 1)) per video, O(N·T) overall**.
- Step 3: DANTE[v] = max_t DP[N,t] (Eq. 5). The output is the top-k videos by DANTE[v], with N keyframes found by backtracking stored argmaxes.
- Tuning in the AIC 2025 final: **λ ∈ [0.001, 0.01]. λ = 0.001 was best when GT event keyframes were 3–15 indices apart, and λ = 0.01 was best for tight 1–3 index offsets** — [DANTE p.7]. The penalty unit is keyframe index (4 keyframes/scene), not seconds.
- Scoring notes: the score is the sum of raw cosines minus a linear gap cost. The gap cost does not depend on the time between scenes of different lengths.
- Claimed novel. Evidence is qualitative (3 examples) plus the "Outstanding" TRAKE grade.

**MADTempo TRAKE (exact algorithm)** — [MAD pp.6–8]
- Decomposition: Q → {C, E_1..E_n} by GPT-5 or manually — [MAD p.7].
- Boundary stage: SimScore_i(k) = sim(z_Ei, z_k) with CLIP. For each keyframe k1 matching E_1 (the threshold/top-K for "matching" is not given), the feasible ends are K_n(k1) = {k_n : same video, 0 < t_kn − t_k1 ≤ (n−1)·τ}, where **τ = maximum allowed duration between consecutive events** (value not given). Choose k̂_n = argmax SimScore_n over K_n. BoundaryScore = SimScore_1(k1) + SimScore_n(k̂_n). Keep the **top-M** segments (M not given). Each feasible start keeps only its single best end.
- Context stage: Meta_i = (caption of k1, caption of k̂_n, ASR transcript between them, video id). ContextScore = f_LLM(Meta_i, C, {E_j}) / 100, where the LLM outputs 0–100 — [MAD p.8]. This is M LLM calls per query (cost/latency not reported).
- Event stage: a filtered corpus K_F of keyframes in the top-M segments. For intermediate E_2..E_{n−1}, a **beam search** per segment keeps the top-b partial paths by cumulative similarity, enforcing same video, strictly increasing timestamps and gap ≤ τ between consecutive events. EventScore = max over paths of Σ_j SimScore_j(k_j) — [MAD p.8].
- Final: FinalScore = α·EventScore + (1−α)·ContextScore, with α ∈ [0,1] (value not given) — [MAD p.8]. EventScore is a sum over n events while ContextScore lies in [0,1], so the scales differ and α must absorb this. The paper does not discuss it.
- Complexity (my derivation, not stated in the paper): the boundary stage is O(|cands(E1)| × frames within (n−1)τ). The event stage is O(M·(n−2)·b·W), where W is the number of frames per window. There is also an LLM term of O(M).
- UI: the context and each event can carry OCR/caption filters — [MAD p.11]. Claimed novel ("robust temporal search pipeline"). Evidence: only the "Excellent" TRAKE grade.

**Contrast**
- LLandMark: per step, an independent vector search gives V_i with S_i(v) = the max keyframe similarity. Then V_cand = ∩V_i and Score(v) = min_i S_i(v) — [LLM p.3]. **No ordering and no gap constraint.** It ranks videos only and does not localize frames per event.
- NII-UIT: explores shots before and after the initial result for the next description and aggregates scores. It is designed for KIS-T where order is ambiguous — [NII p.5].
- AIO 2025 baseline: sim(Q1, F_i) + max sim(Q2, F_{i+1..i+T}) with a fixed T — [AIO25-1 p.64].

### Inferences
- For TRAKE scoring (per-event frame within ±ε, partial credit k/N, wrong video = 0), **DANTE's output type matches the answer format exactly**: one video plus N frame indices. Its per-video max also gives a natural ranked list of videos for the 100 slots. Its dense S[i,t] on our corpus costs 335,477 × N dot-products per query. For SigLIP 2 so400m (1152-d) that is about 0.39 GFLOP per event, which is cheap on CPU with a single matmul. **Exact DP over the whole corpus is feasible on CPU without candidate pruning.** This is an inference, not measured.
- DANTE's linear penalty in keyframe-index units is sensitive to keyframe density. Our keyframe spacing may differ from 4 per scene, so λ must be re-tuned. Converting to seconds (λ·Δt_sec) would make it density-invariant. The DP stays O(NT) with a running max as long as the penalty is linear in position. It also stays O(NT) with an exponential decay as in CASC only if restructured. Linear is the only form with the trivial running-max trick.
- MADTempo's hard cap (n−1)·τ plus top-M pruning can lose the GT if E_1 or E_n is poorly matched. DANTE has no pruning but also no cap, so a small λ can chain events across a whole 5-hour video. The 2026 egocentric/lifelog 5-hour videos [S26-1 p.16] make a max-gap cap more important than it was for 2025 news clips.
- Neither handles **unordered or partially ordered** events, event repetition, or "A before B" when the query order differs from video order. NII's bidirectional search covers the ambiguous-order case.
- Scores in both are raw cosine sums. Mixing encoders (SigLIP 2 vs BEiT-3) inside a DP needs per-encoder calibration, for example z-scoring per event column.

### Gaps
- No paper reports TRAKE accuracy numerically, compares DP vs beam, or states MADTempo's τ, M, b and α.
- DANTE does not say how it obtains S over all T keyframes from Milvus in practice (whether full scan or top-K per event with −∞ elsewhere). The text says "via Milvus queries to reduce load", which suggests possible top-K sparsification. This is ambiguous.

---

## Q4. How does 2512.12935 fuse BEiT-3 + SigLIP and rerank with BLIP-2 ITM (top-K, weights, cost), and how does its temporal decay work?

### Takeaway
Two visual scores are fused by **SRRF (Score-Reflected Reciprocal Rank Fusion)**, described only as "preserves the original similarity scores rather than relying solely on rank positions as in standard RRF". The formula, the k constant and the per-model weights are **not given**. The **top-100** fused frames are reranked with the **BLIP-2 ITM head**. Visual, OCR and ASR lists are then min-max normalized and combined with **GPT-4o-predicted weights**. Rerank cost is described only as linear in pairs. There is no latency, hardware or ablation number.

### Cited Findings
- Visual branch: each keyframe has SigLIP and BEiT-3 vectors in Qdrant, both cosine-matched, and the two lists are combined with SRRF. **Top-100 SRRF candidates → BLIP-2 ITM rerank** ("cross-modal attention … fine-grained alignment") — [CASC p.3]. With QE, all 4 variants are embedded with both encoders, "then reranked and searched in parallel" — [CASC p.5]. How the 4 variants' lists are merged is not stated.
- OCR/ASR branch: Elasticsearch documents scored by "exact phrase, full-term match, partial match, and fuzzy match" — [CASC p.3].
- Cross-modality fusion: s_norm_m(f) = (s_m(f) − min s_m)/(max s_m − min s_m + ε) (Eq. 1); S(f) = Σ_{m∈{vis,ocr,asr}} w_m · s_norm_m(f) (Eq. 2), with w_m from the GPT-4o agent — [CASC p.3]. The example weights w_ocr ≈ 0.7 and w_vis ≈ 0.4 do not sum to 1, so the weights are not normalized — [CASC p.6].
- Cost rationale: cross-encoders are "impractical at collection scale as the computational cost grows linearly with the number of pairs", so BLIP-2 is used only on a small set — [CASC p.4]. BLIP-2's "O(n) complexity … limits scale" — [CASC p.2]. The BLIP-2 variant, GPU and latency are not given.
- Temporal: K events with M candidates each, beam search keeping the top-B partial sequences, giving **O(B·K·M)** — [CASC p.4]. Decay λ_i = e^{−α·Δt_i}, Δt_i = t_i − t_{i−1} (Eq. 3). Sequence score SS_j = Σ_i s_i·e^{−α(t_i − t_{i−1})} (Eq. 4), an **additive** aggregation chosen over a multiplicative Π s_i λ_i to tolerate one weak link. Then **BLIP-2 validation**: S_i(final) = s_i·λ_i·b_i and SS(final) = Σ s_i λ_i b_i (Eqs. 6–7), a **multiplicative gate** — [CASC pp.4–5]. The text states "Δt < 2 s gives weight near 1, Δt > 10 s is exponentially penalized". With **α = 0.01** (the setting) [CASC p.6], e^{−0.01·10} = 0.905 and e^{−0.01·34} = 0.71 if Δt is in seconds. The penalty is much weaker than the prose implies unless Δt is in frames (e^{−0.01·250} = 0.08 for 10 s at 25 fps). The unit of Δt is not specified. (Calculation mine.)
- Beam width **B = 8** — [CASC p.6]. The paper contrasts itself with ABTS: a global sequence-level decay versus ABTS's local variance-based stability — [CASC p.4].
- The only evidence is qualitative. The "Without Rerank" case misses the GT and "With Rerank" puts it at the top (blue-bird example). No R@K, no latency — [CASC pp.6–7].

### Inferences
- This architecture is closest to our assets (SigLIP 2 + BEiT-3, BLIP-2 already used for captions). The AIO 2026 deck presents it as the reference system for 2026 [AIO26 pp.6–10], so many competing teams will likely build this baseline. Differentiation must come from elsewhere: temporal solver, interaction, KISC dialogue, diversification of the 100 slots.
- Since SRRF is unspecified, our benchmark should compare: plain RRF (k = 60); score-weighted RRF variants (e.g., Σ w_m·s_m/(k + rank_m)); z-score sum. Per-query min-max over the top-K list (CASC Eq. 1) is known to be outlier-sensitive (already noted in the earlier review).
- BLIP-2 ITM on 100 candidates × 4 QE variants means up to 400 cross-encoder passes per query. On CPU-only FAISS infrastructure this is likely the latency bottleneck. It should be measured, or the rerank should run on GPU or on fewer candidates. (Inference; no cost numbers in the source.)
- The multiplicative b_i gate means one BLIP-2 miss (b_i ≈ 0) removes that event's contribution. For TRAKE partial credit it may be better to keep the additive form after reranking.

### Gaps
- Missing from the paper: the SRRF formula, BEiT-3/SigLIP relative weights, BLIP-2 checkpoint (OPT/FlanT5, pretrain vs COCO-ft), ITM score normalization, the M (candidates per event) value, how the 4 QE lists are merged, and any timing. None can be recovered from local files.
