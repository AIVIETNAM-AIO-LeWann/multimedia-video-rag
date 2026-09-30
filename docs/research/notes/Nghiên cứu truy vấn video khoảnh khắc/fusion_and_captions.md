# Score fusion for multi-model / multi-query text-to-image retrieval, and the role of generated captions

Project context for the reader: 335k keyframes from Vietnamese news and cooking videos; BLIP-2 OPT-2.7B-COCO English captions; Vietnamese queries; exact CPU FAISS. Most evidence below comes from other domains (English text IR, MS-COCO/Flickr30k, TRECVid). Each section says where that matters.

## 1. Fusion methods: RRF, CombSUM/CombMNZ, normalization, learned fusion, and CLIP-family ensembles

### Takeaway
The strongest controlled study found (Bruch et al., TOIS 2023) shows that a convex combination (CC) of normalized scores beats RRF(k=60) with statistical significance on every dataset tested. CC needs very few labeled queries to tune its single weight, and the choice of normalization matters little once the weight is tuned. RRF is sensitive to k, and a k tuned in one domain does not transfer to another. This evidence comes from lexical+dense *text* hybrid retrieval, not from fusing two image embedding models. I found no controlled study of SigLIP 2 + BEiT-3 fusion, so the gain from that combination has to be measured on the project's own data.

### Cited Findings
- Bruch, Gai & Ingber (TOIS 2023), abstract: "Contrary to existing studies, we find RRF to be sensitive to its parameters; that the learning of a CC fusion is generally agnostic to the choice of score normalization; that CC outperforms RRF in in-domain and out-of-domain settings; and finally, that CC is sample efficient, requiring only a small set of training examples to tune its only parameter" — [arXiv 2210.11934](https://arxiv.org/abs/2210.11934)
- Setup in the same paper: RRF η (=k) was set to 60 following Chen et al.; TM2C2 (convex combination over theoretical-min-max-normalized scores) used α=0.8 tuned on in-domain validation data. "TM2C2 significantly outperforms RRF on all datasets in terms of NDCG, and does generally better in terms of Recall." RRF still beat pure lexical and pure semantic retrieval on most datasets and was "particularly effective on out-of-domain datasets" — [arXiv 2210.11934 PDF](https://arxiv.org/pdf/2210.11934)
- NDCG@1000 numbers (Table 3), in the order TM2C2 / RRF(60,60) / RRF(5,5) / RRF(10,4): MS MARCO 0.454 / 0.425 / 0.435 / 0.451; NQ 0.542 / 0.514 / 0.521 / 0.528; HotpotQA (zero-shot) 0.699 / 0.675 / 0.693 / 0.621; FEVER 0.744 / 0.721 / 0.727 / 0.649; FiQA 0.496 / 0.464 / 0.470 / 0.482. An RRF tuned in-domain "does not generalize well to out-of-domain datasets": RRF(10,4) is best in-domain but drops sharply on HotpotQA and FEVER — [arXiv 2210.11934 PDF](https://arxiv.org/pdf/2210.11934)
- Normalizations compared: min-max (per-query set), theoretical min-max (minimum = −1 for cosine similarity, 0 for BM25), z-score (per-query μ, σ), and unnormalized variants. The paper's finding is that learning a CC is "generally agnostic" to which of these is used — [arXiv 2210.11934 PDF](https://arxiv.org/pdf/2210.11934)
- Why RRF loses information: "because RRF is a function of ranks, it disregards the distribution of scores and, as such, discards useful information… the distance between raw scores plays no role." A smoothed RRF variant (SRRF) raised NDCG over RRF, which the authors attribute to Lipschitz continuity — [arXiv 2210.11934 PDF](https://arxiv.org/pdf/2210.11934)
- Sample efficiency: "with less than 5% of the training data, which is often a small set of queries, TM2C2's α converges, regardless of the magnitude of domain shift." Parametric RRF is "less sample-efficient and converges to a relatively less effective retrieval system" — [arXiv 2210.11934 PDF](https://arxiv.org/pdf/2210.11934)
- Per-query score calibration for contrastive VLMs: Nearest Neighbor Normalization (NNN, EMNLP 2024) subtracts a per-image "hubness" bias estimated from a reference set of queries. It needs no training and adds constant overhead once the biases are precomputed. Text→image R@1 before → after NNN: CLIP COCO 30.43 → 37.53, Flickr 58.82 → 64.60; SigLIP COCO 47.15 → 50.24, Flickr 74.62 → 76.54; BEiT-3 COCO 47.62 → 50.64, Flickr 75.52 → 76.66; BLIP (COCO-finetuned) COCO 62.68 → 64.44. In-distribution reference queries work best (α≈0.75); out-of-distribution queries still help, with a smaller α. The BEiT-3 checkpoint used in NNN is not identified; its 47.6 COCO R@1 is far below the finetuned 67.2 — [arXiv 2410.24114](https://arxiv.org/html/2410.24114v1); [ACL Anthology](https://aclanthology.org/2024.emnlp-main.1257/)
- Ensembling CLIP models: one 2025 conference paper reports that combining OpenAI ViT-L-14-336 and Apple ViT-L-14 with per-model confidence normalization in FAISS "can match or surpass" the larger ViT-H-14 models. The summary gives no COCO/Flickr numbers, and the venue is minor — [ACM ICIIT 2025](https://dl.acm.org/doi/10.1145/3731763.3731800)
- Multimedia practice: VBS systems depend heavily on CLIP-like joint embeddings (Lokoč et al., "lessons from the 11th VBS", Multimedia Systems 2023). Several VBS 2025 systems use late-fusion re-ranking of dense and sparse signals. These are system descriptions, not controlled fusion ablations — [search summary incl. Fusionista VBS 2025](https://link.springer.com/chapter/10.1007/978-981-96-2074-6_33); [Can Agents Win the VBS?](https://arxiv.org/html/2609.07311)

### Inferences
- For fusing SigLIP 2 with BEiT-3, and for fusing query variants, a reasonable default is a per-query z-score or min-max normalized weighted sum (CombSUM with weights) over the union of each model's top-K. Tune the weight on a small labeled set of the project's own KIS/QA queries; Bruch finds a handful of queries is enough. Keep RRF (k≈60) as the zero-label baseline. Whether this ranking of methods carries over from text hybrid retrieval to VLM+VLM fusion is an assumption that needs testing.
- The two models' cosine distributions differ in scale: SigLIP's sigmoid-trained logits versus BEiT-3's ITC cosines. Raw CombSUM without normalization is therefore risky. Theoretical min-max (minimum = −1) keeps score gaps and does not depend on the top-K set, so it is attractive for exact FAISS.
- CombMNZ multiplies by the number of lists that retrieve an item. It favors items both models agree on, which is a form of consensus bias. I found no image-retrieval study measuring CombMNZ against CombSUM.
- NNN is a cheap per-model calibration step to apply *before* fusion. It is relevant here because news footage has "hub" frames such as anchor shots and studio backgrounds.

### Gaps
- I found no peer-reviewed study that reports COCO/Flickr30k numbers for late fusion of two strong dual encoders (e.g. SigLIP + EVA-CLIP, or SigLIP + BEiT-3). Complementarity between these models is unproven by public benchmarks.
- I found no controlled rank-vs-score fusion comparison inside multimedia (VBS/AIC) systems.
- I could not verify the original RRF paper (Cormack et al., 2009, which predates the requested window). The k=60 default above is cited only as Bruch et al. report it.

## 2. Query-side ensembles: paraphrases, multilingual queries, LLM rewriting

### Takeaway
Measured gains exist, but they are modest on average and larger on "hard" queries. The best evidence for video search is GenSearch (TRECVid AVS/KIS). There, fusing the original query with LLM paraphrases and caption-derived paraphrases gives +4.9% relative mean xinfAP, halves median rank on textual KIS, and helps most on out-of-vocabulary, complex and negation queries. Weak variants such as text-to-image-generated queries had to be down-weighted. For Vietnamese→English translation into an English-leaning model, one data point from Vietnamese legal QA shows that translating queries helped a multilingual dense retriever.

### Cited Findings
- GenSearch ("Multimodal LLM-based Query Paraphrasing for Video Search", 2024) produces 10 GPT-4 rewrites (T2T), 15 Stable Diffusion images (T2I) and 10 BLIP-2 captions per image (I2T). A QA-consistency check filters the paraphrases, and the four rank lists are fused linearly: S = S_user + S_T2T + 0.5·S_T2I + S_I2T — [arXiv 2407.12341](https://arxiv.org/html/2407.12341v2)
- GenSearch results: mean xinfAP 0.299 vs 0.285 for the improved-ITV baseline (+4.9%) across TRECVid AVS 2016–2023, beating the TRECVid top-1 on 7 of 8 query sets. On textual KIS, median rank drops from 12.08 to 6.33. OOV queries improve by 14.21%. On negation queries, the I2T paraphrase scores 0.149 vs 0.082 for the user query. Failure mode: T2I alone scores 0.220 and was down-weighted; T2I verification degraded 44 of 210 AVS queries; generated images miss "motion dynamics" — [arXiv 2407.12341](https://arxiv.org/html/2407.12341v2)
- GenQREnsemble / GenQRFusion (text IR, 2024): an ensemble of N diverse LLM query reformulations improves retrieval "by up to 18%". This is ad-hoc text retrieval, not the visual domain — [arXiv 2404.03746](https://arxiv.org/pdf/2404.03746); [arXiv 2405.17658](https://arxiv.org/pdf/2405.17658)
- DREAM (video-text retrieval with LLM augmentation): LLaMA-based augmentation reached the best recall in that study (60.8 text→video, 60.6 video→text). The dataset and baseline need to be checked in the paper — [arXiv 2404.05083](https://arxiv.org/pdf/2404.05083)
- Cross-lingual Vietnamese signal (different domain, legal QA with BGE-M3): English→Vietnamese translated queries reach R@5 0.394 vs 0.358 for direct English queries under dense retrieval. In the same work, BGE-M3 sparse was poor for English→Vietnamese, and hybrid RRF underperformed dense-only on every metric. This comes from a 2026 preprint seen only through a search snippet — [arXiv 2609.27376](https://arxiv.org/pdf/2609.27376)

### Inferences
- A Vietnamese original fused with an English translation (plus 2–5 English paraphrases) against SigLIP 2 is likely to help. SigLIP 2 is multilingual, but its English COCO/Flickr numbers are much higher than its XM3600 average (see §3). BEiT-3 is English-only (SentencePiece, English corpora), so it *requires* the English translation.
- Per-variant weights matter. GenSearch had to down-weight a noisy branch, and a Bruch-style tuned convex weight is the analogue. Averaging the embeddings of query variants (vector mean) is a cheaper alternative to rank fusion, but I found no direct image-retrieval comparison of the two.

### Gaps
- I found no study that measures a multilingual-query ensemble (native + English translation) on XM3600 or a Vietnamese keyframe benchmark with SigLIP/SigLIP 2.
- I found no measured result for CLIP-style prompt templates ("a photo of …") on *retrieval* (as opposed to classification) that meets the 2019–2026 window and quality bar.

## 3. SigLIP 2 capabilities, BEiT-3 strengths, and complementarity

### Takeaway
SigLIP 2 is a strong zero-shot English retriever (COCO T→I R@1 about 52–56, Flickr about 80–86). Its multilingual retrieval is much better than SigLIP's, reaching XM3600 T→I R@1 of 40–48, and nearly matches mSigLIP (50.0). Vietnamese is among the XM3600 languages plotted per language, but the paper does not state a Vietnamese number. BEiT-3's headline COCO T→I 67.2 comes from the *1.9B giant model finetuned on COCO*. The public base and large COCO-retrieval checkpoints reach 61.4 and 63.4, which beat SigLIP 2 on COCO-style English captions but are English-only and COCO-domain-tuned.

### Cited Findings
- SigLIP 2, Table 1, zero-shot recall@1. Columns: COCO T→I / COCO I→T / Flickr T→I / Flickr I→T / XM3600 T→I / XM3600 I→T.
  - B/16@256: SigLIP 47.4 / 65.1 / 78.3 / 91.1 / 22.5 / 29.9; SigLIP 2 53.2 / 69.7 / 81.7 / 94.4 / 40.7 / 51.0.
  - L/16@256: SigLIP 51.2 / 69.6 / 81.3 / 92.0 / 30.9 / 40.1; SigLIP 2 54.7 / 71.5 / 84.1 / 94.5 / 46.5 / 56.5.
  - So400m/14@384: SigLIP 52.0 / 70.2 / 80.5 / 93.5 / 17.8 / 26.6; SigLIP 2 55.8 / 71.7 / 85.7 / 94.9 / 48.4 / 57.5.
  - So/16@256: mSigLIP 49.4 / 68.6 / 80.0 / 92.1 / **50.0 / 62.8**; SigLIP 2 55.4 / 71.5 / 84.4 / 94.2 / 48.1 / 57.5.
  - g/16@256: SigLIP 2 55.7 / 72.5 / 85.3 / 95.3 / 48.2 / 58.2.
  - Baselines at B/16@224: EVA-CLIP 42.2 / 58.7 / 71.2 / 85.7; OpenAI CLIP 33.1 / 52.4 / 62.1 / 81.9. At L/14: EVA-CLIP 47.5 / 63.7 / 77.3 / 89.7. DFN (whose data filter was fine-tuned on COCO/Flickr) reaches COCO T→I 59.6 at L/14 and 63.1 at H/14.
  - Source: [SigLIP 2 PDF, arXiv 2502.14786](https://arxiv.org/pdf/2502.14786). Caution: one HTML-summary tool misaligned these columns. The figures above were checked against the PDF text.
- SigLIP 2 multilingual design: "90% of the training image-text pairs is sourced from English web pages, and the remaining 10% from non-English web pages." On XM3600 (36 languages), "SigLIP 2's recall exceeds that of SigLIP by a large margin, while only lagging slightly behind mSigLIP, which in turn performs substantially worse than SigLIP and SigLIP 2 on English-focused benchmarks." Figure 2 plots per-language R@1 including Vietnamese, but gives no numeric table — [arXiv 2502.14786](https://arxiv.org/html/2502.14786v1); [PDF](https://arxiv.org/pdf/2502.14786)
- Known SigLIP 2 weaknesses stated by the authors: the NaFlex variant "interpolates fairly well between training resolutions, but does not extrapolate well," and it omits self-distillation and masked prediction. There is a residual gap to mSigLIP on non-English retrieval (XM3600 T→I 48.1 vs 50.0 at So/16@256) — [arXiv 2502.14786](https://arxiv.org/pdf/2502.14786)
- BEiT-3 paper (giant, finetuned), Table 5: COCO 5K I→T R@1 84.8, T→I R@1 67.2 (R@5 87.7, R@10 92.8); Flickr30K 1K I→T 98.0, T→I 90.3. Zero-shot Flickr30K (Table 6): I→T 94.9, T→I 81.5. It is a dual-encoder usage; pretraining used English corpora plus CC12M/CC3M/SBU/COCO/VG, with a 64k SentencePiece tokenizer — [arXiv 2208.10442 PDF](https://arxiv.org/pdf/2208.10442)
- Released BEiT-3 retrieval checkpoints (384×384): base (222M) COCO IR@1 61.4 / TR@1 79.1, Flickr IR@1 86.2 / TR@1 96.3; large (675M) COCO IR@1 63.4 / TR@1 82.1, Flickr IR@1 88.1 / TR@1 97.2 — [microsoft/unilm beit3 README](https://github.com/microsoft/unilm/tree/master/beit3)
- Vietnamese-specific VLM retrieval: ViCLIP-OT (2026) reports an average Recall@K of 67.34% on UIT-OpenViIC (+5.75 pp over CLIP) and +11.72 pp over CLIP zero-shot on Crossmodal-3600. It also compares against SigLIP baselines, but the abstract gives no SigLIP numbers — [arXiv 2602.22678](https://arxiv.org/abs/2602.22678)

### Inferences
- Complementarity hypothesis. It is plausible, but no published measurement confirms it.
  - BEiT-3 (COCO-finetuned) should be strong on COCO-style literal descriptions ("a man cutting vegetables on a table").
  - SigLIP 2, trained on web alt-text, should be stronger on long-tail entities, text-in-image, and non-English queries.
  - The NNN results show both models have hubness bias of similar size (COCO +3.1 SigLIP, +3.0 BEiT-3), so calibrating both before fusion is sensible.
- The benchmark gaps (COCO T→I 63.4 for BEiT-3-large-ft vs about 55 for SigLIP 2) partly reflect in-domain finetuning on COCO. They may not hold on Vietnamese news/cooking keyframes.

### Gaps
- There is no public number for SigLIP 2 Vietnamese XM3600 R@1; it is only in a bar chart. I could not extract it.
- I found no study that ensembles SigLIP (any version) with BEiT-3.

## 4. Generated captions: caption retrieval vs embedding retrieval, rerank/verification, dense caption embeddings, quality issues

### Takeaway
"Vision-free" retrieval over *rich* VLLM captions can match or beat SigLIP on COCO/Flickr. The key conditions are detailed, structured captions from a strong 8B VLLM and a text retriever fine-tuned on them. Short, generic captions carry little signal beyond the image embedding, and captions lose information on crowded or small-object scenes. BLIP-2 OPT-2.7B-COCO produces short COCO-style captions, so it is closer to the weak regime. In Vietnamese AIC systems, captions are increasingly generated with context from a Gemini-class LVLM, but the published evidence is qualitative.

### Cited Findings
- Vision-Free Retrieval / LexiCLIP (2025): captions from InternVL-2.5-8B-MPO (detailed scene descriptions plus JSON object annotations) are indexed with BGE-large-en-v1.5 (0.3B). T→I R@1: LexiCLIP-ft 91.6 Flickr30K / 67.4 COCO vs SigLIP ViT-B/16 89.1 / 65.7 and CLIP ViT-B 77.8 / 51.0. The authors claim a reduced modality gap and better compositionality. The SigLIP numbers here differ from the SigLIP 2 paper's zero-shot table, which suggests different protocols or checkpoints. Stated limitations: "descriptions of crowded scenes or those containing many small objects are likely to omit a significant amount of information." Captioning costs about 0.2 s/image on an A100, and the method inherits VLLM "biases and hallucinations" — [arXiv 2509.19203](https://arxiv.org/html/2509.19203)
- A caption-augmented CLIP retrieval approach reportedly improves Recall@1 by only about 1% over image-only. This comes from a search snippet whose exact source paper was unclear, so treat it as weak evidence — [search results incl. arXiv 2605.07544](https://arxiv.org/pdf/2605.07544)
- GenSearch uses BLIP-2 captions on the *query side*: images are generated from the query, captioned, and the captions used as paraphrases. On negation queries I2T reached xinfAP 0.149 vs 0.082 for the original query. This shows captioning as a query-expansion tool, not a keyframe index — [arXiv 2407.12341](https://arxiv.org/html/2407.12341v2)
- U-CESE (AI Challenge HCMC 2025, Vietnamese news) builds both a VisualDB (MobileCLIP embeddings) and a TextualDB (Whisper subtitles plus captions). Keyframe captions are generated by Gemini using a window of ±k neighboring keyframes and the subtitle. "ReCap" adds recurrent shot memory so that captions carry program, topic, entity names and locations. The paper shows that no-memory captions describe "individuals as generic participants," while ReCap recovers names and context. Its evidence is qualitative (Fig. 8); I found no retrieval-metric ablation for captions in the text — [arXiv 2605.23274](https://arxiv.org/pdf/2605.23274)
- Multilingual text embedders for cross-lingual caption search: on MKQA cross-lingual retrieval, BGE-M3 (All) reaches Recall@100 75.5 vs 70.9 for mE5-large and 70.1 for E5-mistral-7B (open-domain QA, not captions) — [BGE-M3 arXiv 2402.03216](https://arxiv.org/html/2402.03216v3)
- A search summary of VBS/AIC system papers reports that "CLIP and BLIP-2 struggled with time- and event-specific queries". This is a secondary claim from an EasyChair preprint, found via search only — [EasyChair preprint 15807](https://easychair.org/publications/preprint/dfqX/open)

### Inferences
- With BLIP-2 OPT-2.7B-COCO captions (English, about 10 words, COCO style, trained for generic captions), expect:
  - (a) Little independent recall beyond SigLIP 2 / BEiT-3, because the caption is a lossy summary of the same visual evidence the image embedding already encodes, and BEiT-3 is itself COCO-finetuned.
  - (b) No coverage of the cues that distinguish Vietnamese news: named entities, on-screen text, and events. Those come from OCR and ASR.
  - (c) A hallucination risk that directly causes false positives for object-mention queries.
- Cross-lingual: searching English captions with Vietnamese queries through BGE-M3 or mE5 is feasible (MKQA evidence), but the pipeline stacks two noisy translations: image→English caption, then Vietnamese query↔English caption. Translating the query to English first is simpler, and the legal-QA snippet suggests it helps dense retrieval.
- The LexiCLIP result shows caption retrieval *can* be competitive only with dense VLLM captions and a fine-tuned text retriever. Neither holds for the current BLIP-2 captions.

### Gaps
- I found no controlled study of BLIP-2 OPT-2.7B generic-caption hallucination rates on video keyframes. General VLM hallucination benchmarks (CHAIR/POPE) were not retrieved in this pass.
- I found no published retrieval-metric ablation, with captions on vs off, in any AIC HCMC system paper.
- I found no measurement of BGE-M3 or multilingual-e5 on Vietnamese-query → English-caption image retrieval.

## 5. "Subordinate" (rerank top-K) vs peer-retriever use of captions

### Takeaway
There is no direct head-to-head evidence. The indirect evidence favors keeping weak, generic captions subordinate: a low-weight or rerank-only signal over the top-K of the visual retrievers. Rich, contextual captions can act as a peer retriever (LexiCLIP; U-CESE's TextualDB). In image-text retrieval, the literature pattern of dual-encoder recall followed by a stronger reranker on top-K (ALBEF/BLIP) is well established. It uses cross-encoders, though, not captions.

### Cited Findings
- The BEiT-3 paper's Table 5 separates "Dual encoder + Fusion encoder reranking" systems (ALBEF, BLIP), which rerank dual-encoder top-K with a cross-attention model, e.g. BLIP COCO T→I R@1 65.1. BEiT-3's pure dual encoder reaches 67.2, showing that reranking is a standard way to add a stronger but more expensive signal only on top-K — [arXiv 2208.10442 PDF](https://arxiv.org/pdf/2208.10442)
- GenSearch fuses all branches as peers but had to halve the weight of the weakest branch (T2I). 44 of 210 queries degraded after T2I verification, showing that a noisy branch fused at full weight hurts — [arXiv 2407.12341](https://arxiv.org/html/2407.12341v2)
- Bruch et al.: CC fusion is robust and sample-efficient, and a learned weight near 0 reduces a weak signal to near-subordinate status automatically. Fixed-k RRF gives every list equal voice regardless of quality — [arXiv 2210.11934 PDF](https://arxiv.org/pdf/2210.11934)
- LexiCLIP (dense VLLM captions) works *as* the sole retriever at SigLIP-level accuracy on COCO/Flickr — [arXiv 2509.19203](https://arxiv.org/html/2509.19203)

### Inferences
- Practical design implication, a hypothesis to benchmark:
  1. Retrieve with SigLIP 2 (Vietnamese and English query variants) and BEiT-3 (English translation). Calibrate each, optionally with NNN, and fuse with a tuned convex combination over the union of top-K.
  2. Use BLIP-2 captions only as (a) a low-weight additive term tuned on labeled queries, or (b) a rerank feature on the top few hundred.
  3. Measure whether the tuned caption weight goes to about 0. If it does, the captions are redundant for retrieval, though they may still be useful for display.
- Under RRF with equal weights, adding a weak caption list as a peer can pull hallucination-driven false positives into the top ranks. That is a specific reason to prefer weighted score fusion, or a lower weight on RRF terms, for captions.
- If richer contextual captions (ReCap-style, with subtitle/OCR context) are produced later, re-test them as a peer retriever. The LexiCLIP and U-CESE evidence suggests they carry information the visual embedding lacks, such as entities and program context.

### Gaps
- I found no published experiment comparing captions-as-rerank-only against captions-as-peer-retriever for text-to-image or keyframe retrieval, and none in VBS/AIC system papers either.
- I found no evidence on the optimal rerank depth (top-K) for caption-based verification over keyframes.
