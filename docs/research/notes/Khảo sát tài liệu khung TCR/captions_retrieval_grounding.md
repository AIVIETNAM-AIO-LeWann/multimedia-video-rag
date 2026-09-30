# Generated captions in text→image/video retrieval and temporal grounding: evidence for the two caption roles in our framework

Scope reminder. Our two caption roles are:
- ① **Verification of top-K visual candidates**: s ← s + α·cos(query, caption_emb).
- ② **Per-event unary inside an ordered multi-event DP**: U[i,t] = visual(i,t) + β·cos(sub-query_i, caption_t).

Captions are short COCO-style English captions from BLIP-2 OPT-2.7B-COCO, about 10 words each, over 335k sparse shot keyframes. The visual encoders are SigLIP 2 so400m and BEiT-3-large (COCO-retrieval finetuned). This file builds on `docs/research/notes/Nghiên cứu truy vấn video khoảnh khắc/fusion_and_captions.md` and does not repeat that material (GenSearch, Bruch CC-vs-RRF, NNN, U-CESE/ReCap).

**Correction to that earlier note.** The earlier note gave LexiCLIP-FT "T→I R@1 91.6 Flickr / 67.4 COCO vs SigLIP 89.1 / 65.7". Those are **text-retrieval (I→T)** numbers. The image-retrieval (T→I) numbers are in §2 below and are much lower. They were verified against the PDF.

Training-free vs trained is marked **[TF]** or **[TR]** on each method.

---

## 1. Captions as a bridge for zero-shot temporal grounding / moment retrieval (VTG-GPT, Moment-GPT, ChatVTG, TFVTG)

### Takeaway
Two training-free zero-shot moment-retrieval papers, VTG-GPT and Moment-GPT, measure the same thing: replacing direct frame–query VLM similarity with query–caption text similarity helps. The gain is about +5 to +9 R1@0.5 on QVHighlights and about +3.6 on Charades-STA. The conditions are narrow, though:
- the captions are **long, detailed MLLM descriptions** (MiniGPT-v2, "describe in detail");
- frames are **dense** (0.5–1 fps within one video);
- the text side uses a **text-only encoder** (Sentence-BERT or LLaMA-3 pooled), not CLIP-T;
- the visual baseline is 2021–2023 CLIP, InternVideo or BLIP-2, not SigLIP 2 or BEiT-3-large.

The strongest training-free method that uses **ordered sub-events** (TFVTG, ECCV 2024) uses **no captions at all**. It works on visual similarity plus LLM decomposition, and the LLM decomposition adds only about +2 R1@0.5.

### Cited Findings

**VTG-GPT [TF]** (Xu et al., *Applied Sciences* 2024) — [arXiv 2403.02076](https://arxiv.org/abs/2403.02076)
- Pipeline:
  - Baichuan2-7B-Chat rewrites each query into N_q=5 "debiased" queries.
  - MiniGPT-v2 (LLaMA-2-Chat-7B) captions every sampled frame with the prompt "[image caption] Please describe the content of this image in detail." Sampling is 0.5 fps on QVHighlights and Charades-STA, and 1 frame every 3 s on ActivityNet.
  - Sentence-BERT (RoBERTa) pooled features give S_s = cos(f_q, f_c) ∈ R^{N_q×N_v}.
  - A histogram-based adaptive-threshold proposal generator (10 bins, top k=8, continuity threshold 6) forms proposals.
  - A proposal scorer mixes similarity and length with balance 0.5, followed by NMS at IoU 0.75.
  - It is purely inferential. Source: [PDF](https://arxiv.org/pdf/2403.02076), §3.3–4.1.
- **Caption vs visual ablation** (Table 1, QVHighlights val, proposal generator + scorer, no NMS; R1@0.5 / R1@0.7 / mAP@0.5 / mAP@0.75 / mAP avg):

  | Similarity model | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP avg |
  |---|---|---|---|---|---|
  | CLIP (frame–query, visual) | 45.59 | 26.03 | 45.56 | 23.14 | 24.91 |
  | InternVideo (visual) | 49.13 | 32.49 | 48.65 | 25.82 | 26.94 |
  | CLIP-T on captions | 52.85 | 34.82 | 48.07 | 28.05 | 28.29 |
  | RoBERTa on captions | 54.99 | 37.58 | 53.77 | 29.18 | 30.15 |
  | Sentence-BERT on captions | 54.26 | 38.45 | 53.96 | 29.25 | 30.38 |

  The row-to-number alignment was reconstructed from pdftotext output. The column order is consistent with the text's claims that captions help and that a text-only encoder beats CLIP-T. Sentence-BERT beats visual CLIP by about +8.7 R1@0.5 — [PDF](https://arxiv.org/pdf/2403.02076)
- The authors explain the gain as "over-reliance of traditional methods on directly modeling raw frames, which is often influenced by background details", which the MLLM caption abstracts away — [PDF](https://arxiv.org/pdf/2403.02076)
- Captioner ablation (Table 5, QVH val): with Baichuan2 debiasing, MiniGPT-4 captions give R1@0.5 52.78 / R1@0.7 33.84 / mAP 28.54, and MiniGPT-v2 captions give 54.26 / 38.45 / 30.91. **Caption quality moves R1@0.7 by about 4.6 points** — [PDF](https://arxiv.org/pdf/2403.02076)
- Debiasing with 5 rewrites vs the original query: R1@0.5 +3.87 (to 54.26) and mAP avg +2.59. Performance declines for N_q > 5 — [PDF](https://arxiv.org/pdf/2403.02076)
- Headline: QVHighlights test R1@0.5 53.81, R1@0.7 38.13, mAP avg 30.50. VTG-GPT's Charades-STA R1@0.5 43.68 and R1@0.7 25.94 are cross-checked by the TFVTG paper. VTG-GPT lags Luo et al. on ActivityNet, which the authors attribute to the sparse 1/3 fps sampling. The limitation is stated explicitly: "our framework relies solely on image-based GPT, thus needing more temporal information modeling" — [PDF](https://arxiv.org/pdf/2403.02076); [TFVTG](https://arxiv.org/html/2408.16219)

**Moment-GPT [TF]** (Xu et al., AAAI 2025) — [arXiv 2501.07972](https://arxiv.org/abs/2501.07972); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32971)
- Pipeline:
  - LLaMA-3-8B produces N_d=3 debiased queries.
  - MiniGPT-v2 frame captions ("[image caption] Please provide a detailed description of the image content") at 1 fps on Charades and ActivityNet and 0.5 fps on QVH.
  - The **frame scorer** is S_f = cos(X_d, X_f) using LLaMA-3 pooled features (d=4096).
  - The span generator uses an adaptive histogram threshold (10 bins, count threshold 7, 5 consecutive moments).
  - **Video-ChatGPT captions each candidate span** ("What is this video about?").
  - The **span scorer** is S_s = avg cos(LLaMA3(span caption), LLaMA3(debiased queries)).
  - The final score is S = (1−γ)·S_s + γ·E_s, where E_s is span length and γ=0.2, followed by NMS at IoU 0.9.
  - Source: [PDF](https://arxiv.org/pdf/2501.07972), §3–4.2.
- **Frame-scorer ablation** (Table 5-left, Charades-STA). Row 1 uses BLIP-2 frame–query (visual) similarity directly: R1@0.5 34.8 / R1@0.7 19.3 / mIoU 32.3. The final setting (MiniGPT-v2 captions + LLaMA-3 text features) reaches 38.4 / 21.6 / 36.5, i.e. **+3.6 R1@0.5, +4.2 mIoU**. The intermediate rows (BLIP-2 text encoder on captions; BERT; SimCSE) fall in between, and a text-only LM beats BLIP-2's text encoder. The exact intermediate numbers could not be aligned reliably from the PDF extraction — [PDF](https://arxiv.org/pdf/2501.07972)
- **Span-level caption re-scoring ablation** (Table 7, Charades-STA). This is the closest grounding analogue to role ①.

  | Span scoring | R1@0.5 | R1@0.7 | mIoU |
  |---|---|---|---|
  | None (average of frame-level caption similarity) | 30.2 | 12.4 | 28.7 |
  | MiniGPT-v2 (image-MLLM) captions of span frames | 32.4 | 14.3 | 30.7 |
  | VideoLLaMA span captions | 38.9 | 21.2 | 36.2 |
  | Video-ChatGPT span captions | 38.4 | 21.6 | 36.5 |

  **Image-level captions re-scoring the candidates add only +2.2 R1@0.5. Video-level captions add +8.2** — [PDF](https://arxiv.org/pdf/2501.07972)
- Captioner choice (Table 4, Charades): LLaVA 37.1/20.3/35.2, MiniGPT-4 37.8/20.7/35.9, MiniGPT-v2 38.4/21.6/36.5. Span generator vs shot detection (PySceneDetect) vs sliding window: 38.4 vs 32.1 vs 27.4 R1@0.5 — [PDF](https://arxiv.org/pdf/2501.07972)
- The paper also says that zero-shot methods based on shot detection are "not suitable for scenarios with rapid shot transitions" — [PDF](https://arxiv.org/pdf/2501.07972)

**ChatVTG [TF]** (Qu et al., CVPR 2024 Workshops) — [CVF](https://openaccess.thecvf.com/content/CVPR2024W/PVUW/html/Qu_ChatVTG_Video_Temporal_Grounding_via_Chat_with_Video_Dialogue_Large_CVPRW_2024_paper.html)
- The abstract only says that a Video Dialogue LLM generates **multi-granularity segment captions**, these are matched to the query for coarse grounding, and moment refinement over fine-grained caption proposals follows. It is evaluated on Charades-STA, ActivityNet-Captions and TACoS and "surpasses current zero-shot methods". **Snippet/abstract only**: the PDF returned 403, so there are no numbers or ablations.

**TFVTG [TF]** (Zheng et al., ECCV 2024) — [arXiv 2408.16219](https://arxiv.org/html/2408.16219). Closest structural prior for role ②.
- An LLM outputs sub-events "in chronological order" and a relationship label: single, "simultaneously", or "sequentially".
- Frame similarity is BLIP-2 Q-Former ITC cosine S = f^c·F^vᵀ/(‖f^c‖‖F^v‖), computed per sub-event text. **No captions are used.**
- Proposal score:
  - S^final_{i,j} = max_k (S^dynamic_{i,k} + S^static_{k,j}).
  - The dynamic score sums positive similarity increments D_l above δ=5×10⁻⁴.
  - The static score is the mean similarity inside the segment minus the mean outside.
- Combination: for sequential relations, a combination is discarded "if s^i > e^j" (the order constraint) and the union P₁∪…∪P_m is taken. For simultaneous relations the intersection is taken.
- Results: Charades-STA R1@0.5 49.97, R1@0.7 24.32, mIoU 44.51. ActivityNet R1@0.5 27.02, R1@0.7 13.39, mIoU 34.10.
- Ablation: without the LLM R1@0.5 48.01 (so the LLM decomposition adds +1.96); without the dynamic score 47.63; without the static score 45.48; naive baseline 42.32.

**Other signals**
- A 2026 CVPRW paper "Towards Robust Zero-Shot VTG" and DSE-VTG (arXiv 2609.08850, training-free) exist but were **not read**, so there are no numbers — [CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/papers/Banditakkarakul_Towards_Robust_Zero-Shot_Video_Temporal_Grounding_CVPRW_2026_paper.pdf); [arXiv 2609.08850](https://arxiv.org/pdf/2609.08850)

### Inferences
- The caption-bridge gains in VTG-GPT and Moment-GPT come from three things at once:
  - (a) **detailed** 7B-MLLM descriptions;
  - (b) a stronger **text** encoder than the VLM's own text tower;
  - (c) weak visual baselines (CLIP ViT-B-era, BLIP-2 ITC).
- None of these three transfers cleanly to our setting. Our captions are short COCO-style captions, and our visual baselines are SigLIP 2 so400m (COCO T→I R@1 about 55.8 zero-shot) and BEiT-3-large-COCO (63.4 finetuned). The relative gain should be expected to shrink substantially or vanish. This is an extrapolation and must be measured.
- Moment-GPT's Table 7 is the most directly relevant number for role ①. Re-scoring candidates with **image-level** MLLM captions gave only +2.2 R1@0.5, even with *detailed* MiniGPT-v2 captions. Temporal (video-level) captions did the heavy lifting.
- TFVTG shows that an ordered multi-event decomposition works **without captions**. Its incremental value over the plain query is small (+2 R1@0.5 on Charades). The ordering constraint in role ② is therefore well supported. The caption term inside the unary has no direct precedent in TFVTG.

### Gaps
- There are no ChatVTG numbers or ablations (403 on the PDF).
- I found no zero-shot VTG paper that uses **short BLIP-2 COCO captions** as the bridge, and none that compares the caption bridge against SigLIP / SigLIP 2 / EVA-CLIP-level visual similarity.
- I found no paper that *adds* caption similarity to visual similarity in the frame scorer (VTG-GPT and Moment-GPT *replace* the visual term). The additive "visual + β·caption" variant is untested in this literature.

---

## 2. Caption text-embedding retrieval (LexiCLIP and caption length/density)

### Takeaway
Vision-free retrieval over captions reaches SigLIP-B-level T→I **only** under specific conditions: long, dense VLLM descriptions (up to 256 tokens, plus JSON object lists) and a text retriever fine-tuned on caption↔caption pairs. In zero-shot mode with the same rich captions, it is clearly *below* SigLIP ViT-B/16. **Truncating descriptions to 64 tokens costs about 6 R@1 on Flickr**, and our ~15-token captions are far below even that.

### Cited Findings

**LexiCLIP / "Vision-Free Retrieval"** (Ntinou et al., 2025) — [arXiv 2509.19203](https://arxiv.org/abs/2509.19203)
- Setup:
  - Images are converted by InternVL-2.5-8B-MPO into a detailed scene description plus JSON object annotations.
  - A BGE-large-en-v1.5-based (0.3B) single-tower text encoder is used, or BGE-en-ICL-7B for the 7B variant.
  - The ZS variant uses no image-text training. The FT variant is trained on 1.5M text samples: a first stage on BLIP-2 captions, then a mix of BLIP-2 and compositional captions.
- **Correct T→I (image retrieval) numbers** (Table 3; R@1 / R@10):

  | Model | Flickr30K | COCO |
  |---|---|---|
  | SigLIP ViT-B/16 | 74.6 / 95.6 | 47.8 / 81.0 |
  | BLIP-2 (ViT-L) | 74.5 / 97.0 | 50.0 / 86.1 |
  | EVA-02-CLIP L-336 | 78.0 / 96.8 | 47.9 / 80.0 |
  | CLIP ViT-L | 67.3 / 93.3 | 37.0 / 71.5 |
  | **LexiCLIP-ZS (0.3B)** | 69.5 / 94.2 | **41.7** / 76.7 |
  | **LexiCLIP-FT (0.3B)** | 79.2 / 97.4 | **52.7** / 84.5 |

  The I→T column gives FT 91.6 / 67.4, which is the source of the earlier note's error — [PDF](https://arxiv.org/pdf/2509.19203)
- **Caption length ablation** (Table 9, LexiCLIP-0.3B-ZS, Flickr T→I R@1):

  | Max description tokens | T→I R@1 |
  |---|---|
  | 64 | 63.3 |
  | 128 | 67.5 |
  | 256 | 69.5 |
  | 512 | 69.1 |

  The authors: "a clear performance drop for descriptions shorter than 128 and diminishing returns above 256" — [PDF](https://arxiv.org/pdf/2509.19203)
- **Density ablation** (Table 7a): removing the object-based descriptions drops 0.3B-ZS from 69.5 to 66.4 (Flickr) and from 41.7 to 38.7 (COCO) R@1 — [PDF](https://arxiv.org/pdf/2509.19203)
- Captioner size is less critical when the output is rich. InternVL2.5-MPO 4B/8B and InternVL3 9B/14B all give Flickr R@1 of 74.3–75.6 with the 7B retriever, and the FT model transfers across captioners — [PDF](https://arxiv.org/pdf/2509.19203)
- Information loss: generated descriptions match only **69%** of ground-truth COCO object classes under exact class-name matching (possibly more with synonyms). The authors state that crowded or small-object scenes lose information — [PDF](https://arxiv.org/pdf/2509.19203)
- Modality gap on Flickr (PCA centroid distance): SigLIP 1.008, LexiCLIP pre-FT 0.476, post-FT 0.260 — [PDF](https://arxiv.org/pdf/2509.19203)

**Text encoder choice for caption matching**
- CLIP's text encoder is weaker than text-only encoders for caption↔query matching:
  - VTG-GPT: CLIP-T 52.85 vs RoBERTa/SBERT 54.3–55.0 R1@0.5.
  - Moment-GPT: BLIP-2 text encoder below BERT, SimCSE and LLaMA-3.
  - Sources: [VTG-GPT](https://arxiv.org/pdf/2403.02076); [Moment-GPT](https://arxiv.org/pdf/2501.07972)
- LexiCLIP notes that CLIP's effective text length is about 20–25 tokens despite the 77-token limit — [PDF](https://arxiv.org/pdf/2509.19203)

### Inferences
- The regime where caption retrieval "matches or beats CLIP" requires dense descriptions of at least 128 tokens *and* fine-tuning. Our BLIP-2 COCO captions (about 10 words, about 15 tokens) are 4–8× shorter than LexiCLIP's worst ablated setting.
- A reasonable prior is that **standalone** BLIP-2-caption retrieval will sit well below SigLIP 2 and BEiT-3 on our data. LexiCLIP-ZS with rich captions was already −6 R@1 vs SigLIP-B/16 on COCO.
- If captions are used at all, embed them with a text-only encoder (BGE/E5/GTE/MiniLM-class) rather than the SigLIP or BEiT-3 text towers. The two VTG papers agree on this direction, although both measure it with detailed captions.

### Gaps
- There is no published measurement of BGE/E5/GTE/all-MiniLM retrieval over **short BLIP-2 captions** vs CLIP/SigLIP image retrieval on COCO. LexiCLIP ablates length only from 64 tokens upward.
- There is no MTEB-style measurement specific to caption↔query short-text similarity for these encoders in this domain.

---

## 3. Fusing caption-text similarity with image-text similarity; captions used to rerank top-K

### Takeaway
The best-documented form of role ① is **"rerank the top-k of a base retriever by a weighted sum with a second similarity"**, as in CLIPRerank. There, the second signal is CLIP *image* similarity, not caption similarity. Gains are large when the base model is weak and small (+0.6% to +4.7% relative mean infAP) when the base model is already CLIP-class. For re-scoring with *captions* specifically, the only measured grounding analogue is Moment-GPT's: image-MLLM captions gave +2.2 R1@0.5, while video captions gave +8.2. I found no measured additive caption + image fusion for keyframe or image retrieval with strong dual encoders.

### Cited Findings

**CLIPRerank [TF]** (Chen et al., ICASSP 2024) — [arXiv 2401.08449](https://arxiv.org/pdf/2401.08449); [GitHub](https://github.com/ruc-aimc-lab/CLIPRerank)
- Formula: S_re(q,v) = λ·M(q,v) + (1−λ)·S(q,v), where S is CLIP frame–query cosine max-pooled over frames. k = 1000 and λ = 0.4.
- TRECVID AVS mean infAP (TV16–21), before → after reranking:

  | Base model | Before | After | Relative gain |
  |---|---|---|---|
  | W2VV++ | 0.154 | 0.183 | +18.8% |
  | DE | 0.170 | 0.196 | +15.3% |
  | X-CLIP | 0.181 | 0.209 | +15.5% |
  | LAFF (already uses CLIP features) | 0.221 | 0.226 | +2.1% |
  | LAFF* | 0.296 | 0.305 | +3.1% |
  | TS2-Net | 0.193 | 0.202 | +4.7% |
  | SEA | 0.154 | 0.155 | +0.6% |

- With BLIP-2 as the reranker over LAFF* on TV22 (k = 5k, λ = 0.5), infAP rose from 0.241 to 0.271.
- The authors report that difficult queries remain that re-scoring "fails to respond to".
- Row labels for the middle entries were reconstructed from a column-scrambled PDF extraction. The W2VV++, LAFF, LAFF* and BLIP-2 numbers are the most reliable.

**Moment-GPT span scorer [TF]**
- Re-scores generated candidate spans with caption-to-query similarity, and this *replaces* the frame-average score (see §1).
- Image-caption re-scoring: +2.2 R1@0.5. Video-caption re-scoring: +8.2.
- Source: [PDF](https://arxiv.org/pdf/2501.07972)

**GenSearch (earlier note)** fused BLIP-2-caption-derived query paraphrases with weight 1 and had to down-weight its weak branch — [arXiv 2407.12341](https://arxiv.org/html/2407.12341v2)

**Search-level evidence** (snippet only, no numbers verified):
- VBS/AIC systems combine CLIP, BLIP-2 and BEiT-3 embeddings "with an ensemble score" — [EasyChair 15807](https://easychair.org/publications/preprint/dfqX/open); [CVPRW 2025 IViSE](https://openaccess.thecvf.com/content/CVPR2025W/IViSE/papers/Quan_Toward_Automation_in_Text-based_Video_Retrieval_with_LLM_Assistance_CVPRW_2025_paper.pdf)

### Inferences
- CLIPRerank's pattern suggests that a re-scoring signal helps in proportion to how much *new* information it carries relative to the base retriever. When the base model already contained CLIP features (LAFF, SEA), the gain collapsed to about 0.6–3%.
- BLIP-2 OPT-2.7B-**COCO** captions are generated by a model trained on COCO captions. BEiT-3-large is a retriever finetuned on COCO captions. The caption term in ① is therefore plausibly highly correlated with the BEiT-3 score, which is the "LAFF/SEA" low-gain regime. This is a hypothesis and should be measured, e.g. with the Spearman correlation between cos(q, caption) and BEiT-3 cos over top-K.
- A tuned α (convex combination on labeled queries, per Bruch et al. in the earlier note) is the right test. If α* → 0, drop the term.

### Gaps
- I found no paper reporting COCO/Flickr or keyframe KIS numbers for "dual-encoder image score + α·cos(query, generated-caption embedding)" with a strong dual encoder.
- I found no evidence on the optimal rerank depth K for caption verification.

---

## 4. Captions and textual descriptions in step localization / procedure alignment

### Takeaway
The training-free state of the art for step grounding does not use frame captions. BaGLM (NeurIPS 2025) asks an LMM directly for per-segment step probabilities and imposes an **LLM-estimated prerequisite (ordering) prior** through Bayesian filtering, which is an HMM-style forward pass. The ordering prior adds about +5 R@1 over the raw LMM on HT-Step. Caption-based step grounding (HowToCaption, DenseStep2M) appears mainly as *training-data generation*, where captions are built from ASR plus LLM rather than from generic image captions.

### Cited Findings

**BaGLM [TF]** (Zanella et al., NeurIPS 2025) — [arXiv 2510.16989](https://arxiv.org/html/2510.16989); [NeurIPS PDF](https://papers.nips.cc/paper_files/paper/2025/file/8f18b9b639c6c376556ffeba6420d37a-Paper-Conference.pdf)
- **Prerequisite prior.** LLaMA3-70B-Instruct estimates D ∈ R^{K×K}, where D_{i,j} = P(step a_j is a prerequisite of a_i). The transition is T = Dᵀ with self-transitions and row normalization.
- **Progress adjustment.** T is reweighted per time step by readiness r_t (weighted maximum past progress of the prerequisites) and validity v_t (successors not yet completed): T̃_t[i,j] ∝ T[i,j]·r_t[j]·v_t[j].
- **Predict step.** predict_t(a_i) = Σ_j T̃_t[j,i]·bel_{t−1}(a_j).
- **Update step.** The belief is multiplied by the LMM's direct step prediction f_LMM(S_t, π_VSG)[i] over 2-second segments. Progress is also read from LMM token probabilities over 0–9.
- Results: HT-Step R@1 57.4, CrossTask Avg R@1 59.8, Ego4D Goal-Step 43.3. For comparison, NaSVA offline scores 53.1 / 46.7 (CrossTask) / 29.1, and VINA scores 39.1 / 44.8.
- The raw InternVL2.5-8B scores 52.0 on HT-Step.
- Transition ablation (HT-Step / CrossTask / Ego4D): no readiness and no validity 55.9 / 58.0 / 42.1; both 57.4 / 59.8 / 43.3.
- The earlier methods listed, VINA and NaSVA, are **[TR]** on HowTo100M narrations.
- BaGLM describes prior work as matching step text to frames via CLIP cosine, which has an "inability to effectively capture temporal dynamics".

**Caption/ASR→LLM pipelines**
- HowToCaption (ECCV 2024) prompts an LLM to turn ASR subtitles into plausible video captions for training — [ECCV PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07249.pdf)
- DenseStep2M (2026) is a training-free pipeline that reasons jointly over frames and transcripts to produce temporally grounded steps. It was not read beyond the snippet — [arXiv 2604.26565](https://arxiv.org/html/2604.26565v1)

### Inferences
- The closest prior work for role ② is a combination: **TFVTG** (ordered sub-events from an LLM, per-frame VLM similarity, hard order constraint) and **BaGLM** (soft ordering via an LLM prerequisite matrix inside a recursive filter, with per-segment LMM likelihood).
- Our DP with a per-event visual unary is essentially TFVTG's ordering idea implemented as a Viterbi/DTW-style exact optimizer over sparse keyframes.
- The novel part is the β·cos(sub-query, caption_t) term. Neither of those papers uses captions in the unary. BaGLM's per-segment evidence is an LMM *judgment over the whole candidate step list* (a discriminative, contrastive choice among steps), which is what disambiguates similar-looking steps.
- A generic caption such as "a person is cooking food in a pan" is not contrastive across steps. It will produce nearly equal cos for sub-queries "stir-fry the garlic" and "add the fish sauce", so β·cos adds a near-constant offset and no discrimination.
- For visually similar cooking steps, the discriminating information is usually in OCR/ASR (ingredient names, spoken instructions), not in BLIP-2 COCO captions. HowToCaption's ASR→caption approach points the same way.

### Gaps
- I found no training-free step-grounding paper that uses generated **image captions** as the per-frame unary.
- I found no measurement of caption distinctiveness across adjacent procedural steps.

---

## 5. Caption hallucination and genericness (BLIP / BLIP-2)

### Takeaway
On in-domain COCO, BLIP-2 OPT-2.7B hallucinates rarely by closed-vocabulary CHAIR (CHAIR_s 2.6, CHAIR_i 1.7). By open-vocabulary OpenCHAIR the rate is about 17%. On out-of-domain Vietnamese news/cooking keyframes, rates should be expected to be higher, but I found no measurement. Genericness, i.e. low information, is the bigger practical issue for retrieval. LexiCLIP shows that information loss grows as descriptions get shorter.

### Cited Findings
- MOCHa (Ben-Kish et al., "Mitigating Open-Vocabulary Caption Hallucinations"), COCO Karpathy test, CHAIR_i / CHAIR_s:

  | Model | CHAIR_i | CHAIR_s | OpenCHAIR | + MOCHa (CHAIR_i / CHAIR_s / OpenCHAIR) |
  |---|---|---|---|---|
  | BLIP-Base | 2.6 | 2.8 | 17.6% | 2.2 / 2.5 / 16.4% |
  | BLIP-Large | 2.3 | 3.5 | 19.2% | 2.1 / 3.1 / 18.3% |
  | BLIP-2 (OPT-2.7B) | 1.7 | 2.6 | 17.0% | 1.4 / 2.3 / 16.6% |

  The authors argue that closed-vocabulary CHAIR (80 COCO objects) misses most real hallucinations — [arXiv 2312.03631](https://arxiv.org/html/2312.03631v3)
- Even rich 8B-VLLM descriptions recover only 69% of GT COCO object classes (exact match) — [LexiCLIP PDF](https://arxiv.org/pdf/2509.19203)
- U-CESE (earlier note) shows qualitatively that context-free captions describe people as "generic participants" — [arXiv 2605.23274](https://arxiv.org/pdf/2605.23274)
- Related works seen only via search, **not read**: ALOHa (a hallucination metric) and "Plausible May Not Be Faithful" (object hallucination in VLP captioners) — [ALOHa](https://arxiv.org/html/2404.02904v1); [arXiv 2210.07688](https://arxiv.org/pdf/2210.07688)

### Inferences
- About 17% of captions carrying an out-of-vocabulary hallucinated object, even in-domain, is a relevant false-positive rate for role ①. An object-mention query such as "a bowl of noodles" could get a caption boost on frames without that object.
- The more serious problem for roles ① and ② is **genericness**: COCO-style captions collapse many distinct shots, such as news anchors or cooking-at-a-pan frames, onto a handful of templates.
- Cheap diagnostic, no literature needed: count unique captions per video and compute mean pairwise cosine between captions of consecutive keyframes. If most adjacent keyframes in cooking videos share near-identical captions, role ② cannot discriminate steps.

### Gaps
- I found no hallucination or genericness measurement for BLIP-2 on video keyframes, news or cooking domains.
- I found no study linking caption hallucination rate to retrieval precision.

---

## 6. Cost of encoding 335k short captions with small text encoders

### Takeaway
Encoding cost is negligible compared with captioning. The Sentence-Transformers documentation lists MiniLM-L6-class models at about 18,000 queries/s on GPU (hardware not stated in the fetched excerpt) and mpnet-base-class models at about 4,000/s. That implies roughly 20 s to 1.5 min for 335k ten-word captions, well within a 6 GB GPU for these sizes. This is an extrapolation from documented speeds.

### Cited Findings
- multi-qa-MiniLM-L6-dot-v1: "18,000 / 750" queries per second (GPU/CPU). multi-qa-mpnet-base-dot-v1: "4,000 / 170". all-MiniLM-L6-v2 is described as "5 times faster" than all-mpnet-base-v2 — [SBERT pretrained models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- LexiCLIP reports about 0.2 s/image on an A100 for captioning with an 8B VLLM (from the earlier note), so *re-captioning* is the real cost — [arXiv 2509.19203](https://arxiv.org/html/2509.19203)

### Inferences
- 335k × 384-dim float32 (MiniLM) is about 0.5 GB, or 1 GB at 768-dim. Exact CPU dot products over top-K candidates are trivial. Role ① and ② never need an ANN index over captions, only lookups by frame_uid.
- BGE-large or E5-large (335M) will likely run at a few hundred to about 1–2k short sentences/s on a 6 GB consumer GPU with fp16. This is an estimate and was not sourced.

### Gaps
- I found no sourced throughput for BGE-M3, BGE-large, E5 or GTE on a 6 GB-class GPU, and the fetched SBERT excerpt did not name its benchmark hardware.

---

## 7. Judgment on our two caption roles and the closest prior work

### Takeaway
- **Role ①** has a clear methodological precedent: weighted-sum re-scoring of top-k (CLIPRerank; Moment-GPT's caption span scorer). The literature predicts **small or zero gain** when the added signal is redundant with the base retriever. Short BLIP-2-COCO captions on top of SigLIP 2 and BEiT-3-COCO are likely that case.
- **Role ②** has a precedent for the *ordering* part (TFVTG's order constraint; BaGLM's prerequisite transitions) but **no precedent for a caption-similarity unary**. The mechanism that disambiguates similar steps in the literature is a contrastive LMM judgment or ASR text, not a generic caption.
- Both roles should be kept as **ablatable, tuned-weight terms whose default is 0**. They need a pre-registered test, not an assumed contribution.

### Cited Findings

**Closest prior to ① (verification of top-K by adding caption similarity)**
- **CLIPRerank**: S_re = λ·M + (1−λ)·S over the top-k=1000 of the base model, λ=0.4. The added score is CLIP image–query similarity, not caption similarity. Gains are +0.6% to +18.8% relative infAP and shrink when the base model already uses CLIP features — [arXiv 2401.08449](https://arxiv.org/pdf/2401.08449)
- **Moment-GPT span scorer**: re-scores generated candidates by cos(LLaMA3(caption), LLaMA3(query)). Image-MLLM captions gave +2.2 R1@0.5 and video-MLLM captions +8.2 on Charades-STA. The caption score *replaces* rather than adds to the base score, and the base score is itself caption-based — [arXiv 2501.07972](https://arxiv.org/pdf/2501.07972)
- **VTG-GPT**: caption text-similarity with detailed MiniGPT-v2 captions beats CLIP visual similarity by about +8.7 R1@0.5 on QVH. The baseline is weak (CLIP) and there is no additive fusion — [arXiv 2403.02076](https://arxiv.org/pdf/2403.02076)
- **What ① adds beyond these**: an additive caption term on top of *strong, COCO-tuned* dual encoders over a corpus of 335k keyframes, rather than within a single video. No paper measures this.

**Closest prior to ② (ordered multi-event DP with a caption unary)**
- **TFVTG**: LLM sub-events plus "sequentially" relation, per-frame BLIP-2 ITC unary, and a hard order constraint (discard if s^i > e^j). The LLM decomposition adds +1.96 R1@0.5 on Charades. It uses **no captions** — [arXiv 2408.16219](https://arxiv.org/html/2408.16219)
- **BaGLM**: an LLM-estimated prerequisite matrix plus progress-aware transitions in a Bayesian filter, with a per-segment LMM likelihood. The transition model adds +5.4 R@1 over the raw LMM on HT-Step (52.0 → 57.4). It uses **no captions** — [arXiv 2510.16989](https://arxiv.org/html/2510.16989)
- **What ② adds beyond these**: exact DP over *sparse* shot keyframes with a hybrid visual + caption unary. The caption part lacks evidence.

**Conditions under which captions *did* help, and whether our setup meets them**

| Condition from the literature | Our setup |
|---|---|
| Detailed MLLM captions ([VTG-GPT](https://arxiv.org/pdf/2403.02076); [Moment-GPT](https://arxiv.org/pdf/2501.07972)) | No: ~10-word COCO-style captions |
| Descriptions of 128–256 tokens ([LexiCLIP](https://arxiv.org/pdf/2509.19203)) | No: ~15 tokens |
| Text-only encoder for matching | Yes, achievable |
| Weak visual baseline | No: SigLIP 2 / BEiT-3-large |
| Temporal (video-level) captions for candidate re-scoring ([Moment-GPT Table 7](https://arxiv.org/pdf/2501.07972)) | No: per-keyframe captions only |

### Inferences
- **Role ① judgment.** Expected gain on our data is small, with a real risk of hallucination-driven false positives (OpenCHAIR about 17% in-domain). Keep it only if a tuned α > 0 improves R@k or MRR on held-out labeled queries.
  - Encode captions with a text-only model (BGE/E5/MiniLM).
  - Normalize cos per query over the top-K before adding it, since sentence-encoder cosines are compressed and differ in scale from SigLIP and BEiT-3.
  - Report the correlation between the caption score and BEiT-3 as the redundancy diagnostic.
  - Expected failure mode: α* ≈ 0 because of COCO-on-COCO redundancy with BEiT-3.
- **Role ② judgment.** The ordering DP is well founded (TFVTG, BaGLM). The β·caption unary is the weakest component of the framework for the stated motivation, visually similar cooking steps.
  - Short generic captions are unlikely to differ between such steps, so the term adds a near-constant per-video offset (no effect) or noise.
  - Before tuning β, run the caption-distinctiveness diagnostic from §5.
  - More promising substitutes for the "text disambiguates steps" function:
    - (a) ASR/OCR text windows as the text unary;
    - (b) a contrastive per-keyframe choice among the K sub-queries, softmax over the steps (BaGLM-style), instead of an absolute cos per step;
    - (c) richer, context-aware captions (ReCap-style with ASR/OCR) if re-captioning is affordable.
- **A cheap improvement for either role.** Use a *within-video contrastive* caption score, e.g. cos(sub-query_i, caption_t) − mean_j cos(sub-query_j, caption_t), or a softmax over the steps. This removes the generic offset and is analogous to TFVTG's static score (inside minus outside) and BaGLM's normalization over steps. This is an inference from their formulations, not a tested result.
- **Wording for the framework document.** Captions should be described as an *auxiliary, ablated* signal, not a core modality. Do not claim they "disambiguate visually similar steps" unless the ablation shows it.

### Gaps
- No paper tests the exact formulations ① or ②. Both need in-house ablation, with α, β ∈ {0, tuned} on labeled KIS/TRAKE-style queries, including multi-event cooking queries.
- ChatVTG's numbers and two 2026 training-free VTG papers were not read.
- There are no domain-specific (Vietnamese news/cooking) caption quality measurements.
