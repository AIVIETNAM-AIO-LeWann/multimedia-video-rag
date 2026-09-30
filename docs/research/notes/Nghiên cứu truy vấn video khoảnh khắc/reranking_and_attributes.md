# Reranking and attribute/compositional query handling for text-to-keyframe retrieval

Context assumed by these notes: 335k keyframes, precomputed SigLIP 2 so400m + BEiT-3 large embeddings, object boxes (400 labels, no color), CPU search, occasional Colab/Kaggle T4 offline, interactive latency needed.

## Q1. Cross-modal rerankers (BLIP/BLIP-2 ITM, cross-encoders, VLM-as-reranker, jina-reranker-m0, Qwen3-VL-Reranker): gains, top-K, latency, CPU vs GPU

### Takeaway
Cross-attention/VLM rerankers give large R@1 gains on COCO/Flickr (e.g. LamRA Qwen2-VL pointwise rerank: COCO t2i R@1 53.3 -> 77.0), but they are standardly applied to a small candidate pool (k=10–128) and benchmarked on multi-GPU hardware; none of the sources report CPU latency, so on this project they are realistic only on GPU (T4) or for a very small top-K.

### Cited Findings
- BLIP and BLIP-2 retrieval inference: "first select k=128 candidates based on the image-text feature similarity, followed by a re-ranking based on pairwise ITM scores" — [BLIP-2 (arXiv 2301.12597)](https://arxiv.org/html/2301.12597); same k=128 ITC->ITM protocol in BLIP — [BLIP (arXiv 2201.12086)](https://arxiv.org/pdf/2201.12086)
- In the ITC+ITM protocol the final score is the sum of ITC and ITM scores over top-N=128 candidates — [ReFIT / BLIP protocol description via search result](https://arxiv.org/pdf/2305.11744) (secondary description; verify in BLIP paper)
- BLIP-2 (ViT-g) reported with this rerank protocol: zero-shot Flickr30K t2i R@1 89.7; fine-tuned COCO 5K t2i R@1 68.3 — [BLIP-2](https://arxiv.org/html/2301.12597). Note: I could not locate a clean "ITC-only vs ITC+ITM" ablation row in BLIP-2; the ablation fetched was about the ITG loss, not the rerank.
- LamRA (Qwen2-VL 2B/7B backbone) supports pointwise (P(YES) as score) and listwise (outputs index of best candidate) reranking; reranks top-50 on M-BEIR and top-10 on unseen datasets — [LamRA (arXiv 2412.01720)](https://arxiv.org/html/2412.01720)
- LamRA text-to-image R@1: Flickr30K 80.8 -> 95.3 with pointwise rerank; MSCOCO 53.3 -> 77.0 — [LamRA](https://arxiv.org/html/2412.01720)
- LamRA per-query inference cost measured on 8x A100, batch 32: pointwise 0.020–0.084 s, listwise 0.010–0.099 s depending on task — [LamRA](https://arxiv.org/html/2412.01720)
- Qwen3-VL-Reranker (2B: 28 layers; 8B: 36 layers; 32K ctx; Apache-2.0) is a single-tower model scoring via probability of "yes"/"no" tokens; intended as second stage after Qwen3-VL-Embedding recall. MMEB-v2 retrieval avg 75.2 (2B) / 79.2 (8B), image retrieval 78.2 (8B). No recommended top-K or latency given — [QwenLM/Qwen3-VL-Embedding GitHub](https://github.com/QwenLM/Qwen3-VL-Embedding); [Qwen3-VL-Embedding-2B model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- jina-reranker-m0: 2.4B-parameter VLM-based multilingual multimodal reranker; supports text->image, image->text, mixed; SOTA on ViDoRe (91.02 NDCG@5); BEIR 58.95 NDCG@10; evaluated also on M-BEIR and Winoground; GGUF quantizations exist — [Jina model page](https://jina.ai/models/jina-reranker-m0/), [Jina blog](https://jina.ai/news/jina-reranker-m0-multilingual-multimodal-document-reranker/), [GGUF repo](https://github.com/jina-ai/jina-reranker-m0-gguf). Its focus is visual documents; I found no reported natural-photo (COCO/Flickr) t2i numbers.
- MLLM reranking cost is a recognised bottleneck: miniReranker (2026) reuses visual caches and sparsifies interaction, reaching ~15% of original latency for MS COCO i2t top-100 and <1% for video top-100, keeping 96.3–98.6% of dense reranker quality (MMEB-v2, 2B/4B/8B; baselines incl. Qwen3-VL-Reranker, GME, VLM2Vec) — [miniReranker (arXiv 2606.10759)](https://arxiv.org/html/2606.10759)
- "MLLM Is a Strong Reranker" (RoRA-VLM-style knowledge-enhanced reranking for multimodal RAG) — [arXiv 2407.21439](https://arxiv.org/pdf/2407.21439) (not fetched in detail)
- Qwen2-VL used as a scorer of image–caption similarity followed by rank-aware selection in an event-aware image retrieval system; larger Qwen2-VL (7B vs 2B) performed better — [EVENT-Retriever (arXiv 2509.00751)](https://arxiv.org/html/2509.00751v1)
- BLIP's ITM head is much better at color-word-order binding than contrastive scoring: on ColorSwap, BLIP ITM ~87 vs CLIP ~12 and SigLIP ~30 (as reported in summary; see Q4 for caveats) — [ColorSwap (ACL Findings 2024)](https://aclanthology.org/2024.findings-acl.99/)

### Inferences
- The gains are largest where first-stage embeddings are weakest (COCO: +23.7 R@1 for LamRA). SigLIP 2 so400m is a much stronger first stage than LamRA's own embedding, so the absolute gain on top of it should be expected to be smaller; this must be measured on project queries.
- CPU feasibility (my estimate, no source): a 2B VLM reranker processes each (query, image) pair with hundreds of visual tokens; on CPU this is likely seconds per pair, so top-50 reranking would take minutes — not interactive. BLIP/BLIP-2 ITM (ViT + Q-Former/BERT cross-attention) is lighter but still a full image forward per candidate unless image-side features are cached. Practical design: cache image-side encoder outputs offline on T4 (BLIP-2 Q-Former query tokens are 32x768 per image — caching for 335k frames is ~ 335k*32*768*2 B ≈ 16 GB in fp16) and run only the text-conditioned cross-attention part at query time; or offer VLM rerank as an explicit "slow/verify top-20" button on GPU.
- Listwise VLM reranking of top-10 is the cheapest VLM mode per query (one call) and fits interactive "verify" steps.

### Gaps
- No source reported CPU latency for BLIP-2 ITM, jina-reranker-m0, or Qwen3-VL-Reranker.
- No clean ablation found (in fetched text) quantifying BLIP-2 ITC-only vs ITC+ITM R@1.
- No published numbers found for jina-reranker-m0 or Qwen3-VL-Reranker on natural-image COCO/Flickr t2i, nor on video keyframe KIS tasks.
- No evidence found on reranking on top of SigLIP 2 specifically.

## Q2. Global-feature reranking without cross-attention (SuperGlobal, αQE, AQE, DBA, diffusion) and transfer to text queries

### Takeaway
SuperGlobal shows global-feature-only reranking is highly effective and cheap for image-to-image landmark retrieval, but all evidence found is uni-modal (image query); for text queries it can only be used as a second-round "pseudo-relevance feedback" step (use top-ranked images as an image query), which is known to help recall but is fragile when the initial top-k is noisy.

### Cited Findings
- SuperGlobal (Shao et al., ICCV 2023) uses only global features in both stages; reranking refines global features of query and top-ranked images using a small set of neighbours; +3.7% two-stage gain on Revisited Oxford+1M Hard with a 64,865x speedup vs local-feature reranking; +7.1% single-stage; surpasses previous single-stage SOTA by 16.3% — [arXiv 2308.06954](https://arxiv.org/abs/2308.06954), [CVF open access](https://openaccess.thecvf.com/content/ICCV2023/html/Shao_Global_Features_are_All_You_Need_for_Image_Retrieval_and_ICCV_2023_paper.html), [code](https://github.com/ShihaoShao-GH/SuperGlobal)
- Pseudo-relevance feedback builds a new query from first-round results; effectiveness is "constrained by the quality of the top-k pseudo-relevant documents" since non-relevant items inject noise — [Enhancing Interactive Image Retrieval With Query Rewriting (arXiv 2404.18746)](https://arxiv.org/pdf/2404.18746)
- Inter-media pseudo-relevance feedback (using image modality to expand the text query) significantly improved a multimedia retrieval system in ImageCLEF — [Springer, ImageCLEF](https://link.springer.com/chapter/10.1007/978-3-540-74999-8_92)
- Interactive video retrieval systems apply Rocchio feedback on CLIP/SigLIP2 vectors: users mark preferred/non-preferred frames and the query vector is moved accordingly (Vortex, HCMC AI Challenge 2025) — [Vortex (arXiv 2606.19682)](https://arxiv.org/html/2606.19682)

### Inferences
- αQE/AQE/DBA are image-image operations. For text queries they transfer as: (a) text->image first stage, then (b) blend the text embedding with a weighted mean of top-k image embeddings (αQE weights by similarity^α) in the same SigLIP 2 space and re-search. Because SigLIP text and image embeddings live in a shared but modality-gapped space, mixing weights must be tuned; this is effectively automatic Rocchio.
- DBA on the 335k-frame database (replace each frame embedding with a weighted mean of its k image neighbours) is offline and free at query time; it tends to merge near-duplicate keyframes of the same shot, which may help KIS recall but blur fine attributes. Needs benchmark.
- User-driven Rocchio (as in Vortex) is safer than blind PRF for KIS because the user filters noise.

### Gaps
- I found no paper evaluating SuperGlobal, αQE, DBA, or diffusion reranking for text-to-image (cross-modal) queries with CLIP/SigLIP; transfer is unproven.
- No quantitative result for Rocchio in VBS/AIC systems was found.

## Q3. Temporal-context reranking for video keyframes (neighbour aggregation, shot pooling)

### Takeaway
AIC/VBS systems routinely use neighbour-score aggregation and multi-stage (before/now/after) temporal search, but published evidence is largely qualitative; the main risk is short scenes/sudden cuts where neighbours belong to a different shot.

### Cited Findings
- A HCMC-AIC-style moment retrieval system proposes neighbour score aggregation: compute similarity for neighbouring frames around a candidate keyframe and aggregate, motivated by "areas surrounding a keyframe are most likely to possess visual and semantic features in common" after shot detection; "conventional reranking overlooks temporal context, resulting in inconsistent rankings" — [arXiv 2504.08384](https://arxiv.org/html/2504.08384)
- Same paper: evidence is qualitative (case study showing a single BEiT-3 model ranks the correct frame low and neighbour reranking raises it; ensemble + reranking "most robust"); no recall/mAP reported; uses "conditional score checking" to handle "sudden changes in the scene" but no empirical failure analysis — [arXiv 2504.08384](https://arxiv.org/html/2504.08384)
- Vortex (HCMC AI Challenge 2025, 79.6/88 = 90.5% preliminary): CLIP ViT-L/14 + SigLIP2 fused with RRF; multi-stage temporal search decomposes queries into Before/Now/After with O(K log K) reranking; keyframes via AutoShot + CLIP-difference filter (threshold 0.4, every 8 frames) — [Vortex (arXiv 2606.19682)](https://arxiv.org/html/2606.19682)
- vitrivr at VBS supports queries specifying temporal context — [Springer, Multi-modal Interactive Video Retrieval with Temporal Queries](https://link.springer.com/chapter/10.1007/978-3-030-98355-0_44)

### Inferences
- Failure modes (reasoned, not measured in sources): (1) short shots/fast cuts — neighbour windows cross shot boundaries and pull in unrelated content; restrict aggregation to within-shot neighbours using shot IDs and use max- or weighted pooling rather than mean; (2) sparse keyframes — if the keyframe extractor already removes near-duplicates, "neighbours" can be seconds apart; (3) static news-anchor segments inflate scores for long shots; normalise by shot length.
- Recommended evaluation: compare frame-only vs within-shot max/mean vs cross-shot window on the project's KIS benchmark, sliced by shot length.

### Gaps
- No quantitative ablation of temporal neighbour reranking on a public KIS benchmark found.

## Q4. Attribute-binding failures of CLIP-like models (ARO, Winoground, SugarCrepe, CREPE, EqBen); SigLIP / SigLIP 2 / BEiT-3

### Takeaway
Contrastive dual encoders (CLIP, SigLIP, SigLIP 2) still fail systematically at attribute-object binding and swaps; SigLIP 2 is better than CLIP/SigLIP but far from reliable (e.g. 54.6% half-truth accuracy vs 40.6% CLIP), while cross-attention ITM heads (BLIP) are much better on colour swaps.

### Cited Findings
- SugarCrepe: 7 hard-negative types (replace/swap/add over object/attribute/relation), 7,512 examples; showed ARO/CREPE-style benchmarks are hackable — text-only "blind" models beat VLMs on 9 of 10 prior tasks; on SugarCrepe, best CLIP ~73% on swap-att and ~86% replace-att vs 99% human; "all models struggle at identifying Swap hard negatives, regardless of their pretraining dataset and model size" — [SugarCrepe (arXiv 2306.14610)](https://arxiv.org/html/2306.14610)
- Half-truth test on MS-COCO (adding a plausible but wrong entity/relation to a correct caption should lower score): overall accuracy CLIP 40.6%, SigLIP 45.7%, SigLIP 2 54.6% (relation additions: 32.9 / 38.8 / 45.2) — [Half-Truths Break Similarity-Based Retrieval (arXiv 2602.23906)](https://arxiv.org/html/2602.23906)
- Toolkit across 12 benchmarks (ARO, CREPE, SugarCrepe, VALSE, VL-Checklist, WhatsUp, ImageCoDe, SVO, Winoground, ColorSwap, EqBen, MMVP-VLM) and 274 CLIP checkpoints: among pre-trained models, compositionality correlates positively with recognition, and "SigLIP ... exhibits superior compositional abilities"; fine-tuning for compositionality trades off recognition — [arXiv 2406.09388](https://arxiv.org/html/2406.09388)
- ColorSwap (1,000 Winoground-style colour-swap pairs): CLIP near chance (~12), SigLIP ~30, BLIP ITM ~87; fine-tuning raises CLIP group score 11.67 -> 63.00 and BLIP ITM 87.33 -> 95.33 — [ColorSwap (ACL Findings 2024)](https://aclanthology.org/2024.findings-acl.99/). Caveat: figures come from a search summary; the metric (group vs image score) for CLIP/SigLIP was not verified in the paper text.
- Baseline SigLIP 2 group scores on swap-only bidirectional sets are very low: BiSCoR controlled-colour 6.0% (SigLIP 2-Giant), Winoground-171 11.7% (SigLIP 2 B/32), BiVLC 7.8% (SigLIP 2-Giant) — [arXiv 2506.09691](https://arxiv.org/html/2506.09691)
- Hard-negative benchmark bias critique ("A Good CREPE needs more than just Sugar") — [arXiv 2506.08227](https://arxiv.org/html/2506.08227v1) (not fetched in detail)

### Inferences
- For queries like "person in yellow shirt and black trousers", SigLIP 2 will reliably retrieve frames containing a person plus yellow and black somewhere, but will not reliably distinguish "yellow shirt/black trousers" from "black shirt/yellow trousers". Retrieval with SigLIP 2 should be treated as recall; binding needs a verification stage.
- Counting ("three red hats") is a separate weakness; region/detector counts are more reliable than embeddings (see Q5).

### Gaps
- No compositional benchmark numbers found for BEiT-3 (ARO/SugarCrepe/Winoground).
- No SigLIP 2 so400m-specific SugarCrepe numbers retrieved (SigLIP 2 paper did not surface compositional tables in search).

## Q5. Fixes: region-level verification, open-vocab detectors with attributes, pixel colour classification, VLM question checks

### Takeaway
Forcing local (crop-level) matching fixes colour binding dramatically in controlled tests (SigLIP 2-Giant colour group score 6.0% -> 95.7%) but is expensive if done with exhaustive crops; open-vocab detectors are weak at attributes (OVAD colour 41–53%), so the pragmatic path is: use existing boxes (person etc.) -> crop sub-regions -> score attribute phrases with SigLIP 2 or classify colour from pixels, and reserve VLM yes/no checks for the final top-10–20.

### Cited Findings
- Inference-time structuring (training-free): ~270 overlapping crops per image (32x32 to 224x224), caption split into object-attribute and relation segments, each segment matched to its best crop, similarities averaged. Group-score gains: BiSCoR controlled colour 6.0 -> 95.7 (SigLIP 2-Giant, +89.7), BiVLC 7.8 -> 29.8, Winoground-171 11.7 -> 17.0 (SigLIP 2 B/32); improves group score in 8 of 9 dataset-model comparisons — [arXiv 2506.09691](https://arxiv.org/html/2506.09691), [abs](https://arxiv.org/abs/2506.09691)
- Cost of the same method on A100: CLIP 5 s -> 8 min, SigLIP 2-Giant 35 s -> 2 h for BiVLC/Winoground evaluation; limitations: "59% of text segments are wrong" when LLM-generated from natural captions; Size/Quantity attributes gain little vs Color/Material — [arXiv 2506.09691](https://arxiv.org/html/2506.09691)
- ABE-CLIP: training-free attribute-binding enhancement for compositional image-text matching — [arXiv 2512.17178](https://arxiv.org/html/2512.17178) (not fetched in detail)
- OVAD attribute detection: OWLv2 B/16 colour 45.1% (avg 35.4%); OWLv2 L/14 colour 53.3% (avg 37.3%); Grounding DINO colour 41.0% (avg 33.1%); colour/material (high-frequency attributes) are easier than pattern/transparency — [HA-FGOVD (arXiv 2409.16136)](https://arxiv.org/pdf/2409.16136)
- Grounding DINO "struggles with fine-grained detection because category tokens tend to dominate the region-text alignment", weakening attribute semantics — [DSAA (arXiv 2605.18023)](https://arxiv.org/pdf/2605.18023) (as summarised in search)
- Fine-grained OVOD evaluation (CVPR 2024) found colour and other attribute variations challenging for open-vocab detectors — [Bianchi et al., CVPR 2024 PDF](https://iris.cnr.it/bitstream/20.500.14243/468224/1/2024-CVPR-fine-grained-OVOD.pdf) (fetch returned 403; claim from search snippet only)
- Cross-attention ITM is strong on colour binding (BLIP ITM ~87 on ColorSwap vs SigLIP ~30), making BLIP/BLIP-2 ITM a cheaper verifier than a full VLM for colour-swap cases — [ColorSwap](https://aclanthology.org/2024.findings-acl.99/)
- VLM colour perception has its own benchmark (ColorBench) indicating colour understanding in VLMs is imperfect — [ColorBench (arXiv 2504.10514)](https://arxiv.org/html/2504.10514) (not fetched in detail)

### Inferences
- A cheap, CPU-friendly verifier for this project (design proposal, not from sources): parse the query into (object, attribute) pairs with a small rule set/LLM; for top-K (e.g. 200) SigLIP 2 candidates, take existing "person"/"hat" boxes, split person boxes into upper/lower halves (shirt vs trousers), compute (a) SigLIP 2 image embedding of each crop vs short phrases "yellow shirt", "black trousers" and/or (b) dominant colour via HSV/Lab histograms mapped to ~11 basic colour names; count boxes satisfying the constraint for "three red hats". Crop embeddings could be precomputed offline on T4 for all boxes of target classes (storage: n_boxes x 1152 dims fp16 for so400m).
- Pixel colour naming is fast and interpretable but sensitive to lighting, white balance and video compression; combine with SigLIP crop scores rather than use as a hard filter.
- The 270-crop exhaustive method is too slow for 335k frames; using detector boxes (~a few crops per frame) instead of 270 blind crops is the practical adaptation and matches the paper's finding that local regions carry binding information.
- VLM yes/no checks ("Is the person wearing a yellow shirt and black trousers? yes/no", P(yes) as in Qwen3-VL-Reranker/LamRA pointwise) are the most accurate general option but should be limited to top-10–20 on GPU.

### Gaps
- No reported accuracy for "crop person box + SigLIP colour phrase" on a standard benchmark was found.
- No reported accuracy of HSV/Lab colour naming for clothing in broadcast video found.
- No latency figures for VLM yes/no verification on T4 found.
