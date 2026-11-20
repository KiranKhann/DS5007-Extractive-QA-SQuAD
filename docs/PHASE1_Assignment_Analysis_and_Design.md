# Phase 1: Assignment Analysis and Experimental Design

DS5007 NLP, Programming Assignment 2: Transformer-based Extractive QA on SQuAD 1.1
Status date: 20 Nov 2026, 17:50. **Deadline: 20 Nov 2026, midnight.**

---

## 1. Understanding of the assignment

The official PDF asks for an extractive QA system. Given a paragraph (context) and a question, it predicts the **start** and **end token** of the answer span, and the answer is copied from the original paragraph. The brief requires:

| # | Requirement (from the PDF) | Notes |
|---|---|---|
| R1 | Download and preprocess SQuAD 1.1 (a subset is allowed) | train 87,599 / validation 10,570 |
| R2 | A Transformer encoder that reads question and paragraph **together** | packed `[CLS] q [SEP] c [SEP]` |
| R3 | **Two prediction heads**: start and end position | linear heads + softmax over positions |
| R4 | Train and evaluate | own PyTorch loop, no QA pipeline |
| R5 | Explore **2–3 approaches** with explanation and **empirical motivation** | A1 scratch, A2 DistilBERT, A3 sliding windows |
| R6 | Investigate the effect of **architecture, training-data size, input sequence length** | Experiments A, B, C |
| R7 | Metrics: **EM** and **token-level F1** | official SQuAD v1.1 normalisation |
| R8 | Compare approaches | Experiment A |
| R9 | Effect of **shortening/truncating paragraphs** | Experiment D |
| R10 | Error analysis: **numbers, names, dates, multi-word answers**; **≥10 incorrect examples** with reasons | Phase 7 |
| R11 | Compare **accuracy, training time, computational requirements** | Experiment A table |
| R12 | Tables and graphs; say which approach is best and why | Phases 8–9 |
| D1 | Source code (executable notebook/program) | 5 notebooks + shared module |
| D2 | Trained models + instructions to load and test them | `models/` + README + `predict()` |
| D3 | Results: tables/graphs of EM, F1, training loss, validation performance | `results/` |
| D4 | Model comparison: strengths, limitations, compute cost | NB05 + report |
| D5 | Error analysis ≥ 10 examples | `results/error_analysis.csv` + report |
| D6 | Technical report, 4–6 pages | `report/DS5007_NLP_Technical_Report.pdf` |

## 2. Requirement-to-deliverable mapping

| Requirement | Where it is implemented / reported |
|---|---|
| R1 data loading, EDA, preprocessing | `01_Data_Exploration.ipynb`, `eqa_lib.py` (data + features) |
| R2–R3 encoder + two heads (scratch) | `02_Transformer_From_Scratch.ipynb` |
| R2–R3 encoder + two heads (pretrained) | `03_DistilBERT_Fine_Tuning.ipynb` |
| A3 overlapping windows | `04_DistilBERT_Sliding_Windows.ipynb` |
| R4 training loop, AdamW, scheduler, clipping, checkpoints, early stopping | `eqa_lib.py` (`train_model`), used by NB02–NB05 |
| R6/R8/R11 architecture comparison (Exp A) | `05_Experiments_and_Comparison.ipynb` §A |
| R6 data size (Exp B) | NB05 §B |
| R6 sequence length (Exp C) | NB05 §C |
| R9 truncation (Exp D) | NB05 §D |
| R7 EM/F1 | `eqa_lib.py` (`squad_evaluate`), unit-tested in NB01 |
| R10 error analysis | NB05 §E → `results/error_analysis.csv` |
| R12 graphs/tables | `results/graphs/`, `results/evaluation_results.csv`, `results/experiment_logs.csv` |
| Inference function + 5 demos | NB04 (best model) |
| Gradio demo (optional, last) | NB04 last section + `app.py` |
| Report | `report/DS5007_NLP_Technical_Report.pdf` (+ .docx source) |

**Code organisation.** All five notebooks import one shared, documented module, `eqa_lib.py` (data loading, feature preparation, models, training loop, post-processing, metrics). This keeps one source of truth, so notebooks cannot drift apart. Each notebook prints the source of the functions it introduces next to the explanation, so the code is still visible and explainable. On Colab you upload the project folder (or mount Drive) and run the notebooks in order 01 → 05.

## 3. Proposed architectures

All three approaches share the **same tokenizer** (uncased WordPiece, 30,522 tokens; DistilBERT and BERT-uncased use the identical vocabulary, and we take the `bert-base-uncased` tokenizer because it also returns the segment ids the scratch model needs), the **same span heads and loss**, the **same post-processing** and the **same metrics**. Differences in results are therefore due to the encoder or the input handling only.

**Shared span heads and loss.**
$s_i = w_s^\top h_i + b_s$, $e_j = w_e^\top h_j + b_e$, $P_{start} = \operatorname{softmax}(s)$, $P_{end} = \operatorname{softmax}(e)$ (padding masked),
$\mathcal{L} = -\tfrac12[\log P_{start}(y_s) + \log P_{end}(y_e)]$.
Inference: best $s_i + e_j$ over valid spans ($i \le j$, ≤ 30 tokens, inside the context), mapped back to characters via the offset mapping. Confidence $= P_{start}(\hat i)\,P_{end}(\hat j)$.

### A1: Small Transformer trained from scratch (implemented by hand)
```
token emb (30,522×256) + learned position emb (512×256) + segment emb (2×256) → LayerNorm → dropout
→ 4 × [ LN → Multi-Head Self-Attention (4 heads, d_k = 64, own Q/K/V/O projections, padding mask)
         → residual → LN → FFN 256→1024→256 (GELU) → residual ]          (pre-LN encoder)
→ final LN → start head, end head
```
* Everything is written by hand in PyTorch (`nn.Linear`/`nn.Embedding`/`nn.LayerNorm` only): scaled dot-product attention $\operatorname{softmax}(QK^\top/\sqrt{d_k} + M)V$, head split and merge, residuals, FFN. `nn.TransformerEncoder` is not used.
* About 11M parameters, ~70% of them in the embedding table.
* *Why this size:* large enough to learn question–context matching, small enough to train for 10 epochs in minutes on a 6 GB GPU. Pre-LN is used because it trains stably from random initialisation with a large learning rate.
* *Expected limitation:* 88k questions are far too few to learn English word meanings and syntax from scratch, so a large gap to the pretrained models is expected. That gap is the empirical motivation for A2.

### A2: Fine-tuned DistilBERT (plain truncation)
* `distilbert-base-uncased` (6 layers, d = 768, 12 heads, ~66M parameters, no segment embeddings) + the same span heads (1,538 new parameters), fine-tuned end to end.
* Input: **one** sequence per example, truncated to 384 tokens. If the answer is cut off, the example is labelled `[CLS]` (it cannot be learned). At inference only the first 384 tokens are visible.
* *Motivation:* pretrained bidirectional representations (masked-LM distillation from BERT on Wikipedia + books) already encode syntax, semantics and world knowledge. Fine-tuning only adapts them. DistilBERT keeps ~97% of BERT's GLUE accuracy and runs ~2× faster, which suits a 6 GB laptop GPU.

### A3: DistilBERT + overlapping sliding windows
* Same model as A2, but long contexts are split into overlapping windows (`max_length` = 384, `doc_stride` = 128 overlap). Each window is paired with the full question, and `overflow_to_sample_mapping` links windows to examples. Windows that do not fully contain the answer are labelled `[CLS]`.
* At inference all windows are scored and the best span over windows wins. The answer is reconstructed from the **original** context through the offsets.
* *Motivation:* with truncation, any answer after the cut is unreachable. Windows make every answer reachable, at the cost of more forward passes (more windows per example).

**Extra (already run, optional):** BERT-base-uncased with sliding windows can be reported as a larger-encoder reference point. It is not one of the three required approaches.

## 4. Experimental design (controlled experiments)

Common settings unless stated: seed 42, AdamW (wd 0.01, no decay on bias/LayerNorm), 10% linear warm-up then linear decay, gradient clipping 1.0, bf16 mixed precision (fp16 + GradScaler on a Colab T4), dynamic padding with length-grouped batches, max answer length 30, n-best 20. Evaluation is on the **full validation set (10,570 questions)** after every epoch. The best-F1 checkpoint is kept, with early stopping (patience 2).

| Exp | Question | Runs | Measured |
|---|---|---|---|
| **A** Architecture | scratch vs pretrained vs windows | A1, A2, A3 (all at 384, 100% data) | EM, F1, train/val loss curves, train time, inference time, params, model size on disk, peak GPU memory |
| **B** Data size | how much labelled data is needed? | A1 and A3 at **10/25/50/100%** of training questions (seeded sampling by question; all windows of a question kept; exact counts logged) | EM, F1, train time vs #examples |
| **C** Sequence length | effect of `max_length` | A3 **trained and evaluated** at 128 / 256 / 384 (stride = L/3: 42 / 85 / 128), batch 16, 2 epochs | EM, F1, train time, peak memory, #windows, inference time |
| **D** Truncation | what is lost when context is cut? | (i) full context within the input limit: A2 at 384, which covers ~99% of contexts; (ii) shortened context: first window only at 128 / 256; (iii) sliding windows at the same lengths. Same weights in (ii)/(iii), so only the visible context differs | EM, F1, % answers still reachable, accuracy by answer position, examples of questions that fail once the answer is cut off |
| **E** Error analysis | where does the best model fail? | best model | ≥ 10 incorrect examples (context, question, gold, prediction, EM, F1, reason) for numbers, names, dates, multi-word answers, answers near the end of long paragraphs, truncation-affected and ambiguous questions; breakdown tables |

## 5. Recommended hyperparameters

| | A1 Scratch | A2 DistilBERT (trunc.) | A3 DistilBERT (windows) |
|---|---|---|---|
| Learning rate | 5e-4 | 5e-5 | 5e-5 |
| Batch size | 32 | 16 | 16 |
| Epochs | ≤ 10 (early stop, patience 2) | 2 | 2 |
| Warm-up | 10% | 10% | 10% |
| Weight decay / clip | 0.01 / 1.0 | 0.01 / 1.0 | 0.01 / 1.0 |
| Dropout | 0.1 | 0.1 (DistilBERT default) + 0.1 heads | same |
| max_length / stride | 384 / 128 (windows) | 384 / – | 384 / 128 |
| Max question tokens | 64 | 64 | 64 |

These follow the standard SQuAD recipes (DistilBERT: lr 5e-5, 2–3 epochs; BERT paper: lr 3e-5–5e-5, batch 32, 2–3 epochs). The batch size is reduced to 16 to fit 6 GB.

## 6. Expected computational requirements

Measured on this machine (RTX 4050 Laptop, 6 GB, bf16):
* BERT-base, batch 16, 384 tokens: **≈16 min per epoch** on full SQuAD with dynamic padding (current run), peak 3.8–4.4 GB.
* DistilBERT: ≈2× faster than BERT-base in our benchmark (101 vs 53 sequences/s at fixed length 384). Estimated **≈8–9 min per epoch**, ~2.5 GB.
* A1 scratch (11M parameters): ≈1–2 min per epoch (estimate), ~1.2 GB.

| Block | Estimated GPU time |
|---|---|
| A1 (≤10 epochs) | ~20 min |
| A2 (2 epochs) | ~17 min |
| A3 (2 epochs) | ~18 min |
| Exp B (A3 and A1 at 10/25/50%) | ~30 min |
| Exp C (A3 at 128 and 256) | ~25 min |
| Exp D + error analysis (inference only) | ~10 min |
| **Total** | **≈ 2 h** (Colab T4: roughly 3–4 h) |

## 7. Potential technical challenges and mitigations

| Challenge | Mitigation |
|---|---|
| Character offsets ≠ token indices | offset-mapping alignment; every label is verified by cutting the text back out (smoke test: 99.75% exact; the rest are SQuAD annotation errors such as gold "b" for "been") |
| Answers split across windows / windows without an answer | stride overlap; `[CLS]` label; best span over all windows |
| Short `max_length` (128) with long questions | questions capped at 64 tokens so stride < context budget |
| Training from scratch is unstable | pre-LN, warm-up, larger learning rate, gradient clipping, early stopping |
| 6 GB GPU memory | bf16, batch 16, dynamic padding, length-grouped batches |
| Fair comparison | same tokenizer, heads, loss, post-processing, metrics, seed and validation set for all runs |
| Dev set used for selection and reporting (test set is hidden) | only ≤ 2–10 checkpoints compared; disclosed as a limitation |
| `datasets` package blocked on this PC by a Windows Application Control policy | load the identical parquet files with `huggingface_hub` (works on Colab with `datasets`) |
| Windows crash when torch/pyarrow are imported before OpenSSL | create an SSL context first in the import cell (already fixed and tested) |
| **Deadline tonight** | all code reuses the already smoke-tested pipeline; each run is cached so notebooks re-run without retraining |

## 8. Status: completed vs not yet executed

| Item | Status |
|---|---|
| Preprocessing, label alignment, windows, training loop, metrics, post-processing | **implemented and smoke-tested** (single-notebook version, 800/300-example run) |
| BERT-base + windows, full data | **training now** (optional reference, not required by the new plan) |
| A1 with hand-written attention, A2, A3, Exp B–E, 5-notebook structure, report PDF | **not yet executed**, awaiting approval |
