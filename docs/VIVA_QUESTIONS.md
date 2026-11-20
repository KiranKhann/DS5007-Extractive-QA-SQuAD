# Viva Preparation: Extractive QA on SQuAD 1.1 (DS5007 Assignment 2)

Section references (NB01–NB05) point to the notebooks. All numbers are measured results from `results/`.

---

## A. Problem, data, metrics

**1. What is extractive QA, and how does it differ from generative QA?**
The model receives a question and a paragraph and returns a contiguous span of the paragraph: it predicts a start token and an end token. Generative QA (T5, LLMs) writes the answer token by token and can paraphrase or hallucinate. Extractive answers are always grounded in the text and are easy to evaluate, but the model cannot answer arithmetic, yes/no or "not in the text" questions. Our demo 6 ("how many years after…", predicted "2003") and demo 7 (an unanswerable question, answered "British" with 0.98 confidence) show this.

**2. Describe SQuAD 1.1.**
It has 87,599 training and 10,570 validation questions, written by crowd-workers on 536 Wikipedia articles. Each answer is a span (text + character offset). Validation questions have about 3.3 reference answers, and in 44% of them at least two references differ after normalisation. The test set is hidden, so the dev set is used for evaluation.

**3. What did the EDA tell you?**
Contexts average 120 words (156 tokens), answers are short (median 2 words), and 59% of answers start in the first half of the paragraph. The data is clean: no missing values and every offset is consistent. Only 1.0% of training pairs exceed 384 tokens, but 9.6% exceed 256 and 75.8% exceed 128. That motivated sliding windows and Experiments C and D.

**4. How are EM and F1 computed? Why both?**
Both first normalise the strings: lower-case, remove punctuation, remove a/an/the, collapse spaces. EM = 1 if the prediction equals a reference. F1 treats prediction and reference as bags of tokens: P = common/|pred|, R = common/|gold|, F1 = 2PR/(P+R). Both are maximised over the references and averaged over questions. EM checks exact boundaries, while F1 gives partial credit, so F1 − EM measures boundary errors. Neither is token-classification accuracy.

**5. Why might EM be 0 for a correct answer?**
Because annotators disagree on boundaries. Example 10: "the offices and board room etc. and is not open to the public" scores EM 0 but F1 0.96.

---

## B. Preprocessing

**6. Why can't you use `answer_start` as the label?**
`answer_start` is a character index, while the model predicts token indices. In the Turing example, "1936" starts at character 115 but is token 32. We map between them with the tokenizer's offset mapping.

**7. Explain your label generation.**
For each input, find the first and last context token (`sequence_ids == 1`). If the gold span lies completely inside, start = the last token whose start offset ≤ `answer_start`, and end = the first token whose end offset ≥ `answer_end`. Otherwise both labels are 0 (`[CLS]`). Verification: cutting the labelled tokens back out reproduces the gold text for 99.74% of training windows. The 226 mismatches are annotation errors with boundaries inside a word (e.g. gold "six" inside "sixth").

**8. What is the oracle test?**
We feed "perfect" logits (all mass on the gold labels) through the post-processing. With windows this gives EM 99.7, which proves the pipeline is lossless. With truncation at 128 it gives only 79.5, the ceiling of that strategy.

**9. Special tokens, token type ids, attention mask?**
- `[CLS]` starts the sequence and is used as the "no answer here" label.
- `[SEP]` separates and ends the two segments; `[PAD]` fills batches.
- Token type ids are 0 for the question and 1 for the context. A1 uses them; DistilBERT has no segment embeddings, so we don't pass them to it.
- The attention mask marks padding (0), so padding gets no attention. We also set padded logits to −10⁴.

**10. How does the sliding window work?**
`return_overflowing_tokens=True, stride=128` splits the context into 384-token windows that overlap by 128 tokens, each paired with the full question. `overflow_to_sample_mapping` links windows to questions. Windows without the complete answer are labelled `[CLS]`. At inference all windows are scored, the best valid span over all of them wins, and the offsets map it back to the original text.

**11. Why cap questions at 64 tokens and use `truncation="only_second"`?**
The question must never be cut, only the context. With `max_length` = 128, a very long question would leave no room for the context (the tokenizer requires stride < remaining space).

**12. What are dynamic padding and length-grouped batching?**
Each batch is padded only to its longest sequence, and the sampler groups similar lengths together. Tokens computed per sequence fall from 384 to about 168, with an identical loss.

---

## C. Architectures

**13. Explain self-attention.**
Q = XW_Q, K = XW_K, V = XW_V, and Attention = softmax(QKᵀ/√d_k + M)V. Each row of the softmax is a distribution over tokens, and the output is a weighted average of values. The √d_k scaling keeps scores at unit variance so the softmax doesn't saturate. M = −∞ on padding keys. Multi-head attention runs H of these in parallel and concatenates them.

**14. What exactly did you implement for A1?**
- token, learned position and segment embeddings, summed and passed through LayerNorm;
- 4 pre-LN blocks: x + MHSA(LN(x)), then x + FFN(LN(x)), with our own Q/K/V/O projections, scaling, masking, head split/merge, and a GELU FFN 256→1024→256;
- a final LayerNorm and the span heads.

The model has 11.1M parameters, 70% of them in the embedding table. Our attention matches PyTorch's scaled-dot-product attention to 1.8·10⁻⁷.

**15. Why pre-LN?**
Normalising before each sub-layer keeps gradients well-behaved through the residual path, which makes training from scratch stable with a large learning rate (5e-4).

**16. How does the encoder process a question–context pair?**
They form one sequence, so every context token attends to every question token in every layer. A token's vector encodes both its local context and how well it matches the question. No separate cross-attention is needed.

**17. What is DistilBERT, and why use it?**
It is a 6-layer student of BERT-base trained by knowledge distillation (soft-target loss + MLM + cosine loss). It keeps about 97% of BERT's accuracy with 40% fewer parameters and runs about 60% faster. That suits a 6 GB GPU and three approaches × several experiments.

**18. Describe the heads, loss and inference.**
s_i = w_s·h_i + b_s and e_j = w_e·h_j + b_e, with a softmax over positions. Loss = −½[log P_start(y_s) + log P_end(y_e)]. Inference: top-20 start × top-20 end candidates per window, i ≤ j, length ≤ 30, both inside the context; take argmax s_i + e_j over all windows. Confidence = P_start·P_end.

**19. Why is s_i + e_j the right score?**
Logits are log-probabilities up to a constant, so maximising the sum maximises P_start·P_end, assuming start and end are independent given H.

**20. Did you use a QA pipeline or `AutoModelForQuestionAnswering`?**
No. `AutoModel` only loads the bare DistilBERT encoder. Heads, loss, labels, batching, training loop, span search, metrics and `predict()` are our own code in `eqa_lib.py`.

---

## D. Training

**21. Optimiser, schedule and why?**
AdamW (weight decay 0.01, none on biases or LayerNorm), 10% linear warm-up, then linear decay, gradient clipping at 1.0, and bf16 mixed precision. Warm-up avoids large early updates that would destroy the pretrained weights.

**22. Why lr 5e-5 for DistilBERT but 5e-4 for A1?**
Fine-tuning must only nudge good pretrained weights. A randomly initialised model has nothing to preserve and needs larger steps.

**23. Checkpointing and early stopping?**
Validation EM/F1 is computed on all 10,570 questions every half epoch (A2/A3) or every epoch. We keep the best-F1 checkpoint and stop after 2 evaluations without improvement. A1 stopped after epoch 7, with its best at epoch 5.

**24. Reproducibility?**
Seed 42 everywhere, deterministic cuDNN, a seeded sampler and nested seeded subsets, configuration kept in the notebooks, and every run cached (history + checkpoint). The only residual noise is GPU kernel non-determinism, which is why A2 vs A3 is tested with a bootstrap.

---

## E. Results

**25. Main results?**

| Approach | EM | F1 |
|---|---|---|
| A1 (scratch) | 10.53 | 20.09 |
| A2 (DistilBERT, truncation) | 78.06 | 85.89 |
| A3 (DistilBERT, windows) | 77.69 | 85.67 |

The optional BERT-base reference reached 79.97 / 87.35 after only 1 of its 2 epochs.

**26. Why is A1 so poor?**
It must learn word meaning, syntax and question–context matching from 88k questions with random embeddings. It overfits after epoch 3 (training loss 4.26 → 2.70, validation loss 3.95 → 4.49). It learned answer-type shortcuts: EM 35 on *when* questions, 6 on *what*. In its last layer almost every token attends to the only year in the paragraph. Even the 2016 logistic-regression baseline with hand-made features (F1 ≈ 51) beats it.

**27. Why does pretraining help so much?**
Pretrained layers already encode lexical semantics (synonyms, paraphrases), syntax (phrase boundaries) and world knowledge. Fine-tuning only learns where answers are. Half an epoch of A2 already gives F1 80.6. The biggest gains are on *what*, *which* and *where* questions (≈ +70 F1).

**28. Is A3 better than A2?**
Not overall: A3 − A2 = −0.21 F1, and the 95% bootstrap interval [−0.66, +0.25] includes 0. On the 183 questions with contexts longer than 384 tokens, A3 wins by +10.7 F1 (interval [+5.2, +16.8]). On the 32 questions whose answer lies beyond the cut, A3 scores 83.6 F1 against A2's 14.9. Windows cost about 2% more inputs at 384 tokens.

**29. Which model do you recommend and why?**
A3. It is tied with A2 on average, much better on long inputs, and has no "unreachable answer" failure mode. It powers the demo and `app.py`.

**30. Data-size experiment?** A3 scores F1 74.4 / 80.1 / 83.6 / 85.7 with 10/25/50/100% of the data: roughly logarithmic growth with diminishing returns. A1 scores 9.7 / 14.3 / 16.5 / 20.1. A3 with 10% of the data beats A1 with all of it by 54 F1, so pretraining is worth far more than labelled data. Subsets are nested and seeded; compute is reported in optimisation steps because those runs shared the GPU.

**31. Sequence-length experiment?** With sliding windows, F1 is 85.23 / 85.74 / 85.67 at 128 / 256 / 384 tokens, almost flat, because every token is still seen and answers depend on local context. Memory drops from 2.08 GB to 1.28 GB, but time does not drop: 128-token windows need 11,647 vs 5,533 steps per epoch (17.2 vs 11.0 min/epoch measured in isolation) and 22.5 vs 16.7 s inference.

**32. Truncation experiment – which questions become difficult?** Truncation hurts exactly the questions whose answer is cut off. At 128 tokens only 79.7% of answers stay visible and F1 falls from 85.2 to 71.5; at 256 tokens the loss is 0.9, at 384 only 0.2. Per answer position, F1 is ≈85–87 before the cut and ≈6–14 after it. 1,203 questions are exact with windows but completely wrong with truncation at 128 tokens (e.g. 'South America' instead of 'Brazil'). Shortening paragraphs to 25% keeps 32% of answers (F1 34.1); surviving questions get easier (F1 89.8, fewer distractors).

**33. Computational comparison?**

| | A1 | A2 | A3 |
|---|---|---|---|
| Params | 11.1M | 66.4M | 66.4M |
| Training time | 12.3 min (7 epochs) | 18.5 min (2 epochs) | 18.5 min (2 epochs) |
| Peak GPU memory | 1.3 GB | 2.3 GB | 2.1 GB |
| Inference, full validation set | 4.8 s | 14.2 s | 13.8–14.4 s |
| Checkpoint size | 44 MB | 265 MB | 265 MB |

---

## F. Error analysis

**34. What kinds of errors does the best model make?**
For A3: 77.7% exact, 14.0% partial and 8.4% no overlap. Of the 2,358 non-exact predictions, 764 are too long, 515 too short, 195 shifted and 884 wrong spans, so most errors are about boundaries.

**35. Numbers, names, dates, multi-word answers?**
- By answer type (F1): dates 92.5, names 88.8, numbers 87.1, other phrases 81.3.
- By answer length (EM): 1 word 81.8, 7+ words 58.4.
- Typical errors:
  - Numbers lose or gain units ("103" vs "103 miles").
  - Date ranges are returned instead of the start year ("1954 to 1956").
  - Name lists are not split ("UCLA Bruins and the USC Trojans").
  - The model picks a distractor of the same type ("Francis Marion Crawford" instead of "Mark Twain").

**36. Ambiguous questions?**
F1 is 87.9 when annotators agree, 84.6 when they disagree only on boundaries, and 77.6 when they gave non-overlapping answers. Example: "How often does Doctor Who travel by himself?" has references "rarely" and "often".

**37. Hardest question types and why?**
*Why* (EM 51.6) and *how* (58.8). Their answers are long causal or explanatory phrases with fuzzy boundaries that require reasoning across sentences.

**38. Is the confidence score meaningful?**
Yes, for ranking: exact predictions have much higher confidence than wrong ones (see the reliability diagram). But it is not calibrated for unanswerable questions (demo 7 gave 0.98 on a question the text cannot answer).

**39. How would you improve the system?**
- a larger or better-pretrained encoder (BERT-base reached 87.35 F1 after one epoch; RoBERTa/DeBERTa ≈ 90+);
- an end head conditioned on the start;
- ensembles;
- SQuAD 2.0-style training with a `[CLS]` null answer so the model can abstain;
- a held-out split of training articles for model selection;
- more data or augmentation for *why/how* questions.

**40. Limitations of your study?**
- The dev set is used both for selection (≤4 checkpoints) and for reporting.
- Each configuration was trained with one seed; we use the bootstrap rather than multiple seeds.
- The from-scratch model is not tuned to its best possible size.
- The BERT-base reference was stopped after 1 epoch.
