# Transformer-Based Extractive Question Answering on SQuAD 1.1

DS5007 Natural Language Processing (Fall 2026), Programming Assignment 2.

The system reads a **paragraph** and a **question** and predicts the **start** and **end token** of the answer span. The answer string is then sliced out of the original paragraph using the tokenizer's character offsets, so it is never generated. We implement and compare three approaches and run controlled experiments on architecture, training-data size, input sequence length and context truncation.

| Approach | Model | Input |
|---|---|---|
| **A1** | Transformer encoder written by hand and trained **from scratch** (4 layers, d=256, 11.1M params) | sliding windows 384/128 |
| **A2** | **DistilBERT** (pretrained, 66.4M params) + our span heads, fine-tuned | truncation at 384 tokens |
| **A3** | DistilBERT + our span heads, fine-tuned | **overlapping sliding windows** 384/128 |

## Results (SQuAD 1.1 validation set, 10,570 questions)

**Experiment A – approaches** (full training data, 384 tokens)

| Approach | Params | Train time | Peak GPU | Inference (val) | EM | F1 |
|---|---|---|---|---|---|---|
| A1 Transformer from scratch | 11.1M | 12.3 min (7 ep.) | 1.32 GB | 4.8 s | 10.53 | 20.09 |
| A2 DistilBERT (truncation) | 66.4M | 18.5 min (2 ep.) | 2.33 GB | 14.2 s | 78.06 | 85.89 |
| A3 DistilBERT + sliding windows | 66.4M | 18.5 min (2 ep.) | 2.08 GB | 13.8 s | 77.69 | 85.67 |
| *Reference: BERT-base + windows (1 of 2 epochs)* | 109.5M | – | – | – | 79.97 | 87.35 |

**Experiment B – training-data size** (F1)

| Training questions | A1 scratch | A3 DistilBERT + windows |
|---|---|---|
| 8,760 (10%) | 9.70 | 74.40 |
| 21,900 (25%) | 14.32 | 80.10 |
| 43,800 (50%) | 16.55 | 83.57 |
| 87,599 (100%) | 20.09 | 85.67 |

**Experiments C/D – input length and truncation** (A3 trained at each length)

| max_length | Train min/epoch (isolated) | Peak GPU | F1 sliding windows | F1 truncation | Answers reachable with truncation |
|---|---|---|---|---|---|
| 128 | 17.2 | 1.28 GB | 85.23 | 71.54 | 79.7% |
| 256 | 11.2 | 1.67 GB | 85.74 | 84.82 | 98.5% |
| 384 | 11.0 | 2.08 GB | 85.67 | 85.46 | 99.7% |

Every number comes from executing the notebooks (`results/experiment_logs.csv`, `results/evaluation_results.csv`). Hardware: NVIDIA RTX 4050 Laptop GPU (6 GB), bf16 mixed precision.

## Project structure

```
EQA_Assignment/
├── notebooks/
│   ├── 01_Data_Exploration.ipynb            # setup, SQuAD loading, EDA, preprocessing, metrics, oracle pipeline test
│   ├── 02_Transformer_From_Scratch.ipynb    # A1: hand-written Transformer encoder + span heads, training, evaluation
│   ├── 03_DistilBERT_Fine_Tuning.ipynb      # A2: pretrained DistilBERT + span heads, truncation at 384 tokens
│   ├── 04_DistilBERT_Sliding_Windows.ipynb  # A3: overlapping windows; predict() function; demos; Gradio interface
│   └── 05_Experiments_and_Comparison.ipynb  # Experiments A–D, error analysis, consolidated results
├── eqa_lib.py                               # shared library used by all notebooks (data, preprocessing, models,
│                                            #   training loop, span search, metrics, predict)
├── models/
│   ├── transformer_scratch/                 # A1 best checkpoint
│   ├── distilbert_baseline/                 # A2 best checkpoint
│   ├── distilbert_sliding_window/           # A3 best checkpoint (recommended model)
│   ├── experiments/                         # A3 trained at max_length 128 and 256 (Experiment C)
│   └── extra_bert_base_reference/           # optional BERT-base reference (1 of 2 epochs)
├── results/
│   ├── evaluation_results.csv               # EM / F1 of every evaluated configuration
│   ├── experiment_logs.csv                  # per run: params, time, memory, inference speed, losses, EM, F1
│   ├── error_analysis.csv                   # every non-exact prediction of A3 with categories and a diagnosis
│   ├── error_examples_selected.csv          # the incorrect examples discussed in the report
│   ├── exp_*.csv, A2_vs_A3_*.csv, ...        # tables of each experiment
│   ├── predictions_<run>.json               # {question id: answer} in the official SQuAD format
│   ├── graphs/                              # all figures (01_* … 05_*)
│   └── logs/                                # training_log.txt, history_<run>.json (losses, LR, evaluations)
├── report/
│   ├── DS5007_NLP_Technical_Report.pdf      # technical report
│   └── DS5007_NLP_Technical_Report.docx     # editable source of the report
├── scripts/
│   ├── run_a1_data_size.py                  # Experiment B (A1 runs), executed in parallel to save time
│   └── make_architecture_diagram.py         # Figure 1 of the report
├── docs/
│   ├── PHASE1_Assignment_Analysis_and_Design.md
│   └── VIVA_QUESTIONS.md                    # likely viva questions with answers
├── app.py                                   # Gradio demo / command-line predictor
├── requirements.txt
└── README.md
```

## Setup

**Google Colab.** Upload the whole `EQA_Assignment` folder to Google Drive (`MyDrive/EQA_Assignment`), open a notebook and select a GPU runtime (*Runtime → Change runtime type → T4 GPU*). The first code cell mounts Drive, installs missing packages and makes `eqa_lib.py` importable. Run the notebooks in order 01 → 05. On a T4 the code automatically uses fp16 with loss scaling instead of bf16.

**Local.**
```bash
python -m venv .venv
.venv\Scripts\activate                    # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick the build for your GPU/CPU
pip install -r requirements.txt
```

## Running

Open the notebooks in Jupyter/VS Code and run them in order, or execute them head-less:
```bash
cd notebooks
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output 02_Transformer_From_Scratch.ipynb 02_Transformer_From_Scratch.ipynb
```

| Environment variable | Effect |
|---|---|
| `EQA_QUICK_RUN=1` | smoke test on 800/300 examples (writes to `_smoke_test/`) |
| `EQA_FORCE_RETRAIN=1` | ignore cached runs and retrain |
| `EQA_HEADLESS=1` | build the Gradio interface in notebook 04 without launching it |

**Caching.** Every training run stores `results/logs/history_<run>.json` and its best checkpoint. When they exist, notebooks reload them and replay the saved training log instead of retraining. This keeps the notebooks re-runnable within minutes.

**Measured run times** (RTX 4050 Laptop, 2 epochs unless stated): A1 13.4 min (7 epochs, early-stopped), A2 19.7 min, A3 19.7 min (including validation passes). Experiments B–D need ≈1.5 h of GPU time (estimated from isolated throughput; some runs were executed concurrently to save wall-clock time).

## Loading and testing the trained models

**Getting the weights.** GitHub does not accept files over 100 MB, so the trained checkpoints (`best_model.pt`) are not in the git history. They are attached to the repository's release **`models-v1`** (Releases section on GitHub) as one zip per model:
- `transformer_scratch.zip`
- `distilbert_baseline.zip`
- `distilbert_sliding_window.zip`

Unzip each one into `models/` (e.g. `models/distilbert_sliding_window/best_model.pt`). The configuration and tokenizer files are already in the repository. Alternatively, re-running a notebook retrains the model.

```bash
python app.py                                            # Gradio UI with A3 at http://127.0.0.1:7860
python app.py --model-dir models/distilbert_baseline --no-windows
python app.py --context "Alan Turing was a British mathematician and computer scientist. He introduced the concept of the Turing machine in 1936." --question "In which year did Alan Turing introduce the Turing machine?"
```

```python
import eqa_lib as eqa                     # run from the project root (import it before torch)
eqa.setup()
model = eqa.load_qa_model("models/distilbert_sliding_window")     # rebuilds the model offline from the folder
eqa.predict(model, context, question)    # -> answer, start/end char, start/end token, window, confidence
```

Each model folder contains `best_model.pt` (PyTorch `state_dict`), `run_config.json` (hyper-parameters, input settings, best validation scores), the tokenizer files and, for DistilBERT, `config.json` with the encoder architecture.

## Reproducibility notes

* Fixed seed 42 for Python, NumPy, PyTorch, the batch sampler and subset sampling. cuDNN is deterministic, though GPU attention kernels can still cause tiny run-to-run differences. That is why the A2-vs-A3 comparison uses a paired bootstrap.
* Model selection uses the official dev set, because the SQuAD test set is hidden. At most 4 checkpoints per run are compared, so the optimistic bias is small.
* **Windows-specific workarounds in the code:**
  * During development a Windows *Application Control* policy blocked a compiled dependency of the `datasets` package. `eqa.load_squad()` then falls back to downloading the identical parquet files with `huggingface_hub`. Both paths give the same data, and the notebooks print which one was used.
  * Importing `torch` and then `pyarrow` before OpenSSL is initialised crashes Python at the first HTTPS request. `eqa_lib` creates an SSL context first, so always import `eqa_lib` before `torch`.

## References
Rajpurkar et al. (2016) SQuAD · Vaswani et al. (2017) Attention Is All You Need · Devlin et al. (2019) BERT · Sanh et al. (2019) DistilBERT · Loshchilov & Hutter (2019) AdamW.
