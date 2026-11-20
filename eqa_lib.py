"""eqa_lib – shared code for the DS5007 Assignment 2 notebooks (Extractive QA on SQuAD 1.1).

Every notebook imports this module so that all approaches use exactly the same data loading, preprocessing,
post-processing and metrics (a requirement for a fair comparison). The notebooks display the source of each
function next to its explanation.

Contents
    1. Setup ............ seeds, device, mixed precision, project paths
    2. Data ............. loading SQuAD 1.1, cleaning, EDA helpers
    3. Preprocessing .... tokenisation, offset-based label alignment, truncation / sliding windows
    4. Batching ......... Dataset, dynamic-padding collate, length-grouped sampler
    5. Metrics .......... official SQuAD v1.1 Exact Match and F1
    6. Post-processing .. best span search over windows, answer extraction by character offsets
    7. Utilities ........ saving figures/JSON, showing source code
"""
# Initialise OpenSSL before torch/pyarrow: on some Windows machines importing torch and then pyarrow first makes
# the first HTTPS request crash the process (DLL load-order clash). Harmless everywhere else.
import ssl

_SSL_CONTEXT = ssl.create_default_context()

import collections
import contextlib
import inspect
import json
import math
import os
import random
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

# =============================================================================================================
# 1. Setup
# =============================================================================================================
SEED = 42
TOKENIZER_NAME = "bert-base-uncased"   # uncased WordPiece, identical vocabulary to distilbert-base-uncased
CFG = {
    "max_length": 384,          # tokens per input sequence / window
    "doc_stride": 128,          # overlap between consecutive windows
    "max_question_tokens": 64,  # longer questions are cut (keeps room for context at short max_length)
    "max_answer_length": 30,    # longest span (tokens) considered at inference
    "n_best": 20,               # start/end candidates searched per window
    "eval_batch_size": 64,
}
QUICK_RUN = os.environ.get("EQA_QUICK_RUN", "0") == "1"        # tiny smoke-test run
FORCE_RETRAIN = os.environ.get("EQA_FORCE_RETRAIN", "0") == "1"  # ignore cached runs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    AMP_DTYPE = None


class Paths:
    """Project folders (created on first use)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        base = self.root / "_smoke_test" if QUICK_RUN else self.root
        self.models = base / "models"
        self.results = base / "results"
        self.graphs = self.results / "graphs"
        self.logs = self.results / "logs"
        for d in (self.models, self.results, self.graphs, self.logs):
            d.mkdir(parents=True, exist_ok=True)


PATHS: Paths | None = None
_TOKENIZER = None


def find_project_root() -> Path:
    """The folder that contains eqa_lib.py."""
    return Path(__file__).resolve().parent


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all GPUs) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup(verbose: bool = True) -> dict:
    """Seed everything, create the project folders and report the compute environment."""
    global PATHS
    set_seed(SEED)
    PATHS = Paths(find_project_root())
    import transformers
    transformers.logging.set_verbosity_error()
    try:
        transformers.utils.logging.disable_progress_bar()
    except Exception:
        pass
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    info = {
        "python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1) if DEVICE.type == "cuda" else None,
        "cuda": torch.version.cuda, "amp_dtype": str(AMP_DTYPE), "seed": SEED,
        "quick_run": QUICK_RUN, "force_retrain": FORCE_RETRAIN, "project_root": str(PATHS.root),
    }
    if verbose:
        for k, v in info.items():
            print(f"{k:<15}: {v}")
    return info


def autocast_ctx():
    """Mixed-precision context for forward passes (bf16 on Ampere/Ada GPUs, fp16 otherwise, no-op on CPU)."""
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return contextlib.nullcontext()


def get_tokenizer():
    """The shared WordPiece tokenizer (loaded once)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _TOKENIZER


# =============================================================================================================
# 2. Data
# =============================================================================================================
def load_squad(quick: bool = QUICK_RUN, verbose: bool = True):
    """Load SQuAD 1.1 (train, validation) as DataFrames with list columns `answer_texts` / `answer_starts`.

    Uses Hugging Face Datasets; if that package cannot be imported (on the development PC a Windows Application
    Control policy blocks one of its compiled dependencies) the identical parquet files that `load_dataset` reads
    are downloaded from the Hugging Face Hub instead.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad")
        train_df, val_df = ds["train"].to_pandas(), ds["validation"].to_pandas()
        source = "datasets.load_dataset('rajpurkar/squad')"
    except Exception as exc:
        if verbose:
            print(f"[info] `datasets` not usable here ({type(exc).__name__}: {str(exc)[:80]}...)")
            print("[info] Falling back to the identical parquet files on the Hugging Face Hub.")
        from huggingface_hub import hf_hub_download
        frames = {}
        for split in ("train", "validation"):
            path = hf_hub_download("rajpurkar/squad", f"plain_text/{split}-00000-of-00001.parquet", repo_type="dataset")
            frames[split] = pd.read_parquet(path)
        train_df, val_df = frames["train"], frames["validation"]
        source = "huggingface_hub: rajpurkar/squad (plain_text parquet)"
    for df in (train_df, val_df):
        df["answer_texts"] = [[str(t) for t in a["text"]] for a in df["answers"]]
        df["answer_starts"] = [[int(s) for s in a["answer_start"]] for a in df["answers"]]
        df.drop(columns="answers", inplace=True)
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    if quick:
        train_df = train_df.sample(800, random_state=SEED).reset_index(drop=True)
        val_df = val_df.sample(300, random_state=SEED).reset_index(drop=True)
    if verbose:
        print(f"Loaded from {source}: {len(train_df):,} training / {len(val_df):,} validation examples")
    return train_df, val_df, source


def fix_answer_start(context: str, text: str, start: int):
    """Return (start, status). Repairs an answer_start that does not point at the answer text."""
    if context[start:start + len(text)] == text:
        return start, "ok"
    occurrences = [m.start() for m in re.finditer(re.escape(text), context)]
    if occurrences:
        return min(occurrences, key=lambda o: abs(o - start)), "repaired"
    return None, "invalid"


def clean_training_data(train_df: pd.DataFrame):
    """Check every training answer offset; repair or drop inconsistent ones. Returns (clean_df, status counts)."""
    status, starts, keep = collections.Counter(), [], []
    for ctx, texts, st in zip(train_df["context"], train_df["answer_texts"], train_df["answer_starts"]):
        s, flag = fix_answer_start(ctx, texts[0], st[0])
        status[flag] += 1
        keep.append(s is not None and texts[0].strip() != "")
        starts.append([s if s is not None else -1])
    out = train_df.assign(answer_starts=starts)[keep].reset_index(drop=True)
    return out, dict(status)


def question_type(q: str) -> str:
    """First wh-word of a question ('how many' / 'how much' kept separate)."""
    ql = q.lower()
    for phrase in ("how many", "how much"):
        if phrase in ql:
            return phrase
    for w in re.findall(r"[a-z]+", ql):
        if w in ("what", "which", "who", "whom", "whose", "when", "where", "why", "how"):
            return {"whom": "who", "whose": "who"}.get(w, w)
    return "other"


def add_eda_columns(df: pd.DataFrame, tokenizer=None) -> pd.DataFrame:
    """Length (words/tokens), answer position and question type columns used by the EDA and error analysis."""
    tokenizer = tokenizer or get_tokenizer()
    df = df.copy()
    df["context_words"] = df["context"].str.split().str.len()
    df["question_words"] = df["question"].str.split().str.len()
    df["answer_words"] = [len(a[0].split()) for a in df["answer_texts"]]
    df["answer_start_char"] = [s[0] for s in df["answer_starts"]]
    df["answer_rel_pos"] = [s[0] / max(1, len(c)) for s, c in zip(df["answer_starts"], df["context"])]
    df["qtype"] = df["question"].map(question_type)
    uniq = df["context"].unique().tolist()
    ctx_len = dict(zip(uniq, map(len, tokenizer(uniq, add_special_tokens=False)["input_ids"])))
    df["context_tokens"] = df["context"].map(ctx_len)
    df["question_tokens"] = [len(x) for x in tokenizer(df["question"].tolist(), add_special_tokens=False)["input_ids"]]
    df["pair_tokens"] = df["context_tokens"] + np.minimum(df["question_tokens"], CFG["max_question_tokens"]) + 3
    # token position of the answer start inside the context (needed to know if truncation removes it)
    prefix_tok = tokenizer([c[:s] for c, s in zip(df["context"], df["answer_start_char"])], add_special_tokens=False)["input_ids"]
    df["answer_start_token_in_context"] = [len(p) for p in prefix_tok]
    return df


# =============================================================================================================
# 3. Preprocessing
# =============================================================================================================
def truncate_questions(questions, tokenizer, max_tokens):
    """Strip questions and cut the rare very long ones to `max_tokens` WordPiece tokens."""
    questions = [q.strip() for q in questions]
    enc = tokenizer(questions, add_special_tokens=False, return_offsets_mapping=True)
    out, n_cut = [], 0
    for q, offs in zip(questions, enc["offset_mapping"]):
        if len(offs) > max_tokens:
            q, n_cut = q[:offs[max_tokens - 1][1]], n_cut + 1
        out.append(q)
    return out, n_cut


def prepare_features(df, tokenizer=None, max_length=384, doc_stride=128, sliding_window=True,
                     keep_offsets=False, chunk_size=2000):
    """Tokenise (question, context) pairs and create token-level start/end labels.

    sliding_window=True  -> long contexts become several overlapping windows (`doc_stride` tokens of overlap)
    sliding_window=False -> plain truncation: only the first `max_length` tokens are kept (one feature per example)

    Label rule for each feature: if the gold answer [start_char, end_char) lies completely inside the feature's
    context tokens, start = last token starting at/before start_char and end = first token ending at/after
    end_char; otherwise both labels are 0 (the [CLS] token = "answer not in this window").
    Returns (feats: dict of per-feature lists, stats: alignment diagnostics).
    """
    tokenizer = tokenizer or get_tokenizer()
    questions, n_cut = truncate_questions(df["question"].tolist(), tokenizer, CFG["max_question_tokens"])
    contexts = df["context"].tolist()
    ans_texts = [a[0] for a in df["answer_texts"]]
    ans_starts = [s[0] for s in df["answer_starts"]]

    keys = ["input_ids", "token_type_ids", "start_positions", "end_positions", "example_idx", "window_idx"]
    feats = {k: [] for k in keys + (["offset_mapping"] if keep_offsets else [])}
    stats = collections.Counter(questions_truncated=n_cut)
    examples_with_answer, mismatches = set(), []

    for c0 in range(0, len(df), chunk_size):
        enc = tokenizer(questions[c0:c0 + chunk_size], contexts[c0:c0 + chunk_size],
                        truncation="only_second",                 # never cut the question, only the context
                        max_length=max_length,
                        stride=doc_stride if sliding_window else 0,
                        return_overflowing_tokens=sliding_window,  # several windows per long context
                        return_offsets_mapping=True,               # char span of every token
                        return_token_type_ids=True,
                        padding=False)                             # dynamic padding later, per batch
        sample_map = enc["overflow_to_sample_mapping"] if sliding_window else list(range(len(enc["input_ids"])))
        prev_ex, window = -1, 0
        for i, input_ids in enumerate(enc["input_ids"]):
            ex = c0 + sample_map[i]
            window = window + 1 if ex == prev_ex else 0
            prev_ex = ex
            seq_ids, offsets = enc.sequence_ids(i), enc["offset_mapping"][i]
            ctx_first = seq_ids.index(1)
            ctx_last = len(seq_ids) - 1 - seq_ids[::-1].index(1)

            start_char = ans_starts[ex]
            end_char = start_char + len(ans_texts[ex])
            if start_char < 0 or offsets[ctx_first][0] > start_char or offsets[ctx_last][1] < end_char:
                start_pos = end_pos = 0
                stats["windows_without_answer"] += 1
            else:
                start_pos = ctx_first
                while start_pos <= ctx_last and offsets[start_pos][0] <= start_char:
                    start_pos += 1
                start_pos -= 1
                end_pos = ctx_last
                while end_pos >= ctx_first and offsets[end_pos][1] >= end_char:
                    end_pos -= 1
                end_pos += 1
                stats["windows_with_answer"] += 1
                examples_with_answer.add(ex)
                recovered = contexts[ex][offsets[start_pos][0]:offsets[end_pos][1]]
                if recovered == ans_texts[ex]:
                    stats["span_exact_match"] += 1
                elif normalize_answer(recovered) == normalize_answer(ans_texts[ex]):
                    stats["span_normalized_match"] += 1
                else:
                    stats["span_mismatch"] += 1
                    if len(mismatches) < 8:
                        mismatches.append((ans_texts[ex], recovered))

            feats["input_ids"].append(input_ids)
            feats["token_type_ids"].append(enc["token_type_ids"][i])
            feats["start_positions"].append(start_pos)
            feats["end_positions"].append(end_pos)
            feats["example_idx"].append(ex)
            feats["window_idx"].append(window)
            if keep_offsets:
                feats["offset_mapping"].append([tuple(o) if seq_ids[k] == 1 else None for k, o in enumerate(offsets)])

    stats["examples"] = len(df)
    stats["features"] = len(feats["input_ids"])
    stats["examples_with_answer_in_some_window"] = len(examples_with_answer)
    stats["examples_whose_answer_is_unreachable"] = len(df) - len(examples_with_answer)
    stats = dict(stats)
    stats["mismatch_examples"] = mismatches
    return feats, stats


def select_features(feats: dict, indices) -> dict:
    """Sub-select features (e.g. only the first window of each example = truncation)."""
    return {k: [v[i] for i in indices] for k, v in feats.items()}


def first_windows(feats: dict) -> dict:
    """Keep only window 0 of every example: turns sliding-window features into truncated features."""
    return select_features(feats, [i for i, w in enumerate(feats["window_idx"]) if w == 0])


# =============================================================================================================
# 4. Batching
# =============================================================================================================
class QAFeatureDataset(Dataset):
    """Wraps the feature dict; `indices` selects a subset / order of features."""

    def __init__(self, feats, indices=None):
        self.feats = feats
        self.indices = list(range(len(feats["input_ids"]))) if indices is None else [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        f = self.indices[i]
        return {"feature_idx": f, "input_ids": self.feats["input_ids"][f], "token_type_ids": self.feats["token_type_ids"][f],
                "start_positions": self.feats["start_positions"][f], "end_positions": self.feats["end_positions"][f]}


def collate(batch, pad_id: int = 0):
    """Dynamic padding: pad to the longest sequence in the batch and build the attention mask (1 = token, 0 = pad)."""
    B, L = len(batch), max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    token_type_ids = torch.zeros((B, L), dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        token_type_ids[i, :n] = torch.tensor(b["token_type_ids"])
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask,
            "start_positions": torch.tensor([b["start_positions"] for b in batch]),
            "end_positions": torch.tensor([b["end_positions"] for b in batch]),
            "feature_idx": torch.tensor([b["feature_idx"] for b in batch])}


class LengthGroupedBatchSampler(Sampler):
    """Random batches of similar-length sequences: shuffle -> chunks of 50 batches -> sort chunk by length ->
    cut into batches -> shuffle batches. Less padding, same optimisation problem."""

    def __init__(self, lengths, batch_size, chunk_batches=50, seed=SEED):
        self.lengths, self.bs, self.seed, self.epoch = np.asarray(lengths), batch_size, seed, 0
        self.chunk = batch_size * chunk_batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        perm = rng.permutation(len(self.lengths))
        batches = []
        for c in range(0, len(perm), self.chunk):
            chunk = perm[c:c + self.chunk]
            chunk = chunk[np.argsort(-self.lengths[chunk], kind="stable")]
            batches += [chunk[i:i + self.bs].tolist() for i in range(0, len(chunk), self.bs)]
        for b in rng.permutation(len(batches)):
            yield batches[b]

    def __len__(self):
        full, rem = divmod(len(self.lengths), self.chunk)
        return full * math.ceil(self.chunk / self.bs) + math.ceil(rem / self.bs)


# =============================================================================================================
# 5. Metrics (re-implementation of the official SQuAD v1.1 evaluate-v1.1.py)
# =============================================================================================================
def normalize_answer(s: str) -> str:
    """Lower-case, remove punctuation, remove articles (a/an/the), collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens, gold_tokens = normalize_answer(prediction).split(), normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision, recall = num_same / len(pred_tokens), num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def squad_evaluate(predictions: dict, df: pd.DataFrame):
    """Mean over questions of max-over-references EM and F1 (in %). Also returns per-question arrays."""
    ems, f1s = [], []
    for qid, golds in zip(df["id"], df["answer_texts"]):
        pred = predictions.get(qid, "")
        ems.append(metric_max_over_ground_truths(exact_match_score, pred, golds))
        f1s.append(metric_max_over_ground_truths(f1_score, pred, golds))
    ems, f1s = np.array(ems), np.array(f1s)
    return {"exact_match": 100 * ems.mean(), "f1": 100 * f1s.mean()}, ems, f1s


# =============================================================================================================
# 6. Post-processing: logits -> answer text
# =============================================================================================================
def masked_softmax(x, mask):
    z = np.where(mask, x, -np.inf)
    z = np.exp(z - z[mask].max())
    return z / z.sum()


def postprocess_predictions(df, feats, start_logits, end_logits, n_best=None, max_answer_len=None):
    """Best valid span per example over all its windows; the answer is sliced from the ORIGINAL context.

    Valid span: start <= end, length <= max_answer_len tokens, both tokens inside the context (offset not None).
    Score = start_logit + end_logit. Confidence = P_start * P_end (softmax over the winning window's context tokens).
    """
    n_best = n_best or CFG["n_best"]
    max_answer_len = max_answer_len or CFG["max_answer_length"]
    by_example = collections.defaultdict(list)
    for f, ex in enumerate(feats["example_idx"]):
        by_example[ex].append(f)
    ids, contexts = df["id"].tolist(), df["context"].tolist()
    predictions, details = {}, []
    for ex in range(len(df)):
        best = None
        for f in by_example.get(ex, []):
            sl, el, offs = start_logits[f], end_logits[f], feats["offset_mapping"][f]
            for s in np.argsort(sl)[::-1][:n_best]:
                if offs[s] is None:
                    continue
                for e in np.argsort(el)[::-1][:n_best]:
                    if offs[e] is None or e < s or e - s + 1 > max_answer_len:
                        continue
                    score = sl[s] + el[e]
                    if best is None or score > best["score"]:
                        best = {"score": float(score), "feature": f, "start_token": int(s), "end_token": int(e)}
        rec = {"id": ids[ex], "example_idx": ex}
        if best is None:
            rec.update(prediction="", score=float("nan"), confidence=0.0, start_char=-1, end_char=-1,
                       start_token=-1, end_token=-1, window=-1)
        else:
            f, offs = best["feature"], feats["offset_mapping"][best["feature"]]
            ctx_mask = np.array([o is not None for o in offs])
            p_s = masked_softmax(np.asarray(start_logits[f], dtype=np.float64), ctx_mask)[best["start_token"]]
            p_e = masked_softmax(np.asarray(end_logits[f], dtype=np.float64), ctx_mask)[best["end_token"]]
            sc, ec = offs[best["start_token"]][0], offs[best["end_token"]][1]
            rec.update(prediction=contexts[ex][sc:ec], score=best["score"], confidence=float(p_s * p_e),
                       start_char=sc, end_char=ec, start_token=best["start_token"], end_token=best["end_token"],
                       window=feats["window_idx"][f])
        predictions[rec["id"]] = rec["prediction"]
        details.append(rec)
    return predictions, pd.DataFrame(details)


def oracle_logits(feats, peak=10.0):
    """'Perfect model' logits that put all mass on the gold labels – used to test the pipeline end to end."""
    sl, el = [], []
    for ids, s, e in zip(feats["input_ids"], feats["start_positions"], feats["end_positions"]):
        a, b = np.full(len(ids), -peak), np.full(len(ids), -peak)
        a[s], b[e] = peak, peak
        sl.append(a); el.append(b)
    return sl, el


# =============================================================================================================
# 7. Utilities
# =============================================================================================================
def savefig(name: str, dpi: int = 150):
    """Save the current matplotlib figure to results/graphs/<name>.png and display it."""
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(PATHS.graphs / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.show()


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def show_source(*objs):
    """Display the source code of functions/classes from this module inside a notebook."""
    from IPython.display import Markdown, display
    for obj in objs:
        display(Markdown(f"```python\n{inspect.getsource(obj)}\n```"))


# =============================================================================================================
# 8. Models – shared span heads and the hand-written Transformer encoder (Approach A1)
# =============================================================================================================
import torch.nn as nn
import torch.nn.functional as F


def qa_loss(start_logits, end_logits, start_positions, end_positions, reduction="mean"):
    """L = ( CE(P_start, y_start) + CE(P_end, y_end) ) / 2."""
    return (F.cross_entropy(start_logits, start_positions, reduction=reduction)
            + F.cross_entropy(end_logits, end_positions, reduction=reduction)) / 2


class SpanHeads(nn.Module):
    """Start-position head and end-position head: two independent linear maps R^d -> R (one score per token).

    s_i = w_s . h_i + b_s,   e_j = w_e . h_j + b_e;  padding positions get -1e4 so softmax gives them ~0 probability.
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head = nn.Linear(hidden_size, 1)
        for lin in (self.start_head, self.end_head):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

    def forward(self, hidden, attention_mask, start_positions=None, end_positions=None):
        hidden = self.dropout(hidden)
        start_logits = self.start_head(hidden).squeeze(-1).float()      # (B, L)
        end_logits = self.end_head(hidden).squeeze(-1).float()          # (B, L)
        pad = attention_mask == 0
        start_logits = start_logits.masked_fill(pad, -1e4)
        end_logits = end_logits.masked_fill(pad, -1e4)
        out = {"start_logits": start_logits, "end_logits": end_logits}
        if start_positions is not None and end_positions is not None:
            out["loss"] = qa_loss(start_logits, end_logits, start_positions, end_positions)
        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention, written out explicitly.

    For each head h:  Q_h = X W_q^h,  K_h = X W_k^h,  V_h = X W_v^h
                      head_h = softmax( Q_h K_h^T / sqrt(d_k) + M ) V_h      (M = -inf on padding keys)
    Output: concat(head_1..head_H) W_o
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask, return_weights=False):
        B, L, D = x.shape

        def split_heads(t):                                   # (B, L, D) -> (B, H, L, d_k)
            return t.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        q, k, v = split_heads(self.q_proj(x)), split_heads(self.k_proj(x)), split_heads(self.v_proj(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)              # (B, H, L, L) similarity of every pair
        key_is_pad = (attention_mask == 0)[:, None, None, :]                 # broadcast over heads and queries
        scores = scores.float().masked_fill(key_is_pad, float("-inf"))
        weights = scores.softmax(dim=-1).to(v.dtype)                         # each row sums to 1
        context = self.attn_dropout(weights) @ v                            # (B, H, L, d_k) weighted sum of values
        context = context.transpose(1, 2).reshape(B, L, D)                  # concatenate the heads
        out = self.out_proj(context)
        return (out, weights) if return_weights else out


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear(d -> d_ff) -> GELU -> dropout -> Linear(d_ff -> d)."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block with two residual connections:
         x = x + Dropout(MHSA(LN(x)))
         x = x + Dropout(FFN(LN(x)))
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln_attn, self.ln_ffn = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop_attn, self.drop_ffn = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        x = x + self.drop_attn(self.attn(self.ln_attn(x), attention_mask))   # residual connection 1
        x = x + self.drop_ffn(self.ffn(self.ln_ffn(x)))                       # residual connection 2
        return x


class ScratchTransformerQA(nn.Module):
    """Approach A1: Transformer encoder trained from random initialisation + start/end span heads.

    input embedding e_i = E_tok[x_i] + E_pos[i] + E_seg[s_i]  (learned token, position and segment embeddings)
    -> LayerNorm -> dropout -> N x EncoderBlock -> final LayerNorm -> SpanHeads
    """

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, d_ff=1024, max_len=512, dropout=0.1, pad_id=0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.seg_emb = nn.Embedding(2, d_model)
        self.emb_norm, self.emb_dropout = nn.LayerNorm(d_model), nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.heads = SpanHeads(d_model, dropout)
        self.uses_token_type_ids = True
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = self.tok_emb(input_ids) + self.pos_emb(positions) + self.seg_emb(token_type_ids)
        x = self.emb_dropout(self.emb_norm(x))
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.final_norm(x)                                # contextual representations h_1..h_L

    def forward(self, input_ids, attention_mask, token_type_ids=None, start_positions=None, end_positions=None):
        hidden = self.encode(input_ids, attention_mask, token_type_ids)
        return self.heads(hidden, attention_mask, start_positions, end_positions)


def build_model(rc: dict, encoder_config=None):
    """Create the model described by a run configuration `rc` (see the notebooks)."""
    if rc["arch"] == "scratch":
        tok = get_tokenizer()
        return ScratchTransformerQA(vocab_size=len(tok), pad_id=tok.pad_token_id, **rc["scratch"])
    if rc["arch"] == "pretrained":
        return build_pretrained_model(rc, encoder_config)          # defined in section 10 (approaches A2/A3)
    raise ValueError(f"unknown architecture {rc['arch']}")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================================================
# 9. Training and evaluation
# =============================================================================================================
@torch.no_grad()
def get_logits(model, feats, batch_size=64):
    """Run the model over all features (sorted by length for speed). Returns per-feature logits and mean loss."""
    from torch.utils.data import DataLoader
    model.eval()
    lengths = np.array([len(x) for x in feats["input_ids"]])
    order = np.argsort(-lengths, kind="stable")
    loader = DataLoader(QAFeatureDataset(feats, indices=order), batch_size=batch_size, shuffle=False, collate_fn=collate)
    start_logits, end_logits, loss_sum = [None] * len(lengths), [None] * len(lengths), 0.0
    for batch in loader:
        fidx = batch.pop("feature_idx").tolist()
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        with autocast_ctx():
            out = model(**batch)
        loss_sum += out["loss"].item() * len(fidx)
        sl, el = out["start_logits"].float().cpu().numpy(), out["end_logits"].float().cpu().numpy()
        for r, f in enumerate(fidx):
            start_logits[f], end_logits[f] = sl[r, :lengths[f]], el[r, :lengths[f]]
    return start_logits, end_logits, loss_sum / len(lengths)


def evaluate_model(model, feats, df, batch_size=None):
    """Validation loss, SQuAD EM/F1, per-question details and timing for one model on one feature set."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    sl, el, loss = get_logits(model, feats, batch_size or CFG["eval_batch_size"])
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t_fwd = time.time() - t0
    preds, details = postprocess_predictions(df, feats, sl, el)
    scores, ems, f1s = squad_evaluate(preds, df)
    details["em"], details["f1"] = ems, f1s
    return {"val_loss": loss, "exact_match": scores["exact_match"], "f1": scores["f1"], "predictions": preds,
            "details": details, "eval_seconds": time.time() - t0, "forward_seconds": t_fwd,
            "features_per_sec": len(feats["input_ids"]) / t_fwd, "n_features": len(feats["input_ids"])}


def save_checkpoint(model, run_dir: Path, rc: dict, eval_record: dict):
    """Save weights + everything needed to rebuild the model offline."""
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    meta = {"run_cfg": rc, "tokenizer": TOKENIZER_NAME, "max_question_tokens": CFG["max_question_tokens"],
            "max_answer_length": CFG["max_answer_length"], "n_best": CFG["n_best"], "best_eval": eval_record}
    save_json(meta, run_dir / "run_config.json")
    get_tokenizer().save_pretrained(run_dir)
    if hasattr(model, "encoder") and hasattr(model.encoder, "config"):
        model.encoder.config.save_pretrained(run_dir)


def load_qa_model(run_dir: Path):
    """Rebuild a trained model from models/<run>/ without internet and load its best weights."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_config.json").read_text())
    rc = meta["run_cfg"]
    enc_cfg = None
    if rc["arch"] == "pretrained":
        from transformers import AutoConfig
        enc_cfg = AutoConfig.from_pretrained(run_dir)
    model = build_model(rc, encoder_config=enc_cfg)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    return model.to(DEVICE).eval()


def log_experiment(row: dict, path: Path | None = None):
    """Insert/replace one row (keyed by 'run') in results/experiment_logs.csv."""
    path = path or PATHS.results / "experiment_logs.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df) and "run" in df:
        df = df[df["run"] != row["run"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def _log(msg, lines):
    print(msg, flush=True)
    lines.append(msg)
    with open(PATHS.logs / "training_log.txt", "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def train_model(run_name, rc, train_feats, val_feats, val_df, save_ckpt=True, model_dir=None):
    """Train one configuration with AdamW + warm-up/linear decay + clipping + mixed precision, evaluating EM/F1 on
    the validation set `evals_per_epoch` times per epoch, keeping the best-F1 checkpoint and stopping early after
    `patience` evaluations without improvement. Cached: if the run exists it is reloaded instead of retrained.
    Returns (best model or None, history dict)."""
    from torch.utils.data import DataLoader
    run_dir = Path(model_dir) if model_dir else PATHS.models / run_name
    hist_path = PATHS.logs / f"history_{run_name}.json"
    if not FORCE_RETRAIN and hist_path.exists() and (not save_ckpt or (run_dir / "best_model.pt").exists()):
        history = json.loads(hist_path.read_text())
        print(f"[cache] '{run_name}' already trained – replaying its saved log (set EQA_FORCE_RETRAIN=1 to retrain).")
        for line in history["log_lines"]:
            print(line)
        return (load_qa_model(run_dir) if save_ckpt else None), history

    set_seed(SEED)
    model = build_model(rc).to(DEVICE)
    n_examples = max(train_feats["example_idx"]) + 1
    if rc.get("train_fraction", 1.0) < 1.0:     # reproducible subset of QUESTIONS (all windows of a question kept)
        rng = np.random.default_rng(SEED)
        keep = set(rng.permutation(n_examples)[: int(round(rc["train_fraction"] * n_examples))].tolist())
        feat_idx = [f for f, ex in enumerate(train_feats["example_idx"]) if ex in keep]
        n_used = len(keep)
    else:
        feat_idx, n_used = list(range(len(train_feats["input_ids"]))), n_examples
    sampler = LengthGroupedBatchSampler([len(train_feats["input_ids"][f]) for f in feat_idx], rc["batch_size"])
    loader = DataLoader(QAFeatureDataset(train_feats, indices=feat_idx), batch_sampler=sampler, collate_fn=collate,
                        pin_memory=DEVICE.type == "cuda")

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]          # biases and LayerNorm weights
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": rc["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=rc["lr"], betas=(0.9, 0.999), eps=1e-8)
    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * rc["epochs"]
    warmup_steps = int(rc["warmup_ratio"] * total_steps)

    def lr_lambda(step):           # linear warm-up, then linear decay to 0
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))
    evals_per_epoch = rc.get("evals_per_epoch", 1)
    eval_points = {round(steps_per_epoch * k / evals_per_epoch) for k in range(1, evals_per_epoch + 1)}
    log_every = max(20, steps_per_epoch // 10)

    lines = []
    history = {"run_name": run_name, "config": rc, "train_log": [], "evals": [], "log_lines": lines,
               "n_params": count_parameters(model), "n_train_examples": n_used, "n_train_features": len(feat_idx),
               "steps_per_epoch": steps_per_epoch, "n_val_features": len(val_feats["input_ids"])}
    _log(f"=== {run_name}: {history['n_params'] / 1e6:.2f}M params | {n_used:,} questions -> {len(feat_idx):,} inputs | "
         f"{rc['epochs']} epochs x {steps_per_epoch} steps (batch {rc['batch_size']}, lr {rc['lr']}) ===", lines)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_f1, bad_evals, global_step, stop, eval_time = -1.0, 0, 0, False, 0.0
    t_start = time.time()
    for epoch in range(rc["epochs"]):
        model.train()
        sampler.set_epoch(epoch)
        win_loss = win_n = period_loss = period_n = 0
        for step_in_epoch, batch in enumerate(loader, start=1):
            batch.pop("feature_idx")
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            with autocast_ctx():
                loss = model(**batch)["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), rc["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            lv = loss.item()
            win_loss += lv; win_n += 1; period_loss += lv; period_n += 1

            if global_step % log_every == 0:
                elapsed = time.time() - t_start
                eta = (elapsed - eval_time) / global_step * (total_steps - global_step)
                rec = {"step": global_step, "epoch": global_step / steps_per_epoch, "loss": win_loss / win_n,
                       "lr": scheduler.get_last_lr()[0]}
                history["train_log"].append(rec)
                _log(f"[{run_name}] epoch {epoch + 1}/{rc['epochs']} step {step_in_epoch:>5}/{steps_per_epoch} | "
                     f"train loss {rec['loss']:.4f} | lr {rec['lr']:.2e} | {elapsed / 60:5.1f} min, ETA {eta / 60:5.1f} min", lines)
                win_loss = win_n = 0

            if step_in_epoch in eval_points:
                ev = evaluate_model(model, val_feats, val_df)
                eval_time += ev["eval_seconds"]
                rec = {"step": global_step, "epoch": global_step / steps_per_epoch, "train_loss": period_loss / period_n,
                       "val_loss": ev["val_loss"], "exact_match": ev["exact_match"], "f1": ev["f1"],
                       "elapsed_min": (time.time() - t_start) / 60, "eval_seconds": ev["eval_seconds"],
                       "forward_seconds": ev["forward_seconds"]}
                history["evals"].append(rec)
                improved = ev["f1"] > best_f1
                _log(f"[{run_name}] >>> eval @ epoch {rec['epoch']:.2f}: train loss {rec['train_loss']:.4f} | "
                     f"val loss {rec['val_loss']:.4f} | EM {rec['exact_match']:.2f} | F1 {rec['f1']:.2f}"
                     f"{'  (new best)' if improved else ''}", lines)
                period_loss = period_n = 0
                if improved:
                    best_f1, bad_evals, history["best"] = ev["f1"], 0, rec
                    if save_ckpt:
                        save_checkpoint(model, run_dir, rc, rec)
                else:
                    bad_evals += 1
                    if bad_evals >= rc["patience"]:
                        _log(f"[{run_name}] early stopping: no F1 improvement for {bad_evals} evaluations", lines)
                        stop = True
                        break
                model.train()
        if stop:
            break

    total_min = (time.time() - t_start) / 60
    history.update({
        "train_time_min": total_min,                               # wall clock incl. validation passes
        "pure_train_time_min": total_min - eval_time / 60,        # optimisation steps only
        "epochs_run": global_step / steps_per_epoch,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1024 ** 3 if DEVICE.type == "cuda" else None,
        "model_size_mb": (run_dir / "best_model.pt").stat().st_size / 1e6 if save_ckpt else
                         sum(t.numel() * t.element_size() for t in model.state_dict().values()) / 1e6,
    })
    _log(f"[{run_name}] done: {total_min:.1f} min ({history['pure_train_time_min']:.1f} min training, "
         f"{eval_time / 60:.1f} min validation) | best F1 {best_f1:.2f} | peak GPU memory {history['peak_gpu_mem_gb'] or 0:.2f} GB", lines)
    save_json(history, hist_path)
    del optimizer, scheduler, loader
    model = load_qa_model(run_dir) if save_ckpt else None
    import gc
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return model, history


def plot_history(history, title=None):
    """Training vs validation loss, validation EM/F1 and learning rate of one run."""
    import matplotlib.pyplot as plt
    tl, ev = pd.DataFrame(history["train_log"]), pd.DataFrame(history["evals"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    if len(tl):
        axes[0].plot(tl["epoch"], tl["loss"], alpha=0.35, label="train loss (per log interval)")
    axes[0].plot(ev["epoch"], ev["train_loss"], "o-", label="train loss (mean since last eval)")
    axes[0].plot(ev["epoch"], ev["val_loss"], "s-", label="validation loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("(CE_start + CE_end) / 2"); axes[0].set_title("Training vs validation loss")
    axes[0].legend(fontsize=8)
    axes[1].plot(ev["epoch"], ev["exact_match"], "o-", label="EM"); axes[1].plot(ev["epoch"], ev["f1"], "s-", label="F1")
    for _, r in ev.iterrows():
        axes[1].annotate(f"{r['f1']:.1f}", (r["epoch"], r["f1"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("%"); axes[1].set_title("Validation EM / F1"); axes[1].legend()
    if len(tl):
        axes[2].plot(tl["step"], tl["lr"]); axes[2].set_xlabel("optimisation step"); axes[2].set_title("Learning rate")
    fig.suptitle(title or history["run_name"])
    return fig


def experiment_row(run_name, approach, history, ev, input_strategy, max_length, doc_stride):
    """One row for results/experiment_logs.csv (actual measured values only)."""
    best = history["best"]
    return {"run": run_name, "approach": approach, "input_strategy": input_strategy, "max_length": max_length,
            "doc_stride": doc_stride, "train_fraction": history["config"].get("train_fraction", 1.0),
            "train_questions": history["n_train_examples"], "train_inputs": history["n_train_features"],
            "params_M": history["n_params"] / 1e6, "model_size_MB": history["model_size_mb"],
            "epochs_run": history["epochs_run"], "best_epoch": best["epoch"],
            "train_time_min": history["train_time_min"], "pure_train_time_min": history["pure_train_time_min"],
            "train_min_per_epoch": history["pure_train_time_min"] / history["epochs_run"],
            "peak_train_gpu_mem_GB": history["peak_gpu_mem_gb"], "best_train_loss": best["train_loss"],
            "best_val_loss": best["val_loss"], "val_inputs": ev["n_features"],
            "inference_s_full_val": ev["forward_seconds"], "inference_inputs_per_s": ev["features_per_sec"],
            "EM": ev["exact_match"], "F1": ev["f1"], "timestamp": time.strftime("%Y-%m-%d %H:%M")}


# =============================================================================================================
# 10. Pretrained encoder + span heads (Approaches A2 and A3)
# =============================================================================================================
class TransformerQAModel(nn.Module):
    """Pretrained Transformer encoder (DistilBERT / BERT) + the SAME start/end span heads as approach A1.

    The encoder weights come from Hugging Face (`AutoModel.from_pretrained`); only the two linear heads
    (2 x (768 + 1) = 1,538 parameters) are new. Everything is fine-tuned end to end with our own training loop.
    """

    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.heads = SpanHeads(encoder.config.hidden_size, dropout)
        # DistilBERT has no segment (token type) embeddings -> only pass token_type_ids to encoders that accept them
        self.uses_token_type_ids = "token_type_ids" in inspect.signature(encoder.forward).parameters

    def forward(self, input_ids, attention_mask, token_type_ids=None, start_positions=None, end_positions=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.uses_token_type_ids and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state            # (B, L, 768) contextual representations
        return self.heads(hidden, attention_mask, start_positions, end_positions)


def build_pretrained_model(rc: dict, encoder_config=None):
    """Pretrained weights from the Hub (training) or an empty encoder from a saved config (loading a checkpoint)."""
    from transformers import AutoModel
    encoder = (AutoModel.from_config(encoder_config) if encoder_config is not None
               else AutoModel.from_pretrained(rc["encoder"]))
    return TransformerQAModel(encoder, dropout=rc.get("head_dropout", 0.1))


# =============================================================================================================
# 11. Inference on raw text
# =============================================================================================================
def predict(model, context: str, question: str, sliding_window: bool = True, max_length: int = 384, doc_stride: int = 128):
    """Answer `question` from `context` with a trained QA model.

    Runs exactly the evaluation pipeline on one example: question/context packing (+ overlapping windows if
    `sliding_window`), a forward pass, the constrained best-span search over all windows, and extraction of the
    answer from the ORIGINAL context with the character offsets of the winning tokens.

    Returns a dict with the answer text, its character span, the start/end token positions (indices into the
    winning window's `[CLS] q [SEP] c [SEP]` sequence) and their WordPiece tokens, the window used, and the
    confidence P_start(start) * P_end(end).
    """
    tok = get_tokenizer()
    df = pd.DataFrame({"id": ["q0"], "context": [context], "question": [question],
                       "answer_texts": [[""]], "answer_starts": [[-1]]})
    feats, _ = prepare_features(df, tok, max_length, doc_stride, sliding_window=sliding_window, keep_offsets=True)
    sl, el, _ = get_logits(model, feats, batch_size=16)
    _, details = postprocess_predictions(df, feats, sl, el)
    r = details.iloc[0]
    out = {"answer": r["prediction"], "start_char": int(r["start_char"]), "end_char": int(r["end_char"]),
           "start_token": int(r["start_token"]), "end_token": int(r["end_token"]), "window": int(r["window"]),
           "n_windows": len(feats["input_ids"]), "confidence": float(r["confidence"]), "answer_tokens": []}
    if out["window"] >= 0:
        f = feats["window_idx"].index(out["window"])
        out["answer_tokens"] = tok.convert_ids_to_tokens(feats["input_ids"][f][out["start_token"]:out["end_token"] + 1])
        assert context[out["start_char"]:out["end_char"]] == out["answer"]      # always an exact substring
    return out
