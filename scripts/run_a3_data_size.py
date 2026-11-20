"""Experiment B, approach A3: fine-tune DistilBERT + sliding windows on 10/25/50% of the training questions.

Exactly the call made in notebooks/05_Experiments_and_Comparison.ipynb (section B). Run in a parallel process to
save wall-clock time; the notebook reloads the cached results (history JSON) and replays the training logs.

Usage (from the project root):  python scripts/run_a3_data_size.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eqa_lib as eqa  # noqa: E402  (imported first: OpenSSL/torch load-order workaround)

A3_RC = dict(arch="pretrained", encoder="distilbert-base-uncased", input_strategy="sliding_window",
             max_length=384, doc_stride=128, lr=5e-5, batch_size=16, epochs=2, warmup_ratio=0.1, weight_decay=0.01,
             max_grad_norm=1.0, evals_per_epoch=1, patience=2, train_fraction=1.0)
FRACTIONS = [0.10, 0.25, 0.50]

if __name__ == "__main__":
    eqa.setup()
    train_raw, val_df, _ = eqa.load_squad()
    train_df, _ = eqa.clean_training_data(train_raw)
    tok = eqa.get_tokenizer()
    train_feats, _ = eqa.prepare_features(train_df, tok, 384, 128, sliding_window=True)
    val_feats, _ = eqa.prepare_features(val_df, tok, 384, 128, sliding_window=True, keep_offsets=True)
    for frac in FRACTIONS:
        eqa.train_model(f"A3_frac{int(round(frac * 100))}", dict(A3_RC, train_fraction=frac),
                        train_feats, val_feats, val_df, save_ckpt=False)
