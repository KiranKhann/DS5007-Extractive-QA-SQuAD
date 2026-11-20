"""Experiment B, approach A1: train the from-scratch Transformer on 10/25/50% of the training questions.

This is exactly the call made in notebooks/05_Experiments_and_Comparison.ipynb (section B). It was run in a parallel
process to save wall-clock time; the notebook then reloads the cached results (history JSON) and replays the logs.

Usage (from the project root):  python scripts/run_a1_data_size.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eqa_lib as eqa  # noqa: E402  (imported first: OpenSSL/torch load-order workaround)

A1_RC = dict(arch="scratch", scratch=dict(d_model=256, n_heads=4, n_layers=4, d_ff=1024, max_len=512, dropout=0.1),
             input_strategy="sliding_window", max_length=384, doc_stride=128,
             lr=5e-4, batch_size=32, epochs=10, warmup_ratio=0.1, weight_decay=0.01, max_grad_norm=1.0,
             evals_per_epoch=1, patience=2, train_fraction=1.0)
FRACTIONS = [0.10, 0.25, 0.50]

if __name__ == "__main__":
    eqa.setup()
    train_raw, val_df, _ = eqa.load_squad()
    train_df, _ = eqa.clean_training_data(train_raw)
    tok = eqa.get_tokenizer()
    train_feats, _ = eqa.prepare_features(train_df, tok, 384, 128, sliding_window=True)
    val_feats, _ = eqa.prepare_features(val_df, tok, 384, 128, sliding_window=True, keep_offsets=True)
    for frac in FRACTIONS:
        eqa.train_model(f"A1_frac{int(round(frac * 100))}", dict(A1_RC, train_fraction=frac),
                        train_feats, val_feats, val_df, save_ckpt=False)
