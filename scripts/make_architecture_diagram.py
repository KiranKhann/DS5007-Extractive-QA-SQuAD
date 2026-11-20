"""Draws results/graphs/00_architecture_overview.png – the shared QA architecture and where A1/A2/A3 differ."""
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "results" / "graphs" / "00_architecture_overview.png"
fig, ax = plt.subplots(figsize=(10, 10.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")


def box(x, y, w, h, text, fc, fs=9.5, bold=False):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#444444", lw=1.2, zorder=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True,
            fontweight="bold" if bold else "normal", zorder=2)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333333"))


box(2.5, 10.1, 5, 0.6, "Question  +  context paragraph (raw text)", "#eeeeee", bold=True)
box(0.3, 8.8, 4.5, 0.85, "A2  plain truncation\none input: first 384 tokens", "#dbe9f6")
box(5.2, 8.8, 4.5, 0.85, "A1 / A3  overlapping sliding windows\n384 tokens, stride 128 (overflow mapping)", "#d7efd4")
box(1.5, 7.5, 7, 0.85, "WordPiece tokenizer (uncased, 30,522 tokens):  [CLS] q₁…q_m [SEP] c₁…c_k [SEP]\n"
                       "token + position (+ segment) embeddings;  attention mask;  offset mapping kept", "#f2f2f2")
box(0.3, 5.6, 4.5, 1.45, "A1  Transformer encoder from scratch\n(hand-written: multi-head self-attention,\nFFN, residuals, pre-LayerNorm)\n"
                         "4 layers · d = 256 · 4 heads · random init\n11.1 M parameters", "#ececec")
box(5.2, 5.6, 4.5, 1.45, "A2 / A3  pretrained DistilBERT encoder\n(distilled from BERT, fine-tuned end to end)\n\n"
                         "6 layers · d = 768 · 12 heads\n66.4 M parameters", "#cfe2f3")
box(1.5, 4.55, 7, 0.6, "contextual representations  h₁ … h_n", "#f2f2f2")
box(0.8, 3.25, 3.9, 0.85, "start head:  sᵢ = w_s·hᵢ + b_s\nsoftmax → P_start(i | q, c)", "#fde2c4")
box(5.3, 3.25, 3.9, 0.85, "end head:  eⱼ = w_e·hⱼ + b_e\nsoftmax → P_end(j | q, c)", "#fde2c4")
box(0.3, 1.75, 4.5, 1.0, "training loss\nL = −½ [log P_start(y_s) + log P_end(y_e)]\n(windows without the answer: y = [CLS])", "#fff2cc")
box(5.2, 1.75, 4.5, 1.0, "inference: best span over all windows\nargmax sᵢ + eⱼ,  i ≤ j,  length ≤ 30,\nboth tokens inside the context", "#fff2cc")
box(0.6, 0.35, 8.8, 0.85, "offset mapping → character span → answer = context[char_start : char_end]\n"
                        "(always an exact substring of the original paragraph)", "#d9ead3", bold=True)

for x0, x1 in ((4.2, 2.55), (5.8, 7.45)):
    arrow(x0, 10.1, x1, 9.65)
for x in (2.55, 7.45):
    arrow(x, 8.8, x, 8.35)
for x0, x1 in ((3.5, 2.55), (6.5, 7.45)):
    arrow(x0, 7.5, x1, 7.05)
for x in (2.55, 7.45):
    arrow(x, 5.6, x, 5.15)
for x0, x1 in ((4.2, 2.75), (5.8, 7.25)):
    arrow(x0, 4.55, x1, 4.1)
arrow(2.75, 3.25, 2.55, 2.75); arrow(7.25, 3.25, 7.45, 2.75)
arrow(7.45, 1.75, 6.0, 1.2)
ax.set_title("Shared extractive-QA architecture and the three approaches", fontsize=13, fontweight="bold")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print("saved", OUT)
