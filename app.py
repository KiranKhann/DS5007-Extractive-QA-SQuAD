"""Interactive extractive-QA demo (Gradio) or single prediction from the command line.

Uses the shared library `eqa_lib.py` (same preprocessing, model classes, span search and answer extraction as the
notebooks) and a trained checkpoint folder from `models/`.

Usage (from the project root)
    python app.py                                        # A3 DistilBERT + sliding windows, http://127.0.0.1:7860
    python app.py --model-dir models/distilbert_baseline --no-windows
    python app.py --share                                # public Gradio link (e.g. on Colab)
    python app.py --context "..." --question "..."       # one prediction printed as JSON, no web UI
"""
import eqa_lib as eqa  # imported first: initialises OpenSSL before torch/pyarrow (Windows workaround)

import argparse
import json
from pathlib import Path

EXAMPLES = [
    ["Alan Turing was a British mathematician and computer scientist. He introduced the concept of the Turing machine "
     "in 1936.", "In which year did Alan Turing introduce the Turing machine?"],
    ["Marie Curie was the first woman to win a Nobel Prize. She was born in Warsaw in 1867 and later moved to Paris, "
     "where she studied physics and mathematics. Together with her husband Pierre she discovered the elements polonium "
     "and radium.", "Where was Marie Curie born?"],
    ["The Amazon rainforest covers much of the Amazon basin of South America. The majority of the forest is contained "
     "within Brazil, with 60% of the rainforest, followed by Peru with 13% and Colombia with 10%. The river that flows "
     "through it discharges more water than any other river and empties into the Atlantic Ocean.",
     "What percentage of the rainforest is in Peru?"],
]


def build_demo(model, sliding_window, title):
    import gradio as gr

    def run(context, question):
        if not context or not context.strip() or not question or not question.strip():
            return "Please enter both a paragraph and a question.", "", "", []
        r = eqa.predict(model, context, question, sliding_window=sliding_window)
        s, e = r["start_char"], r["end_char"]
        positions = (f"start token {r['start_token']}, end token {r['end_token']} "
                     f"(window {r['window'] + 1} of {r['n_windows']}); characters [{s}, {e})")
        return r["answer"], f"{100 * r['confidence']:.1f} %", positions, [(context[:s], None), (context[s:e], "answer"), (context[e:], None)]

    with gr.Blocks() as demo:
        gr.Markdown(f"## {title}\nThe answer is always a span copied from your paragraph.")
        with gr.Row():
            with gr.Column():
                ctx_in = gr.Textbox(label="Context paragraph", lines=10, placeholder="Paste a paragraph here…")
                q_in = gr.Textbox(label="Question", placeholder="Ask a question about the paragraph…")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                ans_out = gr.Textbox(label="Extracted answer")
                conf_out = gr.Textbox(label="Confidence (P_start × P_end)")
                pos_out = gr.Textbox(label="Token / character positions")
                hl_out = gr.HighlightedText(label="Answer in context")
        btn.click(run, [ctx_in, q_in], [ans_out, conf_out, pos_out, hl_out])
        q_in.submit(run, [ctx_in, q_in], [ans_out, conf_out, pos_out, hl_out])
        gr.Examples(EXAMPLES, [ctx_in, q_in])
    return demo


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=root / "models" / "distilbert_sliding_window")
    ap.add_argument("--no-windows", action="store_true", help="use plain truncation instead of sliding windows")
    ap.add_argument("--context", help="context paragraph (command-line mode)")
    ap.add_argument("--question", help="question (command-line mode)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = ap.parse_args()

    eqa.setup(verbose=False)
    model = eqa.load_qa_model(args.model_dir)
    meta = json.loads((args.model_dir / "run_config.json").read_text())
    best = meta.get("best_eval", {})
    title = (f"Extractive QA – {args.model_dir.name} (validation EM {best.get('exact_match', float('nan')):.1f} / "
             f"F1 {best.get('f1', float('nan')):.1f})")
    if args.context and args.question:
        print(json.dumps(eqa.predict(model, args.context, args.question, sliding_window=not args.no_windows), indent=2))
        return
    build_demo(model, not args.no_windows, title).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
