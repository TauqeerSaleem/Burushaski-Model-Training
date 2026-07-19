import argparse
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from dotenv import load_dotenv
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data.loader import load_dataset
from train.mt5 import keep_dialects, make_translation_pairs, read_config

load_dotenv()


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def score_text(hypotheses, references):
    hyp_norm = [normalize_text(text) for text in hypotheses]
    ref_norm = [normalize_text(text) for text in references]
    return {
        "bleu": round(BLEU().corpus_score(hyp_norm, [ref_norm]).score, 2),
        "chrf++": round(CHRF(word_order=2).corpus_score(hyp_norm, [ref_norm]).score, 2),
    }


def main(args):
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    config = read_config(args.config)
    dataset = load_dataset(
        task="mt",
        split=args.split,
        use_hf=args.use_hf,
        use_supabase=False,
        use_mdc=False,
    )
    dataset = keep_dialects(dataset, config.get("target_dialects"))
    pairs = make_translation_pairs(dataset, config)
    if not pairs:
        raise ValueError("No MT evaluation pairs found.")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    model.eval()

    rows = []
    for pair in tqdm(pairs, desc="Evaluating mT5"):
        inputs = tokenizer(
            pair["source_text"],
            return_tensors="pt",
            max_length=config.get("max_source_length", 256),
            truncation=True,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=config.get("max_target_length", 256),
                num_beams=args.num_beams,
            )
        hypothesis = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        direction = pair["source_text"].split(":", 1)[0]
        rows.append({
            "direction": direction,
            "source": pair["source_text"],
            "reference": pair["target_text"],
            "hypothesis": hypothesis,
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "mt5_predictions.csv", index=False)

    all_scores = {
        "scope": "all",
        "samples": len(rows),
        **score_text([row["hypothesis"] for row in rows], [row["reference"] for row in rows]),
    }
    by_direction = []
    for direction, group in pd.DataFrame(rows).groupby("direction"):
        by_direction.append({
            "scope": direction,
            "samples": len(group),
            **score_text(group["hypothesis"].tolist(), group["reference"].tolist()),
        })

    pd.DataFrame([all_scores]).to_csv(output_dir / "mt5_metrics_summary.csv", index=False)
    pd.DataFrame(by_direction).to_csv(output_dir / "mt5_metrics_by_direction.csv", index=False)
    print(all_scores)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="outputs/mt5-bsk-eng_final")
    parser.add_argument("--config", default="configs/mt5.yaml")
    parser.add_argument("--output-dir", default="outputs/mt5-bsk-eng/results")
    parser.add_argument("--split", default="test")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", default=False)
    main(parser.parse_args())
