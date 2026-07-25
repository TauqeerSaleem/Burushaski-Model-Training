import argparse
import json
import os
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


def grouped_scores(rows, group_key):
    summary = []
    for group, group_rows in pd.DataFrame(rows).groupby(group_key, dropna=False):
        summary.append({
            group_key: group,
            "samples": len(group_rows),
            **score_text(group_rows["hypothesis"].tolist(), group_rows["reference"].tolist()),
        })
    return summary


def log_to_wandb(metrics, output_dir):
    if not os.getenv("WANDB_API_KEY"):
        return
    import wandb

    wandb.init(project=os.getenv("WANDB_PROJECT", "mt5_bidir_hunza_v1"), job_type="eval", reinit=True)
    wandb.log(metrics)
    wandb.save(str(output_dir / "*.csv"))
    wandb.save(str(output_dir / "*.json"))
    wandb.save(str(output_dir / "*.txt"))
    wandb.finish()


def main(args):
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    config = read_config(args.config)
    dataset = load_dataset(
        task="mt",
        split=args.split,
        use_hf=args.use_hf,
        use_supabase=args.use_supabase,
        dialects=args.dialect or config.get("target_dialects"),
    )
    dataset = keep_dialects(dataset, args.dialect or config.get("target_dialects"))
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
        rows.append({
            "direction": pair.get("direction") or pair["source_text"].split(":", 1)[0],
            "dialect": pair.get("dialect"),
            "participant_id": pair.get("participant_id"),
            "source_name": pair.get("source"),
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

    pd.DataFrame([all_scores]).to_csv(output_dir / "mt5_metrics_summary.csv", index=False)
    (output_dir / "mt5_metrics_summary.json").write_text(json.dumps(all_scores, indent=2), encoding="utf-8")
    (output_dir / "mt5_metrics_summary.txt").write_text(
        "\n".join([
            "mT5 Translation Evaluation Summary",
            f"Samples evaluated: {all_scores['samples']}",
            f"BLEU: {all_scores['bleu']}",
            f"chrF++: {all_scores['chrf++']}",
        ]),
        encoding="utf-8",
    )
    pd.DataFrame(grouped_scores(rows, "direction")).to_csv(output_dir / "mt5_metrics_by_direction.csv", index=False)
    pd.DataFrame(grouped_scores(rows, "dialect")).to_csv(output_dir / "mt5_metrics_by_dialect.csv", index=False)
    pd.DataFrame(grouped_scores(rows, "source_name")).to_csv(output_dir / "mt5_metrics_by_source.csv", index=False)
    log_to_wandb(all_scores, output_dir)
    print(all_scores)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="outputs/mt5-bsk-eng_final")
    parser.add_argument("--config", default="configs/mt5.yaml")
    parser.add_argument("--output-dir", default="outputs/mt5-bsk-eng/results")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dialect", action="append")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-supabase", action="store_true", default=False)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", default=False)
    main(parser.parse_args())
