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
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data.loader import load_dataset
from eval_metrics import mt_scores
from train.mbart import forced_bos_token_id, prepare_tokenizer, read_config
from train.mt5 import keep_dialects, load_processed_mt_rows, make_translation_pairs

load_dotenv()


def normalize_text(text):
    text = str(text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def score_text(hypotheses, references):
    return mt_scores(
        [normalize_text(text) for text in hypotheses],
        [normalize_text(text) for text in references],
    )


def grouped_scores(rows, group_key):
    summary = []
    for group, group_rows in pd.DataFrame(rows).groupby(group_key, dropna=False):
        summary.append({
            group_key: group,
            "samples": len(group_rows),
            **score_text(group_rows["hypothesis"].tolist(), group_rows["reference"].tolist()),
        })
    return summary


def add_group_lines(lines, title, groups, label_key):
    if not groups:
        return
    lines.extend(["", title])
    for row in sorted(groups, key=lambda item: (str(item.get(label_key)), -item["chrf++"])):
        lines.append(
            f"- {row[label_key]}: {row['samples']} samples, BLEU {row['bleu']}, "
            f"chrF++ {row['chrf++']}, TER {row['ter']}"
        )


def log_to_wandb(metrics, output_dir):
    if not os.getenv("WANDB_API_KEY"):
        return
    import wandb

    wandb.init(project=os.getenv("WANDB_PROJECT", "yaraan-hunza-mbart"), job_type="eval", reinit=True)
    wandb.log(metrics)
    wandb.save(str(output_dir / "*.csv"))
    wandb.save(str(output_dir / "*.json"))
    wandb.save(str(output_dir / "*.txt"))
    wandb.finish()


def main(args):
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    config = read_config(args.config)
    pairs = None
    if not args.use_hf and not args.use_supabase:
        pairs = load_processed_mt_rows(config, args.split)
    if pairs is None:
        dataset = load_dataset(
            task="mt",
            split=args.split,
            use_hf=args.use_hf,
            use_supabase=args.use_supabase,
            dialects=args.dialect or config.get("target_dialects"),
        )
        dataset = keep_dialects(dataset, args.dialect or config.get("target_dialects"))
        pairs = make_translation_pairs(dataset, config)
    if args.max_samples:
        pairs = pairs[:args.max_samples]
    if not pairs:
        raise ValueError("No mBART evaluation pairs found. Run analysis/build_clean_mt_dataset.py first.")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(args.model_path), config)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    bos_id = forced_bos_token_id(tokenizer, config)
    model.eval()

    rows = []
    for pair in tqdm(pairs, desc="Evaluating mBART"):
        inputs = tokenizer(
            pair["source_text"],
            return_tensors="pt",
            max_length=config.get("max_source_length", 128),
            truncation=True,
        ).to(device)
        generate_kwargs = {
            "max_length": config.get("max_target_length", 128),
            "num_beams": args.num_beams,
        }
        if bos_id is not None:
            generate_kwargs["forced_bos_token_id"] = bos_id
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)
        hypothesis = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        rows.append({
            "direction": pair.get("direction") or "bsk_to_eng",
            "dialect": pair.get("dialect"),
            "source_name": pair.get("source"),
            "source": pair["source_text"],
            "reference": pair["target_text"],
            "hypothesis": hypothesis,
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "mbart_predictions.csv", index=False)

    all_scores = {
        "scope": args.split,
        "samples": len(rows),
        **score_text([row["hypothesis"] for row in rows], [row["reference"] for row in rows]),
    }
    by_direction = grouped_scores(rows, "direction")
    by_dialect = grouped_scores(rows, "dialect")
    by_source = grouped_scores(rows, "source_name")

    pd.DataFrame([all_scores]).to_csv(output_dir / "mbart_metrics_summary.csv", index=False)
    (output_dir / "mbart_metrics_summary.json").write_text(json.dumps(all_scores, indent=2), encoding="utf-8")
    lines = [
        "mBART Translation Evaluation Summary",
        f"Samples evaluated: {all_scores['samples']}",
        f"BLEU: {all_scores['bleu']}",
        f"chrF++: {all_scores['chrf++']}",
        f"TER: {all_scores['ter']}",
        f"Exact match: {all_scores['exact_match_%']}%",
        f"Empty predictions: {all_scores['empty_prediction_%']}%",
        f"Mean length ratio: {all_scores['mean_length_ratio']}",
    ]
    add_group_lines(lines, "By direction:", by_direction, "direction")
    add_group_lines(lines, "By dialect:", by_dialect, "dialect")
    add_group_lines(lines, "By data source:", by_source, "source_name")
    (output_dir / "mbart_metrics_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(by_direction).to_csv(output_dir / "mbart_metrics_by_direction.csv", index=False)
    pd.DataFrame(by_dialect).to_csv(output_dir / "mbart_metrics_by_dialect.csv", index=False)
    pd.DataFrame(by_source).to_csv(output_dir / "mbart_metrics_by_source.csv", index=False)
    log_to_wandb(all_scores, output_dir)
    print(all_scores)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="outputs/mbart-hunza-bsk-eng-clean_final")
    parser.add_argument("--config", default="configs/mbart_clean.yaml")
    parser.add_argument("--output-dir", default="outputs/mbart-hunza-bsk-eng-clean/results")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dialect", action="append")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-supabase", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", default=False)
    main(parser.parse_args())
