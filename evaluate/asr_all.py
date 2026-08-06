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

from data.audio_io import load_audio_16k
from data.loader import load_dataset
from data.normalization import normalize_burushaski_text
from eval_metrics import asr_scores
from models.asr import ASRModel
from models.registry import get_model_specs

load_dotenv()


def normalize_metric(text):
    text = normalize_burushaski_text(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def plain_metric(text):
    return " ".join(str(text or "").lower().split())


def grouped_scores(rows, group_key):
    if not rows:
        return []
    output = []
    frame = pd.DataFrame(rows)
    for (model_name, group), group_rows in frame.groupby(["model", group_key], dropna=False):
        output.append({
            "model": model_name,
            group_key: group,
            "samples": len(group_rows),
            **asr_scores(group_rows["hypothesis_normalized"].tolist(), group_rows["reference_normalized"].tolist(), prefix="normalized_"),
        })
    return output


def log_to_wandb(metrics, output_dir):
    if not os.getenv("WANDB_API_KEY"):
        return
    import wandb

    wandb.init(project=os.getenv("WANDB_PROJECT", "yaraan-hunza-xlsr-mt5"), job_type="asr_eval", reinit=True)
    for row in metrics:
        model = row["model"]
        wandb.log({f"{model}/{key}": value for key, value in row.items() if key != "model"})
    wandb.save(str(output_dir / "*.csv"))
    wandb.save(str(output_dir / "*.json"))
    wandb.save(str(output_dir / "*.txt"))
    wandb.finish()


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    specs = get_model_specs("asr", args.model, args.registry)
    dataset = load_dataset("asr", args.split, args.use_hf, args.use_supabase, args.dialect)
    if args.max_samples:
        dataset = dataset[:args.max_samples]
    print(f"ASR eval rows loaded: {len(dataset)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    failed_rows = []
    summary_rows = []

    for model_name, spec in specs.items():
        print(f"Evaluating ASR model: {model_name}")
        model = ASRModel(model_name, spec, device)
        rows = []
        for row in tqdm(dataset, desc=model_name):
            try:
                audio = load_audio_16k(row["audio"])
                hypothesis = model.transcribe(audio)
                reference = row["transcript"]
                item = {
                    "model": model_name,
                    "filename": row.get("filename") or row.get("id"),
                    "dialect": row.get("dialect"),
                    "participant_id": row.get("participant_id"),
                    "gender": row.get("gender"),
                    "source": row.get("source"),
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "reference_raw_metric": plain_metric(reference),
                    "hypothesis_raw_metric": plain_metric(hypothesis),
                    "reference_normalized": normalize_metric(reference),
                    "hypothesis_normalized": normalize_metric(hypothesis),
                }
                rows.append(item)
                prediction_rows.append(item)
            except Exception as exc:
                failed_rows.append({
                    "model": model_name,
                    "id": row.get("id"),
                    "filename": row.get("filename"),
                    "error": str(exc),
                })

        if not rows:
            continue
        summary_rows.append({
            "model": model_name,
            "samples": len(rows),
            "failed": sum(1 for failed in failed_rows if failed["model"] == model_name),
            **asr_scores([r["hypothesis_raw_metric"] for r in rows], [r["reference_raw_metric"] for r in rows], prefix="raw_"),
            **asr_scores([r["hypothesis_normalized"] for r in rows], [r["reference_normalized"] for r in rows], prefix="normalized_"),
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(prediction_rows).to_csv(output_dir / "asr_all_predictions.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "asr_all_metrics_summary.csv", index=False)
    pd.DataFrame(grouped_scores(prediction_rows, "source")).to_csv(output_dir / "asr_all_metrics_by_source.csv", index=False)
    pd.DataFrame(grouped_scores(prediction_rows, "gender")).to_csv(output_dir / "asr_all_metrics_by_gender.csv", index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(output_dir / "asr_all_failed_rows.csv", index=False)
    (output_dir / "asr_all_metrics_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    (output_dir / "asr_all_metrics_summary.txt").write_text(
        "\n".join([
            "ASR Model Comparison",
            *[
                f"{row['model']}: {row['samples']} samples, normalized WER {row['normalized_wer_%']}%, normalized CER {row['normalized_cer_%']}%"
                for row in summary_rows
            ],
        ]),
        encoding="utf-8",
    )
    log_to_wandb(summary_rows, output_dir)
    print(summary_rows)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/model_registry.yaml")
    parser.add_argument("--model", action="append")
    parser.add_argument("--output-dir", default="outputs/asr-all/results")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dialect", action="append")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-supabase", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", default=False)
    parsed_args = parser.parse_args()
    if parsed_args.dialect is None:
        parsed_args.dialect = ["hunza"]
    main(parsed_args)
