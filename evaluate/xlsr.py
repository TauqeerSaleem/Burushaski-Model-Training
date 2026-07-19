import argparse
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jiwer
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from data.audio_io import load_audio_16k
from data.loader import load_dataset
from data.normalization import normalize_burushaski_text

load_dotenv()


def normalize_for_wer(text: str) -> str:
    text = normalize_burushaski_text(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def decode_ctc(processor, logits):
    predicted_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(predicted_ids)[0]
    return text.replace("|", " ").strip()


def score_by_group(rows, group_key):
    summary = []
    for group, group_rows in pd.DataFrame(rows).groupby(group_key, dropna=False):
        refs = group_rows["reference_normalized"].tolist()
        hyps = group_rows["hypothesis_normalized"].tolist()
        summary.append({
            group_key: group,
            "samples": len(group_rows),
            "wer_%": round(jiwer.wer(refs, hyps) * 100, 2),
            "cer_%": round(jiwer.cer(refs, hyps) * 100, 2),
        })
    return summary


def main(args):
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    dataset = load_dataset(
        task="asr",
        split=args.split,
        use_hf=args.use_hf,
        use_supabase=False,
        use_mdc=False,
    )
    print(f"ASR evaluation examples: {len(dataset)}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_path).to(device)
    model.eval()

    rows, failed = [], []
    for row in tqdm(dataset, desc="Evaluating XLS-R"):
        try:
            audio = load_audio_16k(row["audio"])
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            hypothesis = decode_ctc(processor, logits)
            reference = row["transcript"]
            rows.append({
                "filename": row.get("filename") or row.get("id"),
                "dialect": row.get("dialect"),
                "participant_id": row.get("participant_id"),
                "reference": reference,
                "hypothesis": hypothesis,
                "reference_normalized": normalize_for_wer(reference),
                "hypothesis_normalized": normalize_for_wer(hypothesis),
            })
        except Exception as exc:
            failed.append({"id": row.get("id"), "error": str(exc)})

    if not rows:
        raise ValueError("No ASR predictions were produced.")

    refs = [row["reference_normalized"] for row in rows]
    hyps = [row["hypothesis_normalized"] for row in rows]
    metrics = {
        "samples": len(rows),
        "failed": len(failed),
        "wer_%": round(jiwer.wer(refs, hyps) * 100, 2),
        "cer_%": round(jiwer.cer(refs, hyps) * 100, 2),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "xlsr_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "xlsr_metrics_summary.csv", index=False)
    pd.DataFrame(score_by_group(rows, "dialect")).to_csv(output_dir / "xlsr_metrics_by_dialect.csv", index=False)
    pd.DataFrame(score_by_group(rows, "participant_id")).to_csv(output_dir / "xlsr_metrics_by_participant.csv", index=False)
    if failed:
        pd.DataFrame(failed).to_csv(output_dir / "xlsr_failed_rows.csv", index=False)

    print(metrics)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="outputs/asr-xlsr_final")
    parser.add_argument("--output-dir", default="outputs/asr-xlsr/results")
    parser.add_argument("--split", default="test")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    main(parser.parse_args())
