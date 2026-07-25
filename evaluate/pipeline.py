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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

from data.audio_io import load_audio_16k
from data.loader import load_dataset
from data.normalization import normalize_burushaski_text
from eval_metrics import asr_scores, mt_scores

load_dotenv()


def normalize_english(text: str) -> str:
    text = str(text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def normalize_bsk_metric(text: str) -> str:
    text = normalize_burushaski_text(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def plain_bsk_metric(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def score_translation(hypotheses, references):
    hyp_norm = [normalize_english(text) for text in hypotheses]
    ref_norm = [normalize_english(text) for text in references]
    return mt_scores(hyp_norm, ref_norm)


def grouped_system_scores(rows, group_key):
    summary = []
    frame = pd.DataFrame(rows)
    for group, group_rows in frame.groupby(group_key, dropna=False):
        references = group_rows["reference_english"].tolist()
        summary.append({
            group_key: group,
            "system": "gold_bsk_to_mt5",
            "samples": len(group_rows),
            **score_translation(group_rows["gold_bsk_to_english"].tolist(), references),
        })
        summary.append({
            group_key: group,
            "system": "xlsr_to_mt5",
            "samples": len(group_rows),
            **score_translation(group_rows["asr_cascade_to_english"].tolist(), references),
        })
    return summary


def log_to_wandb(metrics, output_dir):
    if not os.getenv("WANDB_API_KEY"):
        return
    import wandb

    wandb.init(project=os.getenv("WANDB_PROJECT", "xlsr_mt5_pipeline_hunza_v1"), job_type="eval", reinit=True)
    for row in metrics:
        prefix = row["system"]
        wandb.log({
            f"{prefix}/bleu": row["bleu"],
            f"{prefix}/chrf++": row["chrf++"],
            f"{prefix}/samples": row["samples"],
        })
    wandb.save(str(output_dir / "*.csv"))
    wandb.save(str(output_dir / "*.json"))
    wandb.save(str(output_dir / "*.txt"))
    wandb.finish()


def log_asr_to_wandb(metrics, output_dir):
    if not os.getenv("WANDB_API_KEY"):
        return
    import wandb

    wandb.init(project=os.getenv("WANDB_PROJECT", "xlsr_mt5_pipeline_hunza_v1"), job_type="eval", reinit=True)
    wandb.log({
        "pipeline_asr/raw_wer_%": metrics["raw_wer_%"],
        "pipeline_asr/raw_cer_%": metrics["raw_cer_%"],
        "pipeline_asr/raw_word_accuracy_%": metrics["raw_word_accuracy_%"],
        "pipeline_asr/raw_sentence_error_rate_%": metrics["raw_sentence_error_rate_%"],
        "pipeline_asr/normalized_wer_%": metrics["normalized_wer_%"],
        "pipeline_asr/normalized_cer_%": metrics["normalized_cer_%"],
        "pipeline_asr/normalized_word_accuracy_%": metrics["normalized_word_accuracy_%"],
        "pipeline_asr/normalized_sentence_error_rate_%": metrics["normalized_sentence_error_rate_%"],
        "pipeline_asr/samples": metrics["samples"],
    })
    wandb.save(str(output_dir / "*.csv"))
    wandb.save(str(output_dir / "*.json"))
    wandb.save(str(output_dir / "*.txt"))
    wandb.finish()


def decode_asr(asr_processor, asr_model, audio, device):
    inputs = asr_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = asr_model(inputs.input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return asr_processor.batch_decode(predicted_ids)[0].replace("|", " ").strip()


def translate(mt_tokenizer, mt_model, source_text, device, max_length, num_beams):
    inputs = mt_tokenizer(source_text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    with torch.no_grad():
        output_ids = mt_model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    return mt_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def main(args):
    if not Path(args.asr_model_path).exists():
        raise FileNotFoundError(f"ASR model path not found: {args.asr_model_path}")
    if not Path(args.mt_model_path).exists():
        raise FileNotFoundError(f"MT model path not found: {args.mt_model_path}")

    dataset = load_dataset(
        task="asr",
        split=args.split,
        use_hf=args.use_hf,
        use_supabase=args.use_supabase,
        dialects=args.dialect,
    )

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    asr_processor = Wav2Vec2Processor.from_pretrained(args.asr_model_path)
    asr_model = Wav2Vec2ForCTC.from_pretrained(args.asr_model_path).to(device)
    mt_tokenizer = AutoTokenizer.from_pretrained(args.mt_model_path)
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(args.mt_model_path).to(device)
    asr_model.eval()
    mt_model.eval()

    rows = []
    for row in tqdm(dataset, desc="Evaluating cascade"):
        dialect = str(row.get("dialect") or "unknown").lower()
        reference_english = row.get("english_translation") or ""
        if not reference_english.strip():
            continue

        audio = load_audio_16k(row["audio"])
        asr_text = decode_asr(asr_processor, asr_model, audio, device)
        gold_bsk = normalize_burushaski_text(row.get("transcript"))

        cascade_input = f"translate bsk_{dialect} to eng: {asr_text}"
        gold_input = f"translate bsk_{dialect} to eng: {gold_bsk}"
        cascade_english = translate(mt_tokenizer, mt_model, cascade_input, device, args.max_length, args.num_beams)
        gold_english = translate(mt_tokenizer, mt_model, gold_input, device, args.max_length, args.num_beams)

        rows.append({
            "filename": row.get("filename") or row.get("id"),
            "dialect": dialect,
            "participant_id": row.get("participant_id"),
            "source": row.get("source"),
            "gold_bsk": gold_bsk,
            "asr_bsk": asr_text,
            "reference_english": reference_english,
            "gold_bsk_to_english": gold_english,
            "asr_cascade_to_english": cascade_english,
        })

    if not rows:
        raise ValueError("No cascade predictions were produced.")

    references = [row["reference_english"] for row in rows]
    gold_hypotheses = [row["gold_bsk_to_english"] for row in rows]
    cascade_hypotheses = [row["asr_cascade_to_english"] for row in rows]
    metrics = [
        {"system": "gold_bsk_to_mt5", "samples": len(rows), **score_translation(gold_hypotheses, references)},
        {"system": "xlsr_to_mt5", "samples": len(rows), **score_translation(cascade_hypotheses, references)},
    ]
    gold_bsk_raw = [plain_bsk_metric(row["gold_bsk"]) for row in rows]
    asr_bsk_raw = [plain_bsk_metric(row["asr_bsk"]) for row in rows]
    gold_bsk_norm = [normalize_bsk_metric(row["gold_bsk"]) for row in rows]
    asr_bsk_norm = [normalize_bsk_metric(row["asr_bsk"]) for row in rows]
    asr_metrics = {
        "samples": len(rows),
        **asr_scores(asr_bsk_raw, gold_bsk_raw, prefix="raw_"),
        **asr_scores(asr_bsk_norm, gold_bsk_norm, prefix="normalized_"),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "pipeline_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "pipeline_metrics_summary.csv", index=False)
    pd.DataFrame([asr_metrics]).to_csv(output_dir / "pipeline_asr_metrics_summary.csv", index=False)
    (output_dir / "pipeline_metrics_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "pipeline_asr_metrics_summary.json").write_text(json.dumps(asr_metrics, indent=2), encoding="utf-8")
    (output_dir / "pipeline_metrics_summary.txt").write_text(
        "\n".join([
            "XLS-R -> mT5 Pipeline Evaluation Summary",
            f"Samples evaluated: {len(rows)}",
            f"Gold BSK -> mT5 BLEU: {metrics[0]['bleu']}",
            f"Gold BSK -> mT5 chrF++: {metrics[0]['chrf++']}",
            f"Gold BSK -> mT5 TER: {metrics[0]['ter']}",
            f"XLS-R -> mT5 BLEU: {metrics[1]['bleu']}",
            f"XLS-R -> mT5 chrF++: {metrics[1]['chrf++']}",
            f"XLS-R -> mT5 TER: {metrics[1]['ter']}",
            f"Pipeline ASR raw WER: {asr_metrics['raw_wer_%']}%",
            f"Pipeline ASR raw CER: {asr_metrics['raw_cer_%']}%",
            f"Pipeline ASR raw word accuracy: {asr_metrics['raw_word_accuracy_%']}%",
            f"Pipeline ASR normalized WER: {asr_metrics['normalized_wer_%']}%",
            f"Pipeline ASR normalized CER: {asr_metrics['normalized_cer_%']}%",
            f"Pipeline ASR normalized word accuracy: {asr_metrics['normalized_word_accuracy_%']}%",
            f"Pipeline ASR normalized sentence error rate: {asr_metrics['normalized_sentence_error_rate_%']}%",
        ]),
        encoding="utf-8",
    )
    pd.DataFrame(grouped_system_scores(rows, "dialect")).to_csv(output_dir / "pipeline_metrics_by_dialect.csv", index=False)
    pd.DataFrame(grouped_system_scores(rows, "source")).to_csv(output_dir / "pipeline_metrics_by_source.csv", index=False)
    log_to_wandb(metrics, output_dir)
    log_asr_to_wandb(asr_metrics, output_dir)
    print(metrics)
    print(asr_metrics)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-model-path", default="outputs/asr-xlsr_final")
    parser.add_argument("--mt-model-path", default="outputs/mt5-bsk-eng_final")
    parser.add_argument("--output-dir", default="outputs/pipeline/results")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dialect", action="append")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-supabase", action="store_true", default=False)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", default=False)
    parsed_args = parser.parse_args()
    if parsed_args.dialect is None:
        parsed_args.dialect = ["hunza"]
    main(parsed_args)
