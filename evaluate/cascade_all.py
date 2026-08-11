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
from eval_metrics import asr_scores, mt_scores
from models.asr import ASRModel
from models.registry import get_model_specs
from models.translation import TextTranslator

load_dotenv()


def normalize_english(text):
    text = str(text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def normalize_bsk_metric(text):
    text = normalize_burushaski_text(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def infer_module(row):
    domain = str(row.get("domain") or "").strip()
    if domain and domain.lower() != "unknown":
        return domain
    filename = str(row.get("filename") or row.get("id") or "")
    parts = [part for part in filename.replace("\\", "/").split("/") if part]
    if len(parts) >= 3 and parts[0].lower() in {"hunza", "nagar", "yasin"}:
        return parts[2]
    return "hf_prompted_sentences" if str(row.get("source")) == "hf" else "unknown"


def infer_content_type(row):
    module = infer_module(row).lower()
    if module in {"image-prompts", "image_prompts", "picture_description"}:
        return "image_prompt"
    if str(row.get("source")) == "hf":
        return "hf_prompted_sentence"
    return "prompt_bank"


def score_translation(hypotheses, references):
    return mt_scores(
        [normalize_english(text) for text in hypotheses],
        [normalize_english(text) for text in references],
    )


def grouped_mt_scores(frame, group_keys):
    rows = []
    for keys, group in frame.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        row["samples"] = len(group)
        row.update(score_translation(group["hypothesis_english"].tolist(), group["reference_english"].tolist()))
        rows.append(row)
    return rows


def grouped_asr_scores(frame, group_keys):
    rows = []
    for keys, group in frame.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        row["samples"] = len(group)
        row.update(asr_scores(
            group["asr_bsk"].map(normalize_bsk_metric).tolist(),
            group["reference_bsk"].map(normalize_bsk_metric).tolist(),
            prefix="normalized_",
        ))
        rows.append(row)
    return rows


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    asr_specs = get_model_specs("asr", args.asr_model, args.registry)
    translator_specs = get_model_specs("text_translation", args.translator, args.registry)
    dataset = load_dataset("asr", args.split, args.use_hf, args.use_supabase, args.dialect)
    dataset = [row for row in dataset if str(row.get("english_translation") or "").strip()]
    if args.max_samples:
        dataset = dataset[:args.max_samples]
    print(f"Cascade eval rows loaded: {len(dataset)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failed_rows = []

    for translator_name, translator_spec in translator_specs.items():
        translator = TextTranslator(translator_name, translator_spec, device)
        for asr_name, asr_spec in asr_specs.items():
            print(f"Evaluating cascade: {asr_name} -> {translator_name}")
            asr_model = ASRModel(asr_name, asr_spec, device)
            for row in tqdm(dataset, desc=f"{asr_name}_to_{translator_name}"):
                try:
                    dialect = str(row.get("dialect") or "hunza").lower()
                    reference_bsk = normalize_burushaski_text(row.get("transcript") or "")
                    reference_english = row.get("english_translation") or ""
                    audio = load_audio_16k(row["audio"])
                    asr_text = asr_model.transcribe(audio)
                    translated = translator.translate_bsk_to_eng(
                        asr_text,
                        dialect=dialect,
                        max_length=args.max_length,
                        num_beams=args.num_beams,
                    )
                    rows.append({
                        "system": f"{asr_name}_to_{translator_name}",
                        "asr_model": asr_name,
                        "translator": translator_name,
                        "filename": row.get("filename") or row.get("id"),
                        "dialect": dialect,
                        "participant_id": row.get("participant_id"),
                        "gender": row.get("gender"),
                        "source": row.get("source"),
                        "module": infer_module(row),
                        "content_type": infer_content_type(row),
                        "reference_bsk": reference_bsk,
                        "asr_bsk": asr_text,
                        "reference_english": reference_english,
                        "hypothesis_english": translated,
                    })
                except Exception as exc:
                    failed_rows.append({
                        "system": f"{asr_name}_to_{translator_name}",
                        "id": row.get("id"),
                        "filename": row.get("filename"),
                        "error": str(exc),
                    })
            del asr_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del translator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise ValueError("No cascade predictions were produced.")

    frame = pd.DataFrame(rows)
    summary = []
    asr_summary = []
    for system, group in frame.groupby("system"):
        summary.append({
            "system": system,
            "samples": len(group),
            **score_translation(group["hypothesis_english"].tolist(), group["reference_english"].tolist()),
        })
        if group["asr_bsk"].fillna("").str.strip().any():
            asr_summary.append({
                "system": system,
                "samples": len(group),
                **asr_scores(
                    group["asr_bsk"].map(normalize_bsk_metric).tolist(),
                    group["reference_bsk"].map(normalize_bsk_metric).tolist(),
                    prefix="normalized_",
                ),
            })

    pd.DataFrame(rows).to_csv(output_dir / "cascade_all_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / "cascade_all_metrics_summary.csv", index=False)
    pd.DataFrame(asr_summary).to_csv(output_dir / "cascade_all_asr_metrics_summary.csv", index=False)
    mt_by_source = grouped_mt_scores(frame, ["system", "source"])
    mt_by_content = grouped_mt_scores(frame, ["system", "content_type"])
    mt_by_module = grouped_mt_scores(frame, ["system", "module"])
    mt_by_gender = grouped_mt_scores(frame, ["system", "gender"])
    asr_by_source = grouped_asr_scores(frame, ["system", "source"])
    asr_by_content = grouped_asr_scores(frame, ["system", "content_type"])
    asr_by_module = grouped_asr_scores(frame, ["system", "module"])
    pd.DataFrame(mt_by_source).to_csv(output_dir / "cascade_all_metrics_by_source.csv", index=False)
    pd.DataFrame(mt_by_content).to_csv(output_dir / "cascade_all_metrics_by_content_type.csv", index=False)
    pd.DataFrame(mt_by_module).to_csv(output_dir / "cascade_all_metrics_by_module.csv", index=False)
    pd.DataFrame(mt_by_gender).to_csv(output_dir / "cascade_all_metrics_by_gender.csv", index=False)
    pd.DataFrame(asr_by_source).to_csv(output_dir / "cascade_all_asr_metrics_by_source.csv", index=False)
    pd.DataFrame(asr_by_content).to_csv(output_dir / "cascade_all_asr_metrics_by_content_type.csv", index=False)
    pd.DataFrame(asr_by_module).to_csv(output_dir / "cascade_all_asr_metrics_by_module.csv", index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(output_dir / "cascade_all_failed_rows.csv", index=False)
    (output_dir / "cascade_all_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "cascade_all_asr_metrics_summary.json").write_text(json.dumps(asr_summary, indent=2), encoding="utf-8")
    (output_dir / "cascade_all_metrics_summary.txt").write_text(
        "\n".join([
            "ASR -> Text Translation Cascade Summary",
            "",
            "Translation quality:",
            *[
                f"{row['system']}: {row['samples']} samples, BLEU {row['bleu']}, chrF++ {row['chrf++']}, TER {row['ter']}"
                for row in summary
            ],
            "",
            "ASR quality inside the cascade:",
            *[
                f"{row['system']}: {row['samples']} samples, normalized WER {row['normalized_wer_%']}%, normalized CER {row['normalized_cer_%']}%"
                for row in asr_summary
            ],
            "",
            "Translation quality by content type:",
            *[
                f"{row['content_type']} | {row['system']}: {row['samples']} samples, BLEU {row['bleu']}, chrF++ {row['chrf++']}, TER {row['ter']}"
                for row in sorted(mt_by_content, key=lambda item: (item["content_type"], -item["chrf++"]))
            ],
            "",
            "Translation quality by module:",
            *[
                f"{row['module']} | {row['system']}: {row['samples']} samples, BLEU {row['bleu']}, chrF++ {row['chrf++']}, TER {row['ter']}"
                for row in sorted(mt_by_module, key=lambda item: (item["module"], -item["chrf++"]))
            ],
        ]),
        encoding="utf-8",
    )
    print(summary)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/model_registry.yaml")
    parser.add_argument("--asr-model", action="append")
    parser.add_argument("--translator", action="append")
    parser.add_argument("--output-dir", default="outputs/cascade-all/results")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dialect", action="append")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-supabase", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", default=False)
    parsed_args = parser.parse_args()
    if parsed_args.dialect is None:
        parsed_args.dialect = ["hunza"]
    if parsed_args.translator is None:
        parsed_args.translator = ["mt5"]
    main(parsed_args)
