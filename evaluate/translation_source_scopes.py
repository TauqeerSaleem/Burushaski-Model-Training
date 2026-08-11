import argparse
import json
import string
import subprocess
import sys
from pathlib import Path

import pandas as pd

from eval_metrics import mt_scores


TRANSLATORS = {
    "mt5": {
        "script": "evaluate/mt5.py",
        "model_path": "outputs/mt5-hunza-clean_final",
        "config": "configs/mt5_clean.yaml",
        "prediction_file": "mt5_predictions.csv",
    },
    "mbart": {
        "script": "evaluate/mbart.py",
        "model_path": "outputs/mbart-hunza-bsk-eng-clean_final",
        "config": "configs/mbart_clean.yaml",
        "prediction_file": "mbart_predictions.csv",
    },
}


def normalize_text(text):
    text = str(text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def score_frame(frame):
    hypotheses = [normalize_text(text) for text in frame["hypothesis"].tolist()]
    references = [normalize_text(text) for text in frame["reference"].tolist()]
    return mt_scores(hypotheses, references)


def grouped_summary(frame, group_keys):
    rows = []
    for keys, group in frame.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        row["samples"] = len(group)
        row.update(score_frame(group))
        rows.append(row)
    return rows


def run_eval(name, spec, scope, output_dir, args):
    command = [
        sys.executable,
        spec["script"],
        "--model-path",
        spec["model_path"],
        "--config",
        spec["config"],
        "--split",
        "test",
        "--output-dir",
        str(output_dir),
    ]
    if scope == "supabase_hunza":
        command.extend(["--use-supabase", "--dialect", "hunza"])
    if args.max_samples:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.cpu:
        command.append("--cpu")

    print(f"\nEvaluating {name}: {scope}")
    print(" ".join(command))
    subprocess.run(command, check=True)

    prediction_path = output_dir / spec["prediction_file"]
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file was not created: {prediction_path}")
    frame = pd.read_csv(prediction_path)
    frame.insert(0, "translator", name)
    frame.insert(1, "scope", scope)
    return frame


def write_scope_outputs(frame, output_dir, translator, scope):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{translator}_{scope}_predictions.csv", index=False)
    scores = {
        "translator": translator,
        "scope": scope,
        "samples": len(frame),
        **score_frame(frame),
    }
    pd.DataFrame([scores]).to_csv(output_dir / f"{translator}_{scope}_metrics_summary.csv", index=False)
    (output_dir / f"{translator}_{scope}_metrics_summary.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    (output_dir / f"{translator}_{scope}_metrics_summary.txt").write_text(
        "\n".join([
            f"{translator} translation summary: {scope}",
            f"Samples evaluated: {scores['samples']}",
            f"BLEU: {scores['bleu']}",
            f"chrF++: {scores['chrf++']}",
            f"TER: {scores['ter']}",
            f"Exact match: {scores['exact_match_%']}%",
            f"Empty predictions: {scores['empty_prediction_%']}%",
            f"Mean length ratio: {scores['mean_length_ratio']}",
        ]),
        encoding="utf-8",
    )
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/text-translation")
    parser.add_argument("--translator", action="append", choices=sorted(TRANSLATORS), default=None)
    parser.add_argument("--include-combined", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    selected = args.translator or list(TRANSLATORS)
    output_root = Path(args.output_root)
    summary_rows = []
    direction_rows = []
    source_rows = []
    dialect_rows = []

    for name in selected:
        spec = TRANSLATORS[name]
        hf_dir = output_root / name / "hf_clean_test"
        supabase_dir = output_root / name / "supabase_hunza"
        combined_dir = output_root / name / "hf_clean_test_plus_supabase"

        hf_frame = run_eval(name, spec, "hf_clean_test", hf_dir, args)
        supabase_frame = run_eval(name, spec, "supabase_hunza", supabase_dir, args)

        summary_rows.append(write_scope_outputs(hf_frame, hf_dir, name, "hf_clean_test"))
        summary_rows.append(write_scope_outputs(supabase_frame, supabase_dir, name, "supabase_hunza"))
        direction_rows.extend(grouped_summary(hf_frame, ["translator", "scope", "direction"]))
        direction_rows.extend(grouped_summary(supabase_frame, ["translator", "scope", "direction"]))
        source_rows.extend(grouped_summary(hf_frame, ["translator", "scope", "source_name"]))
        source_rows.extend(grouped_summary(supabase_frame, ["translator", "scope", "source_name"]))
        dialect_rows.extend(grouped_summary(hf_frame, ["translator", "scope", "dialect"]))
        dialect_rows.extend(grouped_summary(supabase_frame, ["translator", "scope", "dialect"]))
        if args.include_combined:
            combined_frame = pd.concat([hf_frame, supabase_frame], ignore_index=True)
            summary_rows.append(write_scope_outputs(combined_frame, combined_dir, name, "hf_clean_test_plus_supabase"))
            direction_rows.extend(grouped_summary(combined_frame, ["translator", "scope", "direction"]))
            source_rows.extend(grouped_summary(combined_frame, ["translator", "scope", "source_name"]))
            dialect_rows.extend(grouped_summary(combined_frame, ["translator", "scope", "dialect"]))

    comparison_dir = output_root
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(comparison_dir / "translation_scope_summary.csv", index=False)
    pd.DataFrame(direction_rows).to_csv(comparison_dir / "translation_metrics_by_direction.csv", index=False)
    pd.DataFrame(source_rows).to_csv(comparison_dir / "translation_metrics_by_source.csv", index=False)
    pd.DataFrame(dialect_rows).to_csv(comparison_dir / "translation_metrics_by_dialect.csv", index=False)

    lines = [
        "Text translation source-scope comparison",
        "",
        "Default run reports clean HF test and Supabase Hunza separately. Combined results are optional because the split is more useful for analysis.",
        "",
        "Overall:",
    ]
    for _, row in summary.sort_values(["scope", "chrf++"], ascending=[True, False]).iterrows():
        lines.append(
            f"{row['scope']} | {row['translator']}: {row['samples']} samples, "
            f"BLEU {row['bleu']}, chrF++ {row['chrf++']}, TER {row['ter']}"
        )
    if direction_rows:
        lines.extend(["", "By direction:"])
        for row in sorted(direction_rows, key=lambda item: (item["scope"], item["direction"], -item["chrf++"])):
            lines.append(
                f"{row['scope']} | {row['direction']} | {row['translator']}: "
                f"{row['samples']} samples, BLEU {row['bleu']}, chrF++ {row['chrf++']}, TER {row['ter']}"
            )
    if source_rows:
        lines.extend(["", "By data source:"])
        for row in sorted(source_rows, key=lambda item: (item["scope"], str(item["source_name"]), -item["chrf++"])):
            lines.append(
                f"{row['scope']} | {row['source_name']} | {row['translator']}: "
                f"{row['samples']} samples, BLEU {row['bleu']}, chrF++ {row['chrf++']}, TER {row['ter']}"
            )
    if dialect_rows:
        lines.extend(["", "By dialect:"])
        for row in sorted(dialect_rows, key=lambda item: (item["scope"], str(item["dialect"]), -item["chrf++"])):
            lines.append(
                f"{row['scope']} | {row['dialect']} | {row['translator']}: "
                f"{row['samples']} samples, BLEU {row['bleu']}, chrF++ {row['chrf++']}, TER {row['ter']}"
            )
    (comparison_dir / "translation_scope_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTranslation source-scope summary saved to {comparison_dir}")


if __name__ == "__main__":
    main()
