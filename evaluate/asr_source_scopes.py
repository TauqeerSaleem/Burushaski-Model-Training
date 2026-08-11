import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCOPES = {
    "hf_test": ["--use-hf"],
    "supabase_hunza": ["--use-supabase"],
    "hf_test_plus_supabase": ["--use-hf", "--use-supabase"],
}

DEFAULT_SCOPES = ["hf_test", "supabase_hunza"]


def run_scope(scope, flags, args):
    output_dir = Path(args.output_root) / scope
    command = [
        sys.executable,
        "evaluate/asr_all.py",
        "--split",
        args.split,
        "--output-dir",
        str(output_dir),
    ]
    for dialect in args.dialect:
        command.extend(["--dialect", dialect])
    for model in args.model or []:
        command.extend(["--model", model])
    if args.max_samples:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.cpu:
        command.append("--cpu")
    command.extend(flags)

    print(f"\nRunning ASR scope: {scope}")
    print(" ".join(command))
    subprocess.run(command, check=True)
    return output_dir


def collect_summaries(output_dirs, output_root):
    rows = []
    source_rows = []
    content_rows = []
    module_rows = []
    gender_rows = []
    for scope, output_dir in output_dirs.items():
        summary_path = output_dir / "asr_all_metrics_summary.csv"
        if not summary_path.exists():
            continue
        frame = pd.read_csv(summary_path)
        frame.insert(0, "scope", scope)
        rows.append(frame)
        content_path = output_dir / "asr_all_metrics_by_content_type.csv"
        if content_path.exists():
            content = pd.read_csv(content_path)
            content.insert(0, "scope", scope)
            content_rows.append(content)
        source_path = output_dir / "asr_all_metrics_by_source.csv"
        if source_path.exists():
            source = pd.read_csv(source_path)
            source.insert(0, "scope", scope)
            source_rows.append(source)
        module_path = output_dir / "asr_all_metrics_by_module.csv"
        if module_path.exists():
            module = pd.read_csv(module_path)
            module.insert(0, "scope", scope)
            module_rows.append(module)
        gender_path = output_dir / "asr_all_metrics_by_gender.csv"
        if gender_path.exists():
            gender = pd.read_csv(gender_path)
            gender.insert(0, "scope", scope)
            gender_rows.append(gender)

    comparison_dir = Path(output_root)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise FileNotFoundError("No ASR summary files were created.")

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(comparison_dir / "asr_scope_summary.csv", index=False)
    if source_rows:
        pd.concat(source_rows, ignore_index=True).to_csv(comparison_dir / "asr_metrics_by_scope_and_source.csv", index=False)
    if content_rows:
        pd.concat(content_rows, ignore_index=True).to_csv(comparison_dir / "asr_metrics_by_scope_and_content_type.csv", index=False)
    if module_rows:
        pd.concat(module_rows, ignore_index=True).to_csv(comparison_dir / "asr_metrics_by_scope_and_module.csv", index=False)
    if gender_rows:
        pd.concat(gender_rows, ignore_index=True).to_csv(comparison_dir / "asr_metrics_by_scope_and_gender.csv", index=False)

    lines = [
        "ASR source-scope comparison",
        "",
        "Default run reports HF test and Supabase Hunza separately. Combined HF+Supabase is optional because it mostly hides where the errors came from.",
        "",
        "Overall:",
    ]
    for _, row in combined.sort_values(["scope", "normalized_wer_%"]).iterrows():
        lines.append(
            f"- {row['scope']} | {row['model']}: {row['samples']} samples, "
            f"normalized WER {row['normalized_wer_%']}%, "
            f"normalized CER {row['normalized_cer_%']}%"
        )
    if source_rows:
        source = pd.concat(source_rows, ignore_index=True)
        lines.extend(["", "By data source:"])
        for _, row in source.sort_values(["scope", "source", "normalized_wer_%"]).iterrows():
            lines.append(
                f"- {row['scope']} | {row['source']} | {row['model']}: "
                f"{row['samples']} samples, normalized WER {row['normalized_wer_%']}%, "
                f"normalized CER {row['normalized_cer_%']}%"
            )
    if content_rows:
        content = pd.concat(content_rows, ignore_index=True)
        lines.extend(["", "By content type:"])
        for _, row in content.sort_values(["scope", "content_type", "normalized_wer_%"]).iterrows():
            lines.append(
                f"- {row['scope']} | {row['content_type']} | {row['model']}: "
                f"{row['samples']} samples, normalized WER {row['normalized_wer_%']}%, "
                f"normalized CER {row['normalized_cer_%']}%"
            )
    if module_rows:
        module = pd.concat(module_rows, ignore_index=True)
        lines.extend(["", "By module:"])
        for _, row in module.sort_values(["scope", "module", "normalized_wer_%"]).iterrows():
            lines.append(
                f"- {row['scope']} | {row['module']} | {row['model']}: "
                f"{row['samples']} samples, normalized WER {row['normalized_wer_%']}%, "
                f"normalized CER {row['normalized_cer_%']}%"
            )
    if gender_rows:
        gender = pd.concat(gender_rows, ignore_index=True)
        lines.extend(["", "By gender:"])
        for _, row in gender.sort_values(["scope", "gender", "normalized_wer_%"]).iterrows():
            lines.append(
                f"- {row['scope']} | {row['gender']} | {row['model']}: "
                f"{row['samples']} samples, normalized WER {row['normalized_wer_%']}%, "
                f"normalized CER {row['normalized_cer_%']}%"
            )
    (comparison_dir / "asr_scope_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nASR source-scope summary saved to {comparison_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/asr-all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dialect", action="append", default=None)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--scope", action="append", choices=sorted(SCOPES), default=None)
    parser.add_argument("--include-combined", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    if args.dialect is None:
        args.dialect = ["hunza"]

    selected_scopes = args.scope or (list(SCOPES) if args.include_combined else DEFAULT_SCOPES)
    output_dirs = {}
    for scope in selected_scopes:
        output_dirs[scope] = run_scope(scope, SCOPES[scope], args)

    collect_summaries(output_dirs, args.output_root)


if __name__ == "__main__":
    main()
