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
        "evaluate/cascade_all.py",
        "--split",
        args.split,
        "--output-dir",
        str(output_dir),
    ]
    for dialect in args.dialect:
        command.extend(["--dialect", dialect])
    for asr_model in args.asr_model or []:
        command.extend(["--asr-model", asr_model])
    for translator in args.translator or []:
        command.extend(["--translator", translator])
    if args.max_samples:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.cpu:
        command.append("--cpu")
    command.extend(flags)

    print(f"\nRunning cascade scope: {scope}")
    print(" ".join(command))
    subprocess.run(command, check=True)
    return output_dir


def collect_summaries(output_dirs, output_root):
    rows = []
    breakdowns = {
        "cascade_metrics_by_scope_and_source.csv": "cascade_all_metrics_by_source.csv",
        "cascade_metrics_by_scope_and_content_type.csv": "cascade_all_metrics_by_content_type.csv",
        "cascade_metrics_by_scope_and_module.csv": "cascade_all_metrics_by_module.csv",
        "cascade_metrics_by_scope_and_gender.csv": "cascade_all_metrics_by_gender.csv",
        "cascade_asr_metrics_by_scope_and_source.csv": "cascade_all_asr_metrics_by_source.csv",
        "cascade_asr_metrics_by_scope_and_content_type.csv": "cascade_all_asr_metrics_by_content_type.csv",
        "cascade_asr_metrics_by_scope_and_module.csv": "cascade_all_asr_metrics_by_module.csv",
    }
    breakdown_rows = {name: [] for name in breakdowns}
    for scope, output_dir in output_dirs.items():
        summary_path = output_dir / "cascade_all_metrics_summary.csv"
        if not summary_path.exists():
            continue
        frame = pd.read_csv(summary_path)
        frame.insert(0, "scope", scope)
        rows.append(frame)
        for output_name, input_name in breakdowns.items():
            path = output_dir / input_name
            if path.exists():
                detail = pd.read_csv(path)
                detail.insert(0, "scope", scope)
                breakdown_rows[output_name].append(detail)

    comparison_dir = Path(output_root)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise FileNotFoundError("No cascade summary files were created.")

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(comparison_dir / "cascade_scope_summary.csv", index=False)
    for output_name, frames in breakdown_rows.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(comparison_dir / output_name, index=False)

    lines = [
        "Cascade source-scope comparison",
        "",
        "Default run reports HF test and Supabase Hunza separately. Combined results are optional because they hide the source of errors.",
        "",
    ]
    for _, row in combined.sort_values(["scope", "chrf++"], ascending=[True, False]).iterrows():
        lines.append(
            f"{row['scope']} | {row['system']}: {row['samples']} samples, "
            f"BLEU {row['bleu']}, chrF++ {row['chrf++']}, TER {row['ter']}"
        )
    content_frames = breakdown_rows["cascade_metrics_by_scope_and_content_type.csv"]
    if content_frames:
        content = pd.concat(content_frames, ignore_index=True)
        lines.extend(["", "By content type:"])
        for _, row in content.sort_values(["scope", "content_type", "chrf++"], ascending=[True, True, False]).iterrows():
            lines.append(
                f"{row['scope']} | {row['content_type']} | {row['system']}: "
                f"{row['samples']} samples, BLEU {row['bleu']}, chrF++ {row['chrf++']}, TER {row['ter']}"
            )
    module_frames = breakdown_rows["cascade_metrics_by_scope_and_module.csv"]
    if module_frames:
        module = pd.concat(module_frames, ignore_index=True)
        lines.extend(["", "By module:"])
        for _, row in module.sort_values(["scope", "module", "chrf++"], ascending=[True, True, False]).iterrows():
            lines.append(
                f"{row['scope']} | {row['module']} | {row['system']}: "
                f"{row['samples']} samples, BLEU {row['bleu']}, chrF++ {row['chrf++']}, TER {row['ter']}"
            )
    gender_frames = breakdown_rows["cascade_metrics_by_scope_and_gender.csv"]
    if gender_frames:
        gender = pd.concat(gender_frames, ignore_index=True)
        lines.extend(["", "By gender:"])
        for _, row in gender.sort_values(["scope", "gender", "chrf++"], ascending=[True, True, False]).iterrows():
            lines.append(
                f"{row['scope']} | {row['gender']} | {row['system']}: "
                f"{row['samples']} samples, BLEU {row['bleu']}, chrF++ {row['chrf++']}, TER {row['ter']}"
            )
    asr_content_frames = breakdown_rows["cascade_asr_metrics_by_scope_and_content_type.csv"]
    if asr_content_frames:
        asr_content = pd.concat(asr_content_frames, ignore_index=True)
        lines.extend(["", "ASR quality inside cascade by content type:"])
        for _, row in asr_content.sort_values(["scope", "content_type", "normalized_wer_%"]).iterrows():
            lines.append(
                f"{row['scope']} | {row['content_type']} | {row['system']}: "
                f"{row['samples']} samples, normalized WER {row['normalized_wer_%']}%, "
                f"normalized CER {row['normalized_cer_%']}%"
            )
    (comparison_dir / "cascade_scope_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCascade source-scope summary saved to {comparison_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/cascade-all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dialect", action="append", default=None)
    parser.add_argument("--asr-model", action="append", default=None)
    parser.add_argument("--translator", action="append", default=None)
    parser.add_argument("--scope", action="append", choices=sorted(SCOPES), default=None)
    parser.add_argument("--include-combined", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    if args.dialect is None:
        args.dialect = ["hunza"]
    if args.translator is None:
        args.translator = ["mt5", "mbart"]

    selected_scopes = args.scope or (list(SCOPES) if args.include_combined else DEFAULT_SCOPES)
    output_dirs = {}
    for scope in selected_scopes:
        output_dirs[scope] = run_scope(scope, SCOPES[scope], args)

    collect_summaries(output_dirs, args.output_root)


if __name__ == "__main__":
    main()
