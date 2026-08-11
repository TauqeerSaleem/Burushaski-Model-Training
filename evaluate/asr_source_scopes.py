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
    for scope, output_dir in output_dirs.items():
        summary_path = output_dir / "asr_all_metrics_summary.csv"
        if not summary_path.exists():
            continue
        frame = pd.read_csv(summary_path)
        frame.insert(0, "scope", scope)
        rows.append(frame)

    combined_dir = Path(output_root) / "source_scope_comparison"
    combined_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise FileNotFoundError("No ASR summary files were created.")

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(combined_dir / "asr_source_scope_comparison.csv", index=False)

    lines = ["ASR source-scope comparison"]
    for _, row in combined.sort_values(["scope", "normalized_wer_%"]).iterrows():
        lines.append(
            f"{row['scope']} | {row['model']}: {row['samples']} samples, "
            f"normalized WER {row['normalized_wer_%']}%, "
            f"normalized CER {row['normalized_cer_%']}%"
        )
    (combined_dir / "asr_source_scope_comparison.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCombined ASR source-scope summary saved to {combined_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/asr-all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dialect", action="append", default=None)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--scope", action="append", choices=sorted(SCOPES), default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    if args.dialect is None:
        args.dialect = ["hunza"]

    selected_scopes = args.scope or list(SCOPES)
    output_dirs = {}
    for scope in selected_scopes:
        output_dirs[scope] = run_scope(scope, SCOPES[scope], args)

    collect_summaries(output_dirs, args.output_root)


if __name__ == "__main__":
    main()
