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
    for scope, output_dir in output_dirs.items():
        summary_path = output_dir / "cascade_all_metrics_summary.csv"
        if not summary_path.exists():
            continue
        frame = pd.read_csv(summary_path)
        frame.insert(0, "scope", scope)
        rows.append(frame)

    combined_dir = Path(output_root) / "source_scope_comparison"
    combined_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise FileNotFoundError("No cascade summary files were created.")

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(combined_dir / "cascade_source_scope_comparison.csv", index=False)

    lines = ["Cascade source-scope comparison"]
    for _, row in combined.sort_values(["scope", "chrf++"], ascending=[True, False]).iterrows():
        lines.append(
            f"{row['scope']} | {row['system']}: {row['samples']} samples, "
            f"BLEU {row['bleu']}, chrF++ {row['chrf++']}, TER {row['ter']}"
        )
    (combined_dir / "cascade_source_scope_comparison.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCombined cascade source-scope summary saved to {combined_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/cascade-all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dialect", action="append", default=None)
    parser.add_argument("--asr-model", action="append", default=None)
    parser.add_argument("--translator", action="append", default=None)
    parser.add_argument("--scope", action="append", choices=sorted(SCOPES), default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    if args.dialect is None:
        args.dialect = ["hunza"]
    if args.translator is None:
        args.translator = ["mt5", "mbart"]

    selected_scopes = args.scope or list(SCOPES)
    output_dirs = {}
    for scope in selected_scopes:
        output_dirs[scope] = run_scope(scope, SCOPES[scope], args)

    collect_summaries(output_dirs, args.output_root)


if __name__ == "__main__":
    main()
