import argparse
import json
import string
from pathlib import Path

import pandas as pd

from eval_metrics import asr_scores, mt_scores
from data.normalization import normalize_burushaski_text


def normalize_bsk(text):
    text = normalize_burushaski_text(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def normalize_text(text):
    text = str(text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def write_outputs(metrics, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_path.with_suffix(".csv"), index=False)
    output_path.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    output_path.with_suffix(".txt").write_text(
        "\n".join([f"{key}: {value}" for key, value in metrics.items()]),
        encoding="utf-8",
    )


def evaluate_asr(frame):
    refs_raw = frame["reference"].fillna("").map(lambda value: " ".join(str(value).lower().split())).tolist()
    hyps_raw = frame["hypothesis"].fillna("").map(lambda value: " ".join(str(value).lower().split())).tolist()
    refs_norm = frame["reference"].fillna("").map(normalize_bsk).tolist()
    hyps_norm = frame["hypothesis"].fillna("").map(normalize_bsk).tolist()
    return {
        "samples": len(frame),
        **asr_scores(hyps_raw, refs_raw, prefix="raw_"),
        **asr_scores(hyps_norm, refs_norm, prefix="normalized_"),
    }


def evaluate_mt(frame):
    references = frame["reference"].fillna("").map(normalize_text).tolist()
    hypotheses = frame["hypothesis"].fillna("").map(normalize_text).tolist()
    return {
        "samples": len(frame),
        **mt_scores(hypotheses, references),
    }


def main(args):
    frame = pd.read_csv(args.predictions)
    if args.task == "asr":
        required = {"reference", "hypothesis"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"ASR predictions file is missing columns: {sorted(missing)}")
        metrics = evaluate_asr(frame)
    else:
        required = {"reference", "hypothesis"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"MT predictions file is missing columns: {sorted(missing)}")
        metrics = evaluate_mt(frame)

    write_outputs(metrics, Path(args.output))
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["asr", "mt"], required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="outputs/recomputed_metrics/metrics")
    main(parser.parse_args())
