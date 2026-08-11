import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow.parquet as pq

from data.normalization import normalize_burushaski_text


def clean_english(text):
    text = str(text or "").lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def first_present(row, *columns):
    for column in columns:
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column]).strip()
    return ""


def parquet_columns(path):
    names = pq.ParquetFile(path).schema_arrow.names
    return [name for name in names if name != "audio"]


def read_parquet_rows(path, split):
    columns = parquet_columns(path)
    frame = pd.read_parquet(path, columns=columns)
    frame["split"] = split
    return frame


def load_local_parquets(parquet_dir):
    parquet_dir = Path(parquet_dir)
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    frames = []
    for path in files:
        split = "test" if path.name.startswith("test") else "train"
        frames.append(read_parquet_rows(path, split))
    return pd.concat(frames, ignore_index=True, sort=False)


def write_schema_summary(parquet_dir, output_dir):
    lines = ["HF parquet schema summary", ""]
    rows = []
    for path in sorted(Path(parquet_dir).glob("*.parquet")):
        schema = pq.ParquetFile(path).schema_arrow
        lines.append(path.name)
        for field in schema:
            lines.append(f"- {field.name}: {field.type}")
            rows.append({"file": path.name, "column": field.name, "type": str(field.type)})
        lines.append("")
    (output_dir / "hf_schema_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "hf_schema_summary.csv", index=False)


def standardize(frame):
    records = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        bsk_raw = first_present(row_dict, "transcription", "transcript", "bsk_text_raw", "burushaski")
        bsk_norm = first_present(row_dict, "bsk_text_normalized") or normalize_burushaski_text(bsk_raw)
        english = first_present(row_dict, "english_translation", "english_text", "translation")
        records.append({
            "split": first_present(row_dict, "split") or "unknown",
            "dialect": first_present(row_dict, "dialect").lower() or "unknown",
            "participant_id": first_present(row_dict, "participant_id", "speaker_id") or "unknown",
            "gender": first_present(row_dict, "gender").lower() or "unknown",
            "filename": first_present(row_dict, "filename", "id", "sample_id"),
            "bsk_raw": bsk_raw,
            "bsk_norm": bsk_norm,
            "english": english,
            "english_norm": clean_english(english),
        })
    return pd.DataFrame(records)


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8")


def missing_summary(frame):
    rows = []
    for split, group in frame.groupby("split", dropna=False):
        rows.append({
            "split": split,
            "rows": len(group),
            "missing_bsk_text": int((group["bsk_norm"].str.len() == 0).sum()),
            "missing_english": int((group["english_norm"].str.len() == 0).sum()),
            "missing_filename": int((group["filename"].str.len() == 0).sum()),
            "missing_participant_id": int((group["participant_id"].eq("unknown")).sum()),
            "missing_gender": int((group["gender"].eq("unknown")).sum()),
        })
    return pd.DataFrame(rows)


def split_counts(frame):
    rows = []
    for split, group in frame.groupby("split", dropna=False):
        rows.append({
            "split": split,
            "rows": len(group),
            "unique_bsk_texts": group["bsk_norm"].nunique(),
            "unique_english_texts": group["english_norm"].nunique(),
            "unique_bsk_english_pairs": group[["bsk_norm", "english_norm"]].drop_duplicates().shape[0],
            "unique_participants": group["participant_id"].nunique(),
            "unique_filenames": group["filename"].nunique(),
        })
    return pd.DataFrame(rows)


def value_counts(frame, column):
    return (
        frame[column]
        .fillna("unknown")
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="rows")
    )


def duplicate_pairs(frame):
    grouped = (
        frame.groupby(["split", "bsk_norm", "english_norm"], dropna=False)
        .agg(
            rows=("bsk_norm", "size"),
            participants=("participant_id", lambda values: len(set(values))),
            example_bsk=("bsk_raw", "first"),
            example_english=("english", "first"),
        )
        .reset_index()
    )
    return grouped[grouped["rows"] > 1].sort_values(["split", "rows"], ascending=[True, False])


def source_reference_conflicts(frame):
    grouped = (
        frame.groupby(["split", "bsk_norm"], dropna=False)
        .agg(
            rows=("bsk_norm", "size"),
            english_refs=("english_norm", "nunique"),
            participants=("participant_id", lambda values: len(set(values))),
            example_bsk=("bsk_raw", "first"),
            examples=("english", lambda values: " || ".join(list(dict.fromkeys(str(v) for v in values if str(v).strip()))[:8])),
        )
        .reset_index()
    )
    return grouped[grouped["english_refs"] > 1].sort_values(["split", "english_refs", "rows"], ascending=[True, False, False])


def english_source_conflicts(frame):
    grouped = (
        frame.groupby(["split", "english_norm"], dropna=False)
        .agg(
            rows=("english_norm", "size"),
            bsk_sources=("bsk_norm", "nunique"),
            example_english=("english", "first"),
            examples=("bsk_raw", lambda values: " || ".join(list(dict.fromkeys(str(v) for v in values if str(v).strip()))[:8])),
        )
        .reset_index()
    )
    return grouped[grouped["bsk_sources"] > 1].sort_values(["split", "bsk_sources", "rows"], ascending=[True, False, False])


def train_test_overlap(frame):
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    train_bsk = set(train["bsk_norm"])
    train_pairs = set(zip(train["bsk_norm"], train["english_norm"]))
    rows = []
    for _, row in test.iterrows():
        rows.append({
            "filename": row["filename"],
            "participant_id": row["participant_id"],
            "bsk_raw": row["bsk_raw"],
            "english": row["english"],
            "bsk_seen_in_train": row["bsk_norm"] in train_bsk,
            "exact_pair_seen_in_train": (row["bsk_norm"], row["english_norm"]) in train_pairs,
        })
    return pd.DataFrame(rows)


def suspicious_rows(frame):
    rows = []
    for _, row in frame.iterrows():
        reasons = []
        if not row["bsk_norm"]:
            reasons.append("missing_bsk")
        if not row["english_norm"]:
            reasons.append("missing_english")
        if row["bsk_norm"] and row["english_norm"] and row["bsk_norm"] == row["english_norm"]:
            reasons.append("same_bsk_and_english_after_normalization")
        if len(row["english_norm"].split()) <= 1 and row["english_norm"]:
            reasons.append("very_short_english")
        if len(row["english_norm"].split()) >= 25:
            reasons.append("very_long_english")
        if reasons:
            rows.append({**row.to_dict(), "reasons": "; ".join(reasons)})
    return pd.DataFrame(rows)


def write_summary(frame, output_dir):
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    overlap = train_test_overlap(frame)
    conflict_rows = source_reference_conflicts(frame)
    pair_dupes = duplicate_pairs(frame)
    dialect_counts = value_counts(frame, "dialect")
    gender_counts = value_counts(frame, "gender")

    lines = [
        "HF parquet audit summary",
        "",
        f"Total rows: {len(frame)}",
        f"Train rows: {len(train)}",
        f"Test rows: {len(test)}",
        f"Train unique Burushaski texts: {train['bsk_norm'].nunique()}",
        f"Test unique Burushaski texts: {test['bsk_norm'].nunique()}",
        f"Train unique Burushaski-English pairs: {train[['bsk_norm', 'english_norm']].drop_duplicates().shape[0]}",
        f"Test unique Burushaski-English pairs: {test[['bsk_norm', 'english_norm']].drop_duplicates().shape[0]}",
        f"Duplicate exact pairs: {len(pair_dupes)}",
        f"Burushaski sources with multiple English references: {len(conflict_rows)}",
        f"Test rows whose Burushaski text appears in train: {int(overlap['bsk_seen_in_train'].sum()) if not overlap.empty else 0}",
        f"Test rows whose exact pair appears in train: {int(overlap['exact_pair_seen_in_train'].sum()) if not overlap.empty else 0}",
        "",
        "Dialect counts:",
        *[f"- {row.dialect}: {row.rows}" for row in dialect_counts.itertuples()],
        "",
        "Gender counts:",
        *[f"- {row.gender}: {row.rows}" for row in gender_counts.itertuples()],
    ]
    (output_dir / "hf_dataset_audit_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "total_rows": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_unique_bsk_texts": int(train["bsk_norm"].nunique()),
        "test_unique_bsk_texts": int(test["bsk_norm"].nunique()),
        "test_rows_bsk_seen_in_train": int(overlap["bsk_seen_in_train"].sum()) if not overlap.empty else 0,
        "test_rows_exact_pair_seen_in_train": int(overlap["exact_pair_seen_in_train"].sum()) if not overlap.empty else 0,
    }
    (output_dir / "hf_dataset_audit_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", default="../Yaraanburushaski_english_translation_v1/data")
    parser.add_argument("--output-dir", default="analysis_outputs/hf_dataset_audit")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_schema_summary(args.parquet_dir, output_dir)
    raw = load_local_parquets(args.parquet_dir)
    frame = standardize(raw)

    write_csv(split_counts(frame), output_dir / "hf_split_counts.csv")
    write_csv(value_counts(frame, "dialect"), output_dir / "hf_dialect_counts.csv")
    write_csv(value_counts(frame, "gender"), output_dir / "hf_gender_counts.csv")
    write_csv(missing_summary(frame), output_dir / "hf_missing_values.csv")
    write_csv(duplicate_pairs(frame), output_dir / "hf_duplicate_pairs.csv")
    write_csv(source_reference_conflicts(frame), output_dir / "hf_source_reference_conflicts.csv")
    write_csv(english_source_conflicts(frame), output_dir / "hf_english_source_conflicts.csv")
    write_csv(train_test_overlap(frame), output_dir / "hf_train_test_overlap.csv")
    write_csv(suspicious_rows(frame), output_dir / "hf_suspicious_rows.csv")
    write_summary(frame, output_dir)

    print(f"HF parquet audit saved to {output_dir}")


if __name__ == "__main__":
    main()
