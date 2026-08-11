import argparse
import random
import re
import sys
from collections import Counter, defaultdict, deque
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


def read_parquets(parquet_dir):
    frames = []
    for path in sorted(Path(parquet_dir).glob("*.parquet")):
        columns = [name for name in pq.ParquetFile(path).schema_arrow.names if name != "audio"]
        frame = pd.read_parquet(path, columns=columns)
        frame["original_split"] = "test" if path.name.startswith("test") else "train"
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


def standardize(frame):
    rows = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        bsk_raw = first_present(row_dict, "transcription", "transcript", "bsk_text_raw", "burushaski")
        english = first_present(row_dict, "english_translation", "english_text", "translation")
        rows.append({
            "original_split": first_present(row_dict, "original_split") or "unknown",
            "dialect": (first_present(row_dict, "dialect") or "unknown").lower(),
            "participant_id": first_present(row_dict, "participant_id", "speaker_id") or "unknown",
            "filename": first_present(row_dict, "filename", "id", "sample_id"),
            "bsk_text": bsk_raw,
            "bsk_norm": normalize_burushaski_text(bsk_raw),
            "english_text": english,
            "english_norm": clean_english(english),
        })
    return pd.DataFrame(rows)


def choose_canonical(group):
    counts = Counter(group["english_norm"])
    top_norm = counts.most_common(1)[0][0]
    top_rows = group[group["english_norm"] == top_norm]
    english_text = top_rows["english_text"].value_counts().index[0]
    return pd.Series({
        "bsk_text": group["bsk_text"].value_counts().index[0],
        "bsk_norm": group.name,
        "english_text": english_text,
        "english_norm": top_norm,
        "dialect": group["dialect"].value_counts().index[0],
        "source": "hf_clean_mt",
        "source_rows": len(group),
        "english_reference_count": group["english_norm"].nunique(),
        "split_group": group["split_group"].value_counts().index[0],
        "original_splits": ",".join(sorted(set(group["original_split"]))),
        "example_filenames": " || ".join(list(dict.fromkeys(group["filename"].astype(str)))[:5]),
    })


def split_sources(rows, validation_size, test_size, seed):
    sources = rows["split_group"].drop_duplicates().tolist()
    rng = random.Random(seed)
    rng.shuffle(sources)

    n_total = len(sources)
    n_test = max(1, round(n_total * test_size)) if n_total > 2 else 0
    n_val = max(1, round(n_total * validation_size)) if n_total > 3 else 0

    test_sources = set(sources[:n_test])
    val_sources = set(sources[n_test:n_test + n_val])

    def label(source):
        if source in test_sources:
            return "test"
        if source in val_sources:
            return "validation"
        return "train"

    rows = rows.copy()
    rows["split"] = rows["split_group"].map(label)
    return rows


def assign_split_groups(frame):
    bsk_to_eng = defaultdict(set)
    eng_to_bsk = defaultdict(set)
    for row in frame.itertuples():
        bsk_to_eng[row.bsk_norm].add(row.english_norm)
        eng_to_bsk[row.english_norm].add(row.bsk_norm)

    seen_bsk = set()
    group_by_bsk = {}
    group_index = 0

    for source in sorted(bsk_to_eng):
        if source in seen_bsk:
            continue
        group_index += 1
        group_id = f"group_{group_index:05d}"
        queue = deque([source])
        seen_bsk.add(source)
        while queue:
            current_bsk = queue.popleft()
            group_by_bsk[current_bsk] = group_id
            for english in bsk_to_eng[current_bsk]:
                for linked_bsk in eng_to_bsk[english]:
                    if linked_bsk not in seen_bsk:
                        seen_bsk.add(linked_bsk)
                        queue.append(linked_bsk)

    frame = frame.copy()
    frame["split_group"] = frame["bsk_norm"].map(group_by_bsk)
    return frame


def build_conflict_report(frame):
    grouped = (
        frame.groupby("bsk_norm", dropna=False)
        .agg(
            rows=("bsk_norm", "size"),
            english_reference_count=("english_norm", "nunique"),
            example_bsk=("bsk_text", "first"),
            english_examples=("english_text", lambda values: " || ".join(list(dict.fromkeys(str(v) for v in values if str(v).strip()))[:10])),
            filenames=("filename", lambda values: " || ".join(list(dict.fromkeys(str(v) for v in values if str(v).strip()))[:10])),
        )
        .reset_index()
    )
    return grouped[grouped["english_reference_count"] > 1].sort_values(
        ["english_reference_count", "rows"],
        ascending=[False, False],
    )


def format_for_training(rows):
    formatted = rows.copy()
    formatted["source_text"] = formatted.apply(
        lambda row: f"translate bsk_{row['dialect']} to eng: {row['bsk_text']}",
        axis=1,
    )
    formatted["target_text"] = formatted["english_text"]
    formatted["direction"] = "bsk_to_eng"
    return formatted[[
        "source_text",
        "target_text",
        "direction",
        "dialect",
        "bsk_text",
        "bsk_norm",
        "english_text",
        "english_norm",
        "source_rows",
        "english_reference_count",
        "split_group",
        "original_splits",
        "example_filenames",
    ]]


def split_leakage_check(rows):
    rows_by_split = {split: group for split, group in rows.groupby("split", dropna=False)}
    splits = sorted(rows_by_split)
    records = []
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            left_rows = rows_by_split[left]
            right_rows = rows_by_split[right]
            records.append({
                "left_split": left,
                "right_split": right,
                "shared_bsk_sources": len(set(left_rows["bsk_norm"]) & set(right_rows["bsk_norm"])),
                "shared_english_prompts": len(set(left_rows["english_norm"]) & set(right_rows["english_norm"])),
                "shared_split_groups": len(set(left_rows["split_group"]) & set(right_rows["split_group"])),
            })
    return pd.DataFrame(records)


def write_summary(rows, conflicts, leakage, output_dir):
    lines = [
        "Clean MT dataset summary",
        "",
        f"Rows after source-level grouping: {len(rows)}",
        f"Train rows: {int((rows['split'] == 'train').sum())}",
        f"Validation rows: {int((rows['split'] == 'validation').sum())}",
        f"Test rows: {int((rows['split'] == 'test').sum())}",
        f"Sources with multiple English references: {len(conflicts)}",
        f"Split groups: {rows['split_group'].nunique()}",
        "",
        "Note:",
        "This dataset is split by connected Burushaski-source/English-prompt groups. The same normalized Burushaski source and the same normalized English prompt should not appear in more than one split.",
        "",
        "Leakage check:",
        *[
            f"- {row.left_split} vs {row.right_split}: shared BSK {row.shared_bsk_sources}, shared English {row.shared_english_prompts}, shared groups {row.shared_split_groups}"
            for row in leakage.itertuples()
        ],
    ]
    (output_dir / "mt_clean_dataset_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", default="../Yaraanburushaski_english_translation_v1/data")
    parser.add_argument("--output-dir", default="data/processed/mt_clean_hunza_v1")
    parser.add_argument("--dialect", default="hunza")
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = standardize(read_parquets(args.parquet_dir))
    frame = frame[
        (frame["dialect"] == args.dialect.lower())
        & (frame["bsk_norm"].str.len() > 0)
        & (frame["english_norm"].str.len() > 0)
    ].copy()

    exact_pairs = assign_split_groups(frame.drop_duplicates(["bsk_norm", "english_norm"]))
    conflicts = build_conflict_report(exact_pairs)
    canonical = (
        exact_pairs.groupby("bsk_norm", group_keys=False)[[
            "original_split",
            "dialect",
            "participant_id",
            "filename",
            "bsk_text",
            "english_text",
            "english_norm",
            "split_group",
        ]]
        .apply(choose_canonical)
        .reset_index(drop=True)
    )
    canonical = split_sources(canonical, args.validation_size, args.test_size, args.seed)
    leakage = split_leakage_check(canonical)

    for split in ["train", "validation", "test"]:
        format_for_training(canonical[canonical["split"] == split]).to_csv(output_dir / f"{split}.csv", index=False)

    conflicts.to_csv(output_dir / "conflict_report.csv", index=False)
    exact_pairs.to_csv(output_dir / "alternate_references.csv", index=False)
    canonical.to_csv(output_dir / "canonical_rows.csv", index=False)
    leakage.to_csv(output_dir / "split_leakage_check.csv", index=False)
    write_summary(canonical, conflicts, leakage, output_dir)

    print(f"Clean MT dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
