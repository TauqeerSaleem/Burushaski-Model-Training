import argparse
import json
from pathlib import Path

from data.loader import load_dataset
from data.normalization import normalize_burushaski_text


def canonicalize_row(row: dict, source: str | None = None) -> dict:
    transcript = row.get("bsk_text_raw") or row.get("transcript") or row.get("burushaski") or ""
    normalized = row.get("bsk_text_normalized") or normalize_burushaski_text(transcript)

    return {
        "sample_id": str(row.get("sample_id") or row.get("id") or row.get("filename") or ""),
        "source": row.get("source") or source or "unknown",
        "domain": row.get("domain") or row.get("prompt_type") or "prompted",
        "dialect": (row.get("dialect") or "unknown").lower(),
        "speaker_id": str(row.get("speaker_id") or row.get("participant_id") or "unknown"),
        "gender": (row.get("gender") or "unknown").lower(),
        "audio_path": row.get("audio") or row.get("audio_path") or "",
        "bsk_text_raw": transcript,
        "bsk_text_normalized": normalized,
        "english_text": row.get("english_text") or row.get("english_translation") or row.get("translation") or "",
        "quality": row.get("quality") or "unverified",
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(args):
    ds = load_dataset(
        task=args.task,
        split=args.split,
        use_hf=args.use_hf,
        use_supabase=args.use_supabase,
        dialects=args.dialect,
    )
    rows = [canonicalize_row(row) for row in ds]

    write_jsonl(rows, Path(args.output))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mt", choices=["asr", "mt"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/manifests/train.jsonl")
    parser.add_argument("--dialect", action="append")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-supabase", action="store_true", default=False)
    main(parser.parse_args())
