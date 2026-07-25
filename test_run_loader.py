import argparse

import numpy as np

from data.audio_io import load_audio_16k
from data.loader import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mt", choices=["asr", "mt"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-supabase", action="store_true", default=False)
    parser.add_argument("--dialect", action="append")
    parser.add_argument("--source", choices=["hf", "supabase"])
    parser.add_argument("--decode-audio", action="store_true", default=False)
    args = parser.parse_args()

    dataset = load_dataset(
        task=args.task,
        split=args.split,
        use_hf=args.use_hf,
        use_supabase=args.use_supabase,
        dialects=args.dialect,
    )

    print(f"Loaded {len(dataset)} rows for task={args.task}, split={args.split}")
    if args.source:
        dataset = [row for row in dataset if row.get("source") == args.source]
        print(f"Rows after source={args.source} filter: {len(dataset)}")
    if len(dataset):
        row = dataset[0]
        safe_row = {key: value for key, value in row.items() if key != "audio"}
        safe_row["audio"] = "<audio omitted>"
        print(safe_row)
        if args.decode_audio:
            audio = load_audio_16k(row["audio"])
            print({
                "decoded_samples": int(audio.shape[0]),
                "sample_rate": 16000,
                "duration_seconds": round(float(audio.shape[0]) / 16000, 3),
                "peak": round(float(np.max(np.abs(audio))), 6),
                "rms": round(float(np.sqrt(np.mean(audio ** 2))), 6),
            })
    elif args.source:
        raise ValueError(f"No rows found after filtering for source={args.source}")


if __name__ == "__main__":
    main()
