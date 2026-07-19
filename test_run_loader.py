import argparse

from data.loader import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mt", choices=["asr", "mt", "s2tt", "tts"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-supabase", action="store_true", default=False)
    parser.add_argument("--use-mdc", action="store_true", default=False)
    args = parser.parse_args()

    dataset = load_dataset(
        task=args.task,
        split=args.split,
        use_hf=args.use_hf,
        use_supabase=args.use_supabase,
        use_mdc=args.use_mdc,
    )

    print(f"Loaded {len(dataset)} rows for task={args.task}, split={args.split}")
    if len(dataset):
        row = dataset[0]
        safe_row = {key: value for key, value in row.items() if key != "audio"}
        safe_row["audio"] = "<audio omitted>"
        print(safe_row)


if __name__ == "__main__":
    main()
