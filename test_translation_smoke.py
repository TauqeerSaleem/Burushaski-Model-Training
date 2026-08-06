import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data.loader import load_dataset
from train.mt5 import limit_rows, make_translation_pairs, read_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mt5_short.yaml")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-supabase", action="store_true", default=False)
    parser.add_argument("--dialect", action="append")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--forward", action="store_true", default=False)
    args = parser.parse_args()

    config = read_config(args.config)
    dialects = args.dialect or config.get("target_dialects")
    train_data = load_dataset("mt", config.get("train_split", "train"), args.use_hf, False, dialects)
    eval_data = load_dataset("mt", config.get("eval_split", "test"), args.use_hf, args.use_supabase, dialects)

    train_pairs = limit_rows(make_translation_pairs(train_data, config), args.max_samples)
    eval_pairs = limit_rows(make_translation_pairs(eval_data, config), args.max_samples)
    if not train_pairs:
        raise ValueError("No train translation pairs found")
    if not eval_pairs:
        raise ValueError("No eval translation pairs found")

    print(f"Train pairs checked: {len(train_pairs)}")
    print(f"Eval pairs checked: {len(eval_pairs)}")
    print("Example source:", train_pairs[0]["source_text"])
    print("Example target:", train_pairs[0]["target_text"])

    tokenizer = AutoTokenizer.from_pretrained(config.get("model_name", "google/mt5-base"))
    sample = tokenizer(
        train_pairs[0]["source_text"],
        text_target=train_pairs[0]["target_text"],
        max_length=config.get("max_source_length", 128),
        truncation=True,
    )
    print("Tokenized source tokens:", len(sample["input_ids"]))
    print("Tokenized target tokens:", len(sample["labels"]))

    if args.forward:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(config.get("model_name", "google/mt5-base")).to(device)
        batch = tokenizer(
            [pair["source_text"] for pair in train_pairs[:2]],
            text_target=[pair["target_text"] for pair in train_pairs[:2]],
            padding=True,
            truncation=True,
            max_length=config.get("max_source_length", 128),
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            loss = model(**batch).loss
        print("Forward loss:", round(float(loss.detach().cpu()), 4))


if __name__ == "__main__":
    main()

