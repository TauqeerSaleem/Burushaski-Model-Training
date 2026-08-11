import argparse
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd
from dotenv import load_dotenv
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data.loader import load_dataset

load_dotenv()


def read_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def keep_dialects(dataset, dialects):
    wanted = {d.lower() for d in dialects or []}
    if not wanted:
        return dataset
    return [row for row in dataset if str(row.get("dialect", "")).lower() in wanted]


def make_translation_pairs(dataset, config):
    rows = []
    directions = set(config.get("directions", ["bsk_to_eng"]))
    bsk_field = config.get("bsk_text_field", "bsk_text_normalized")
    eng_field = config.get("english_field", "english_translation")

    for row in dataset:
        dialect = str(row.get("dialect") or "unknown").lower()
        bsk_text = (row.get(bsk_field) or row.get("transcript") or "").strip()
        english = (row.get(eng_field) or row.get("english_text") or "").strip()
        if not bsk_text or not english:
            continue

        if "bsk_to_eng" in directions:
            rows.append({
                "source_text": f"translate bsk_{dialect} to eng: {bsk_text}",
                "target_text": english,
                "direction": "bsk_to_eng",
                "dialect": dialect,
                "participant_id": row.get("participant_id"),
                "source": row.get("source"),
            })
        if "eng_to_bsk" in directions:
            rows.append({
                "source_text": f"translate eng to bsk_{dialect}: {english}",
                "target_text": bsk_text,
                "direction": "eng_to_bsk",
                "dialect": dialect,
                "participant_id": row.get("participant_id"),
                "source": row.get("source"),
            })

    return rows


def limit_rows(rows, limit):
    if limit is None:
        return rows
    limit = int(limit)
    if limit <= 0:
        return rows
    return rows[:limit]


def load_processed_mt_rows(config, split_name):
    processed_dir = config.get("processed_data_dir")
    if not processed_dir:
        return None

    split_files = config.get("processed_split_files", {})
    filename = split_files.get(split_name, f"{split_name}.csv")
    path = Path(processed_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed MT split not found: {path}")

    frame = pd.read_csv(path).fillna("")
    rows = []
    for _, row in frame.iterrows():
        source_text = str(row.get("source_text") or "").strip()
        target_text = str(row.get("target_text") or "").strip()
        if not source_text or not target_text:
            continue
        rows.append({
            "source_text": source_text,
            "target_text": target_text,
            "direction": row.get("direction") or "bsk_to_eng",
            "dialect": row.get("dialect") or "hunza",
            "participant_id": row.get("participant_id") or "",
            "source": row.get("source") or "processed_mt",
        })
    return rows


def evaluation_strategy_arg(arguments_cls, value):
    parameters = inspect.signature(arguments_cls.__init__).parameters
    key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    return {key: value}


class TextPairDataset:
    def __init__(self, rows, tokenizer, max_source_length, max_target_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        model_inputs = self.tokenizer(
            row["source_text"],
            max_length=self.max_source_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=row["target_text"],
            max_length=self.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def main(args):
    if args.fp16 and args.bf16:
        raise ValueError("Choose either --fp16 or --bf16, not both")

    config = read_config(args.config)
    os.environ.setdefault("WANDB_PROJECT", config.get("wandb_project", "mt5_bsk_eng"))
    target_dialects = config.get("target_dialects")

    train_rows = load_processed_mt_rows(config, "train")
    eval_rows = load_processed_mt_rows(config, config.get("eval_split", "validation"))

    if train_rows is None:
        train_data = load_dataset(
            task="mt",
            split=config.get("train_split", "train"),
            use_hf=args.use_hf,
            use_supabase=False,
            dialects=target_dialects,
        )
        eval_data = load_dataset(
            task="mt",
            split=config.get("eval_split", "test"),
            use_hf=args.use_hf,
            use_supabase=False,
            dialects=target_dialects,
        )
        train_rows = make_translation_pairs(train_data, config)
        eval_rows = make_translation_pairs(eval_data, config)
    elif eval_rows is None:
        eval_rows = []

    train_rows = limit_rows(train_rows, config.get("max_train_samples"))
    eval_rows = limit_rows(eval_rows, config.get("max_eval_samples"))
    train_rows = limit_rows(train_rows, args.max_train_samples)
    eval_rows = limit_rows(eval_rows, args.max_eval_samples)
    if not train_rows:
        raise ValueError("No MT training pairs found. Check transcript/translation fields and dialect filters.")

    print(f"MT train pairs: {len(train_rows)}")
    print(f"MT eval pairs: {len(eval_rows)}")

    model_name = config.get("model_name", "google/mt5-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    train_dataset = TextPairDataset(
        train_rows,
        tokenizer,
        config.get("max_source_length", 256),
        config.get("max_target_length", 256),
    )
    eval_dataset = TextPairDataset(
        eval_rows,
        tokenizer,
        config.get("max_source_length", 256),
        config.get("max_target_length", 256),
    ) if eval_rows else None

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.get("output_dir", "outputs/mt5-hunza-clean"),
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 1e-4),
        warmup_steps=config.get("warmup_steps", 200),
        num_train_epochs=args.num_train_epochs or config.get("num_train_epochs", 10),
        max_steps=args.max_steps if args.max_steps is not None else config.get("max_steps", -1),
        **evaluation_strategy_arg(Seq2SeqTrainingArguments, "steps" if eval_dataset else "no"),
        eval_steps=config.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=config.get("save_steps", 200),
        save_total_limit=2,
        predict_with_generate=True,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = str(config.get("output_dir", "outputs/mt5-hunza-clean")) + "_final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final mT5 model to {final_dir}")
    hub_model_id = args.hub_model_id or config.get("hub_model_id")
    if args.push_to_hub:
        if not hub_model_id:
            raise ValueError("Set --hub-model-id or hub_model_id in the config before pushing to Hugging Face")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        print(f"Pushed mT5 model to Hugging Face: {hub_model_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mt5_clean.yaml")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--push-to-hub", action="store_true", default=False)
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    main(parser.parse_args())
