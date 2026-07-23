import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
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
    return dataset.filter(lambda row: str(row.get("dialect", "")).lower() in wanted)


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
    config = read_config(args.config)
    os.environ.setdefault("WANDB_PROJECT", config.get("wandb_project", "mt5_bsk_eng"))
    target_dialects = config.get("target_dialects")

    train_data = load_dataset(
        task="mt",
        split=config.get("train_split", "train"),
        use_hf=args.use_hf,
        use_supabase=args.use_supabase,
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
    if not train_rows:
        raise ValueError("No MT training pairs found. Check transcript/translation fields and dialect filters.")

    print(f"MT train pairs: {len(train_rows)}")
    print(f"MT eval pairs: {len(eval_rows)}")

    model_name = config.get("model_name", "google/mt5-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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
        output_dir=config.get("output_dir", "outputs/mt5-bsk-eng"),
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 1e-4),
        warmup_steps=config.get("warmup_steps", 200),
        num_train_epochs=config.get("num_train_epochs", 10),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=config.get("save_steps", 200),
        save_total_limit=2,
        predict_with_generate=True,
        fp16=args.fp16,
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
    final_dir = str(config.get("output_dir", "outputs/mt5-bsk-eng")) + "_final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final mT5 model to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mt5.yaml")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-supabase", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--resume-from-checkpoint", default=None)
    main(parser.parse_args())
