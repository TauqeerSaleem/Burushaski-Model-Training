import argparse
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from dotenv import load_dotenv
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data.audio_io import load_audio_16k
from data.loader import load_dataset

load_dotenv()


def read_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def limit_rows(rows, limit):
    if limit is None:
        return rows
    limit = int(limit)
    if limit <= 0:
        return rows
    return rows[:limit]


def evaluation_strategy_arg(arguments_cls, value):
    parameters = inspect.signature(arguments_cls.__init__).parameters
    key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    return {key: value}


def make_target(row, config):
    bsk = str(row.get(config.get("bsk_text_field", "bsk_text_normalized")) or row.get("transcript") or "").strip()
    eng = str(row.get(config.get("english_field", "english_translation")) or row.get("english_text") or "").strip()
    if not bsk or not eng:
        return None
    if config.get("target_mode", "bsk_and_eng") == "english_only":
        return eng
    return f"Burushaski: {bsk}\nEnglish: {eng}"


def keep_complete(rows, config):
    kept = []
    for row in rows:
        if row.get("audio") and make_target(row, config):
            kept.append(row)
    return kept


class WhisperSpeechTranslationDataset(torch.utils.data.Dataset):
    def __init__(self, rows, processor, config):
        self.rows = rows
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        max_samples = int(self.config.get("max_audio_seconds", 20) * 16000)
        audio = load_audio_16k(row["audio"])[:max_samples]
        features = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        labels = self.processor.tokenizer(
            make_target(row, self.config),
            max_length=self.config.get("max_target_length", 160),
            truncation=True,
        ).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": item["input_features"]} for item in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": item["labels"]} for item in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def main(args):
    if args.fp16 and args.bf16:
        raise ValueError("Choose either --fp16 or --bf16, not both")

    config = read_config(args.config)
    os.environ.setdefault("WANDB_PROJECT", config.get("wandb_project", "whisper_bsk_eng"))
    dialects = config.get("target_dialects")

    train_rows = load_dataset("asr", config.get("train_split", "train"), args.use_hf, False, dialects)
    eval_rows = load_dataset("asr", config.get("eval_split", "test"), args.use_hf, False, dialects)
    train_rows = limit_rows(keep_complete(train_rows, config), config.get("max_train_samples"))
    eval_rows = limit_rows(keep_complete(eval_rows, config), config.get("max_eval_samples"))
    if not train_rows:
        raise ValueError("No Whisper speech-translation training rows found")

    print(f"Whisper ST train rows: {len(train_rows)}")
    print(f"Whisper ST eval rows: {len(eval_rows)}")
    print("Target example:", make_target(train_rows[0], config).replace("\n", " | "))

    model_name = config.get("model_name", "openai/whisper-large-v3")
    processor_name = config.get("processor_name", model_name)
    processor = AutoProcessor.from_pretrained(processor_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    train_dataset = WhisperSpeechTranslationDataset(train_rows, processor, config)
    eval_dataset = WhisperSpeechTranslationDataset(eval_rows, processor, config) if eval_rows else None

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.get("output_dir", "outputs/whisper-bsk-eng"),
        per_device_train_batch_size=config.get("batch_size", 1),
        per_device_eval_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 1e-5),
        warmup_steps=config.get("warmup_steps", 100),
        num_train_epochs=config.get("num_train_epochs", 2),
        max_steps=config.get("max_steps", -1),
        **evaluation_strategy_arg(Seq2SeqTrainingArguments, "steps" if eval_dataset else "no"),
        eval_steps=config.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=config.get("save_steps", 500),
        save_total_limit=2,
        predict_with_generate=True,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WhisperDataCollator(processor),
        tokenizer=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = str(config.get("output_dir", "outputs/whisper-bsk-eng")) + "_final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Saved final Whisper speech-translation model to {final_dir}")

    hub_model_id = args.hub_model_id or config.get("hub_model_id")
    if args.push_to_hub:
        if not hub_model_id:
            raise ValueError("Set --hub-model-id or hub_model_id in the config before pushing to Hugging Face")
        model.push_to_hub(hub_model_id)
        processor.push_to_hub(hub_model_id)
        print(f"Pushed Whisper speech-translation model to Hugging Face: {hub_model_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/whisper_st_short.yaml")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--push-to-hub", action="store_true", default=False)
    parser.add_argument("--hub-model-id", default=None)
    main(parser.parse_args())
