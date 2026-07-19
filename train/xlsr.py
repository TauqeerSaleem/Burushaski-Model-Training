import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from dotenv import load_dotenv
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from data.loader import load_dataset
from data.audio_io import load_audio_16k

load_dotenv()


def read_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def keep_dialects(dataset, dialects):
    wanted = {d.lower() for d in dialects or []}
    if not wanted:
        return dataset
    return dataset.filter(lambda row: str(row.get("dialect", "")).lower() in wanted)


def prepare_ctc_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(" ", "|")


def build_vocab(dataset, text_field: str, output_dir: Path) -> Path:
    chars = set()
    for row in dataset:
        chars.update(prepare_ctc_text(row.get(text_field) or row.get("transcript") or ""))

    vocab = {char: idx for idx, char in enumerate(sorted(chars))}
    for token in ["[UNK]", "[PAD]"]:
        if token not in vocab:
            vocab[token] = len(vocab)

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    return vocab_path


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, text_field, max_duration_seconds):
        self.data = dataset
        self.processor = processor
        self.text_field = text_field
        self.max_duration_seconds = max_duration_seconds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        max_samples = int(self.max_duration_seconds * 16000)
        audio = load_audio_16k(row["audio"])[:max_samples]
        text = prepare_ctc_text(row.get(self.text_field) or row.get("transcript") or "")

        inputs = self.processor(audio, sampling_rate=16000)
        labels = self.processor.tokenizer(text).input_ids
        return {
            "input_values": inputs.input_values[0],
            "labels": labels,
        }


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def main(args):
    config = read_config(args.config)
    os.environ.setdefault("WANDB_PROJECT", config.get("wandb_project", "asr_xlsr"))

    train_data = load_dataset(
        task="asr",
        split=config.get("train_split", "train"),
        use_hf=args.use_hf,
        use_supabase=args.use_supabase,
        use_mdc=False,
    )
    eval_data = load_dataset(
        task="asr",
        split=config.get("eval_split", "test"),
        use_hf=args.use_hf,
        use_supabase=False,
        use_mdc=False,
    )

    train_data = keep_dialects(train_data, config.get("target_dialects"))
    eval_data = keep_dialects(eval_data, config.get("target_dialects"))
    if len(train_data) == 0:
        raise ValueError("No ASR training rows found. XLS-R needs audio plus Burushaski transcript.")

    output_dir = Path(config.get("output_dir", "outputs/asr-xlsr"))
    text_field = config.get("normalized_text_field") or config.get("text_field", "transcript")
    vocab_path = build_vocab(train_data, text_field, output_dir)

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = Wav2Vec2ForCTC.from_pretrained(
        config.get("model_name", "facebook/wav2vec2-xls-r-300m"),
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
    )
    model.freeze_feature_encoder()

    train_dataset = ASRDataset(
        train_data,
        processor,
        text_field,
        config.get("max_duration_seconds", 30),
    )
    eval_dataset = ASRDataset(
        eval_data,
        processor,
        text_field,
        config.get("max_duration_seconds", 30),
    ) if len(eval_data) else None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("batch_size", 8),
        per_device_eval_batch_size=config.get("batch_size", 8),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        learning_rate=config.get("learning_rate", 3e-5),
        warmup_steps=config.get("warmup_steps", 200),
        num_train_epochs=config.get("num_train_epochs", 10),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=config.get("save_steps", 200),
        save_total_limit=2,
        fp16=args.fp16,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorCTCWithPadding(processor),
        tokenizer=processor,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = str(output_dir) + "_final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Saved final XLS-R ASR model to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/asr_xlsr.yaml")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-supabase", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--resume-from-checkpoint", default=None)
    main(parser.parse_args())
