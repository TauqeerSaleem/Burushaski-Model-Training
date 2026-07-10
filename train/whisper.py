import argparse
import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import torchaudio
from dotenv import load_dotenv
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from data.loader import load_dataset

load_dotenv()

os.environ.setdefault("WANDB_PROJECT", os.getenv("WANDB_PROJECT", "whisper-v1"))

MODEL_NAME = "openai/whisper-small"


class BurushaskiDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, processor):
        self.data = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        waveform, sr = torchaudio.load(row["audio"])
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        audio = waveform.squeeze().numpy()
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        labels = self.processor.tokenizer(row["english_translation"], return_tensors="pt").input_ids
        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels.squeeze(0),
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main(args):
    print(f"Loading train data (hf={args.use_hf}, supabase={args.use_supabase}, mdc={args.use_mdc})...")
    train_hf = load_dataset(
        split="train",
        use_hf=args.use_hf,
        use_supabase=args.use_supabase,
        use_mdc=args.use_mdc,
    )
    print(f"Total train examples: {len(train_hf)}")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    full_dataset = BurushaskiDataset(train_hf, processor)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_data, eval_data = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=200,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=False,
        generation_max_length=225,
        logging_steps=25,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        processing_class=processor,
    )

    checkpoint = None
    if os.path.isdir(args.output_dir) and any("checkpoint" in f for f in os.listdir(args.output_dir)):
        checkpoint = args.output_dir

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    final_dir = args.output_dir + "_final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="outputs/whisper")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--use-supabase", action="store_true", default=False)
    parser.add_argument("--use-mdc", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
