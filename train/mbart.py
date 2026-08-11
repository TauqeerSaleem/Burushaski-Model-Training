import argparse
import inspect
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

from train.mt5 import TextPairDataset, limit_rows, load_processed_mt_rows

load_dotenv()


def read_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluation_strategy_arg(arguments_cls, value):
    parameters = inspect.signature(arguments_cls.__init__).parameters
    key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    return {key: value}


def prepare_tokenizer(tokenizer, config):
    src_lang = config.get("source_lang")
    tgt_lang = config.get("target_lang", "en_XX")
    if src_lang and hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang
    if tgt_lang and hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = tgt_lang
    return tokenizer


def forced_bos_token_id(tokenizer, config):
    target_lang = config.get("target_lang", "en_XX")
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if lang_code_to_id and target_lang in lang_code_to_id:
        return lang_code_to_id[target_lang]
    token_id = tokenizer.convert_tokens_to_ids(target_lang)
    return None if token_id == tokenizer.unk_token_id else token_id


def main(args):
    if args.fp16 and args.bf16:
        raise ValueError("Choose either --fp16 or --bf16, not both")

    config = read_config(args.config)
    os.environ.setdefault("WANDB_PROJECT", config.get("wandb_project", "yaraan-hunza-mbart"))

    train_rows = load_processed_mt_rows(config, "train")
    eval_rows = load_processed_mt_rows(config, config.get("eval_split", "validation"))
    train_rows = limit_rows(train_rows, config.get("max_train_samples"))
    eval_rows = limit_rows(eval_rows, config.get("max_eval_samples"))
    train_rows = limit_rows(train_rows, args.max_train_samples)
    eval_rows = limit_rows(eval_rows, args.max_eval_samples)
    if not train_rows:
        raise ValueError("No mBART training rows found. Run analysis/build_clean_mt_dataset.py first.")

    print(f"mBART train pairs: {len(train_rows)}")
    print(f"mBART eval pairs: {len(eval_rows)}")

    model_name = config.get("model_name", "facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(model_name), config)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.config.use_cache = False
    bos_id = forced_bos_token_id(tokenizer, config)
    if bos_id is not None:
        model.config.forced_bos_token_id = bos_id
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    train_dataset = TextPairDataset(
        train_rows,
        tokenizer,
        config.get("max_source_length", 128),
        config.get("max_target_length", 128),
    )
    eval_dataset = TextPairDataset(
        eval_rows,
        tokenizer,
        config.get("max_source_length", 128),
        config.get("max_target_length", 128),
    ) if eval_rows else None

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.get("output_dir", "outputs/mbart-hunza-bsk-eng"),
        per_device_train_batch_size=config.get("batch_size", 1),
        per_device_eval_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 3e-5),
        warmup_steps=config.get("warmup_steps", 50),
        num_train_epochs=args.num_train_epochs or config.get("num_train_epochs", 5),
        max_steps=args.max_steps if args.max_steps is not None else config.get("max_steps", -1),
        **evaluation_strategy_arg(Seq2SeqTrainingArguments, "steps" if eval_dataset else "no"),
        eval_steps=config.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=config.get("save_steps", 100),
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
    final_dir = str(config.get("output_dir", "outputs/mbart-hunza-bsk-eng")) + "_final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final mBART model to {final_dir}")

    hub_model_id = args.hub_model_id or config.get("hub_model_id")
    if args.push_to_hub:
        if not hub_model_id:
            raise ValueError("Set --hub-model-id or hub_model_id in the config before pushing to Hugging Face")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        print(f"Pushed mBART model to Hugging Face: {hub_model_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mbart_clean.yaml")
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
