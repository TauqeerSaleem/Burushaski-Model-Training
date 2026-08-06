import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from data.audio_io import load_audio_16k
from data.loader import load_dataset
from train.whisper_st import keep_complete, limit_rows, make_target, read_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/whisper_st_micro.yaml")
    parser.add_argument("--use-hf", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--forward", action="store_true", default=False)
    args = parser.parse_args()

    config = read_config(args.config)
    rows = load_dataset("asr", config.get("train_split", "train"), args.use_hf, False, config.get("target_dialects"))
    rows = limit_rows(keep_complete(rows, config), args.max_samples)
    if not rows:
        raise ValueError("No complete Whisper speech-translation rows found")

    print(f"Rows checked: {len(rows)}")
    print("Target example:", make_target(rows[0], config).replace("\n", " | "))
    audio = load_audio_16k(rows[0]["audio"])
    print("Audio seconds:", round(len(audio) / 16000, 3))

    processor = AutoProcessor.from_pretrained(
        config.get("processor_name") or config.get("model_name", "openai/whisper-large-v3")
    )
    features = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
    labels = processor.tokenizer(
        make_target(rows[0], config),
        max_length=config.get("max_target_length", 160),
        truncation=True,
    ).input_ids
    print("Feature frames:", len(features[0]))
    print("Target tokens:", len(labels))

    if args.forward:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(config.get("model_name")).to(device)
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
        inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        label_tensor = torch.tensor([labels], device=device)
        with torch.no_grad():
            loss = model(input_features=inputs.input_features, labels=label_tensor).loss
        print("Forward loss:", round(float(loss.detach().cpu()), 4))


if __name__ == "__main__":
    main()
