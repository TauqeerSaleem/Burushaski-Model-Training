import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import (
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
)

from models.registry import read_registry


BASE_MODELS = {
    "xlsr_base": "facebook/wav2vec2-xls-r-300m",
    "mt5_base": "google/mt5-base",
    "whisper_st_start": "Yaraan/bsk-eng-stt-translate",
}


def cache_model(name, kind, model_id, processor_id=None, base_model=None):
    print(f"Downloading {name}: {model_id}")
    if kind == "ctc_base":
        AutoFeatureExtractor.from_pretrained(model_id)
        AutoModelForCTC.from_pretrained(model_id)
    elif kind in {"ctc", "mms_ctc"}:
        AutoProcessor.from_pretrained(processor_id or model_id)
        AutoModelForCTC.from_pretrained(model_id)
    elif kind == "whisper":
        AutoProcessor.from_pretrained(processor_id or base_model or model_id)
        AutoModelForSpeechSeq2Seq.from_pretrained(base_model or model_id)
    elif kind == "seq2seq":
        AutoTokenizer.from_pretrained(model_id)
        AutoModelForSeq2SeqLM.from_pretrained(model_id)
    else:
        print(f"Skipping {name}: {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/model_registry.yaml")
    parser.add_argument("--include-trained", action="store_true", default=False)
    args = parser.parse_args()

    cache_model("xlsr_base", "ctc_base", BASE_MODELS["xlsr_base"])
    cache_model("mt5_base", "seq2seq", BASE_MODELS["mt5_base"])
    cache_model("whisper_st_start", "whisper", BASE_MODELS["whisper_st_start"])

    if args.include_trained:
        registry = read_registry(args.registry)
        for name, spec in registry.get("asr", {}).items():
            cache_model(name, spec["kind"], spec["model_id"], spec.get("processor_id"), spec.get("base_model"))

    print("Done. Requested models are cached and ready.")


if __name__ == "__main__":
    main()
