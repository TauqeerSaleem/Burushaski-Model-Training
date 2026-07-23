from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
)

MODELS = {
    "xlsr": "facebook/wav2vec2-xls-r-300m",
    "mt5": "google/mt5-base",
}

print(f"Downloading {MODELS['xlsr']}...")
Wav2Vec2FeatureExtractor.from_pretrained(MODELS["xlsr"])
Wav2Vec2ForCTC.from_pretrained(MODELS["xlsr"])

print(f"Downloading {MODELS['mt5']}...")
AutoTokenizer.from_pretrained(MODELS["mt5"])
AutoModelForSeq2SeqLM.from_pretrained(MODELS["mt5"])

print("Done. Base models are cached and ready.")
