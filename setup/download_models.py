from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_NAME = "openai/whisper-small"

print(f"Downloading {MODEL_NAME}...")
WhisperProcessor.from_pretrained(MODEL_NAME)
WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
print(f"Done. {MODEL_NAME} is cached and ready.")
