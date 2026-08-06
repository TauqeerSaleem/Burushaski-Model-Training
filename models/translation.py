import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TextTranslator:
    def __init__(self, name, spec, device):
        self.name = name
        self.spec = spec
        self.kind = spec["kind"]
        self.device = device
        if self.kind != "seq2seq":
            raise ValueError(
                f"{name} is not a text-to-text translator. "
                "Whisper models take audio input, so they cannot be used for transcript-text -> English cascades."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(spec["model_id"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(spec["model_id"]).to(device)
        self.model.eval()

    def translate_bsk_to_eng(self, text, dialect="hunza", max_length=128, num_beams=4):
        source = f"translate bsk_{dialect} to eng: {text}"
        inputs = self.tokenizer(source, return_tensors="pt", max_length=max_length, truncation=True).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_length=max_length, num_beams=num_beams)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

