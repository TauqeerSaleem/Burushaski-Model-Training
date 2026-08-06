import torch
from transformers import AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor


class ASRModel:
    def __init__(self, name, spec, device):
        self.name = name
        self.spec = spec
        self.kind = spec["kind"]
        self.device = device
        self.processor = AutoProcessor.from_pretrained(spec.get("processor_id") or spec.get("base_model") or spec["model_id"])

        if self.kind in {"ctc", "mms_ctc"}:
            self.model = AutoModelForCTC.from_pretrained(spec["model_id"]).to(device)
        elif self.kind == "whisper":
            self.model = _load_whisper_model(spec, device)
        else:
            raise ValueError(f"Unsupported ASR model kind: {self.kind}")
        self.model.eval()

    def transcribe(self, audio_16k):
        if self.kind in {"ctc", "mms_ctc"}:
            inputs = self.processor(audio_16k, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(predicted_ids)[0]
            return text.replace("|", " ").strip()

        inputs = self.processor(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        generate_kwargs = self.spec.get("generate_kwargs") or {}
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, **generate_kwargs)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def _load_whisper_model(spec, device):
    model_id = spec["model_id"]
    base_model = spec.get("base_model")
    if base_model:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Install peft to load Whisper LoRA adapters") from exc
        model = AutoModelForSpeechSeq2Seq.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, model_id)
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    return model.to(device)
