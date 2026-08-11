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
        if spec.get("source_lang") and hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = spec["source_lang"]
        if spec.get("target_lang") and hasattr(self.tokenizer, "tgt_lang"):
            self.tokenizer.tgt_lang = spec["target_lang"]
        self.model = AutoModelForSeq2SeqLM.from_pretrained(spec["model_id"]).to(device)
        self.forced_bos_token_id = self._forced_bos_token_id()
        if self.forced_bos_token_id is not None:
            self.model.config.forced_bos_token_id = self.forced_bos_token_id
        self.model.eval()

    def _forced_bos_token_id(self):
        target_lang = self.spec.get("target_lang")
        if not target_lang:
            return None
        lang_code_to_id = getattr(self.tokenizer, "lang_code_to_id", None)
        if lang_code_to_id and target_lang in lang_code_to_id:
            return lang_code_to_id[target_lang]
        token_id = self.tokenizer.convert_tokens_to_ids(target_lang)
        return None if token_id == self.tokenizer.unk_token_id else token_id

    def translate_bsk_to_eng(self, text, dialect="hunza", max_length=128, num_beams=4):
        source = f"translate bsk_{dialect} to eng: {text}"
        inputs = self.tokenizer(source, return_tensors="pt", max_length=max_length, truncation=True).to(self.device)
        generate_kwargs = {"max_length": max_length, "num_beams": num_beams}
        if self.forced_bos_token_id is not None:
            generate_kwargs["forced_bos_token_id"] = self.forced_bos_token_id
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
