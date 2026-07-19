from io import BytesIO

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio_16k(audio):
    """Load audio from a path or a Hugging Face Audio-style dictionary."""
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            waveform = torch.tensor(np.asarray(audio["array"]), dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            sr = int(audio.get("sampling_rate") or 16000)
        elif audio.get("bytes") is not None:
            array, sr = sf.read(BytesIO(audio["bytes"]), dtype="float32")
            waveform = torch.tensor(array, dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)
        elif audio.get("path"):
            waveform, sr = torchaudio.load(audio["path"])
        else:
            raise ValueError("Audio dictionary does not contain array, bytes, or path")
    else:
        waveform, sr = torchaudio.load(audio)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform.squeeze(0).numpy()
