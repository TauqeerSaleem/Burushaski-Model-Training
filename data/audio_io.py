from io import BytesIO

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio_16k(audio):
    """Load audio into a mono NumPy array sampled at 16 kHz."""
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        waveform = torch.as_tensor(samples.data, dtype=torch.float32)
        sr = int(samples.sample_rate)
    elif isinstance(audio, dict):
        if audio.get("array") is not None:
            waveform = torch.as_tensor(np.asarray(audio["array"]), dtype=torch.float32)
            sr = int(audio.get("sampling_rate") or 16000)
        elif audio.get("bytes") is not None:
            array, sr = sf.read(BytesIO(audio["bytes"]), dtype="float32")
            waveform = torch.as_tensor(array, dtype=torch.float32)
        elif audio.get("path"):
            waveform, sr = torchaudio.load(audio["path"])
        else:
            raise ValueError("Audio dictionary does not contain array, bytes, or path")
    elif isinstance(audio, bytes):
        array, sr = sf.read(BytesIO(audio), dtype="float32")
        waveform = torch.as_tensor(array, dtype=torch.float32)
    elif isinstance(audio, np.ndarray):
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        sr = 16000
    elif torch.is_tensor(audio):
        waveform = audio.detach().to(dtype=torch.float32).cpu()
        sr = 16000
    else:
        waveform, sr = torchaudio.load(str(audio))

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.transpose(0, 1)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    audio_array = waveform.squeeze(0).numpy()
    if not np.isfinite(audio_array).all():
        raise ValueError("Decoded audio contains NaN or infinite values")
    return audio_array
