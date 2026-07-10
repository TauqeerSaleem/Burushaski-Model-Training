# Burushaski Machine Translation — Project Yaraan

Fine-tuning machine translation models for Burushaski, an endangered low-resource language spoken in northern Pakistan. The project targets Burushaski ↔ English translation using models such as Whisper, mT5, and others to be evaluated.

Training data comes from three sources:
- **Mozilla Data Collective (MDC)** — primary working source; audio + English translation pairs with train/test/sample splits
- **Supabase** — live database of recordings being actively collected and transcribed; currently intermittent API issues
- **Hugging Face** — versioned dataset with train/test/sample splits; not yet configured (repo IDs pending)

Training runs on RunPod cloud GPUs.

---

## RunPod Setup

When you clone this repo onto a new RunPod pod, run these in order:

**1. Install dependencies**
```bash
bash setup/setup.sh
```
This installs pip packages, creates `.env` from the template, and validates that all required env vars are set.

**2. Fill in `.env`**
```
HF_TOKEN=           # Hugging Face token (for private dataset access)
SUPABASE_URL=       # Supabase project URL
SUPABASE_KEY=       # Supabase service role key (not anon)
WANDB_API_KEY=      # Weights & Biases API key
WANDB_PROJECT=      # Weights & Biases project name (e.g. whisper-v1)
MDC_API_KEY=        # Mozilla Data Collective API key
MDC_DATASET_ID=     # MDC dataset ID
CACHE_DIR=          # Optional: set to network volume path to persist audio cache across pods
```

**3. Download base models**
```bash
python setup/download_models.py
```
Pre-downloads model weights so training doesn't fetch them mid-run.

**4. Verify GPU and environment**
```bash
python setup/verify_environment.py
```

**5. Run training**
```bash
python train/whisper.py --use-supabase --use-mdc --epochs 5
# add --use-hf once HuggingFace dataset repo IDs are filled in configs/datasets.yaml
```

**7. Evaluate**
```bash
python evaluate/whisper.py --model-path outputs/whisper_final --use-hf
```

---

## Project Structure

```
configs/
  datasets.yaml        # HuggingFace repo IDs per task (asr, mt, tts)
  models.yaml          # model names and hyperparameters per task (to be added)

data/
  loader.py            # entry point: load_dataset(task, split, use_hf, use_supabase)
  sources/
    hf.py              # loads from HuggingFace Hub
    supabase.py        # loads from Supabase recordings table
  storage/
    audio.py           # downloads audio from Supabase Storage, caches locally

setup/
  setup.sh             # RunPod bootstrap script
  download_models.py   # pre-downloads model weights
  verify_environment.py  # checks CUDA, GPU memory

train/
  whisper.py           # fine-tune Whisper (Burushaski → English)
evaluate/
  whisper.py           # compute BLEU, chrF++, BERTScore, WER, CER

Old_Model/             # historical notebooks (Whisper ASR, HuBERT TTS) — reference only
```

---

## Data

The `recordings` table in Supabase has 245+ rows. Only rows with `audio_path`, `transcript`, and `english_translation` all filled in are loaded — as transcription is ongoing, this number will grow.

Audio files are `.m4a`, stored in Supabase Storage under `dialect/participant_id/module_id/` paths and cached locally under `cache/audio/` (or `CACHE_DIR` if set).

The HuggingFace dataset has `train`, `test`, and `sample` splits. Supabase data is always merged into `train` only.

---

## Known Limitations

- **HuggingFace source disabled by default** — `configs/datasets.yaml` has empty repo IDs. Fill these in before using the `--use-hf` flag, otherwise training will crash.
- **MDC download requires Terms acceptance** — visit your dataset page on `mozilladatacollective.com` while logged in and accept the terms before the MDC source will work.
- **Supabase `--use-supabase` flag** — if you get a 401 error despite a valid service role key, this is a transient Supabase API issue. Retry or disable the flag and train with MDC/HF only.
