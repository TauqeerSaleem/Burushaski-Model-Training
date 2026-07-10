# Burushaski Machine Translation — Project Yaraan

Fine-tuning machine translation models for Burushaski, an endangered low-resource language spoken in northern Pakistan. The project targets Burushaski ↔ English translation using models such as Whisper, mT5, and others to be evaluated.

Training data comes from two sources:
- A custom dataset hosted on Hugging Face (versioned, with train/test/sample splits)
- A live Supabase database of recordings being actively collected and transcribed (treated as additional training data)

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
WANDB_API_KEY=      # Weights & Biases
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

**5. Test data loading**
```bash
python test_run_loader.py
```

**6. Run training**
```bash
# (scripts to be added per model)
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

train/                 # training scripts per model (to be added)
evaluate/              # evaluation scripts per model (to be added)

Old_Model/             # historical notebooks (Whisper ASR, HuBERT TTS) — reference only
```

---

## Data

The `recordings` table in Supabase has 245+ rows. Only rows with `audio_path`, `transcript`, and `english_translation` all filled in are loaded — as transcription is ongoing, this number will grow.

Audio files are `.m4a`, stored in Supabase Storage under `dialect/participant_id/module_id/` paths and cached locally under `cache/audio/` (or `CACHE_DIR` if set).

The HuggingFace dataset has `train`, `test`, and `sample` splits. Supabase data is always merged into `train` only.
