# Burushaski Model Training - Project Yaraan

Training code for the Burushaski speech and translation models used in Project Yaraan. Burushaski is an endangered low-resource language spoken in northern Pakistan. The main plan is deliberately modular: train ASR, text translation, and TTS as separate pieces so that errors can be inspected instead of hidden inside one large black box.

Training data comes from linked project sources:
- **Hugging Face** - versioned consolidated datasets under the `Yaraan` organization
- **Supabase** - live PWA database and storage for newly collected recordings
- **Mozilla Data Collective (MDC)** - optional speech-to-English baseline source

Training runs on RunPod cloud GPUs.

![Project Yaraan training and evaluation flow](assets/pipeline_block_diagram.png)

---

## Status

This branch has the first runnable training pipeline scaffold:

- XLS-R ASR training: implemented in `train/xlsr.py`
- mT5 text translation training: implemented in `train/mt5.py`
- Whisper direct speech-to-English baseline: implemented in `train/whisper.py`
- XLS-R, mT5, Whisper, and cascade evaluation scripts: implemented in `evaluate/`
- shared data loading from HF/Supabase/MDC: implemented in `data/loader.py`

Still pending after the first GPU runs:

- TTS training script
- stricter speaker/prompt-aware split generation

---

## Current Model Plan

Main speech-to-English pipeline:

```text
Burushaski audio -> XLS-R ASR -> Burushaski text -> mT5 -> English text
```

Reverse pipeline:

```text
English text -> mT5 -> Burushaski text -> Burushaski TTS -> Burushaski audio
```

Whisper is kept as a direct speech-to-English baseline. SeamlessM4T is worth testing later, but it should not be the first system because Burushaski is not a standard supported language and debugging hidden ASR/translation mistakes will be harder.

The first training phase focuses on Hunza because the consolidated dataset currently has the strongest Hunza coverage. The translation model still uses dialect tags in its prompts so Nagar and Yasin can be added later without changing the training script.

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
# Main RunPod/Kaggle path: read the consolidated private dataset from Hugging Face
python train/whisper.py --use-hf --use-supabase --epochs 5
python train/xlsr.py --config configs/asr_xlsr.yaml --use-hf --use-supabase --fp16
python train/mt5.py --config configs/mt5.yaml --use-hf --use-supabase --fp16

# MDC-only direct speech-to-English baseline, useful if HF is not ready yet
python train/whisper.py --use-mdc --epochs 5
```

**6. Evaluate**
```bash
# Whisper direct speech-to-English baseline
python evaluate/whisper.py --model-path outputs/whisper_final --use-hf

# XLS-R ASR
python evaluate/xlsr.py --model-path outputs/asr-xlsr_final --use-hf

# mT5 text translation
python evaluate/mt5.py --model-path outputs/mt5-bsk-eng_final --use-hf

# Full cascade: XLS-R transcript -> mT5 English
python evaluate/pipeline.py --asr-model-path outputs/asr-xlsr_final --mt-model-path outputs/mt5-bsk-eng_final --use-hf
```

---

## Hugging Face Access

Private Hugging Face datasets cannot be cloned with your account password. Use a user access token.

On RunPod or Kaggle, the easiest path is usually:

```bash
huggingface-cli login
```

Paste a token from Hugging Face settings. For this repo, set the same token in `.env` too:

```text
HF_TOKEN=hf_...
```

If Windows Git Credential Manager keeps opening a dialog when cloning locally, set the provider manually:

```powershell
git config --global credential.https://huggingface.co.provider generic
```

Then clone with a token when Git asks for a password. Username can be your HF username; password is the token.

You do not need to clone the dataset for normal training if `HF_TOKEN` is set. The training scripts use the Hugging Face `datasets` library and read the configured repo directly.

---

## Project Structure

```
configs/
  datasets.yaml        # HF repo IDs for linked dataset loading
  asr_xlsr.yaml        # XLS-R ASR settings
  mt5.yaml             # mT5 translation settings
  whisper_s2tt.yaml    # Whisper speech-to-English baseline settings

data/
  loader.py            # entry point: load_dataset(...)
  audio_io.py          # loads local path audio and HF Audio dictionaries
  manifest.py          # builds canonical JSONL manifests when needed
  normalization.py     # light Burushaski/English text normalization
  sources/
    hf.py              # loads from HuggingFace Hub
    supabase.py        # loads from Supabase recordings table
    mdc.py             # loads Mozilla Data Collective when enabled
  storage/
    audio.py           # downloads audio from Supabase Storage, caches locally

setup/
  setup.sh             # RunPod bootstrap script
  download_models.py   # pre-downloads model weights
  verify_environment.py  # checks CUDA, GPU memory

train/
  whisper.py           # fine-tune Whisper direct speech-to-English baseline
  xlsr.py              # fine-tune XLS-R CTC ASR
  mt5.py               # fine-tune mT5 text translation
evaluate/
  whisper.py           # compute BLEU, chrF++, BERTScore, WER, CER
  xlsr.py              # compute ASR CER/WER
  mt5.py               # compute MT chrF++/BLEU
  pipeline.py          # compare gold transcript MT vs XLS-R -> mT5 cascade

```

---

## Data

The normal training path is linked mode: Hugging Face for the consolidated dataset and Supabase for newly collected complete PWA rows.

```bash
python train/xlsr.py --use-hf --use-supabase --fp16
python train/mt5.py --use-hf --use-supabase --fp16
```

The live app stores recordings in Supabase. The training loader reads from `active_recordings` and only keeps rows that have the fields needed for the current task:

- ASR: `audio_path` + `transcript`
- MT: `transcript` + `english_translation`
- speech-to-text translation: `audio_path` + `english_translation`
- TTS: `audio_path` + `transcript`

Audio files are `.m4a`, stored in Supabase Storage under `dialect/participant_id/module_id/` paths and cached locally under `cache/audio/` (or `CACHE_DIR` if set).

The Hugging Face consolidated dataset is configured in `configs/datasets.yaml`. Supabase data is merged into `train` only so that live app data does not accidentally leak into the held-out HF test split.

MDC is useful for direct speech-to-English experiments because it has Burushaski audio paired with English text. True ASR needs Burushaski transcripts, so use the consolidated HF/Supabase rows for XLS-R.

The old translation workbook is text-only. If those rows are needed for RunPod training, fold them into the Hugging Face consolidated dataset first rather than reading an `.xlsx` from a local machine.

To export a task-specific manifest:

```bash
python data/manifest.py --task mt --split train --use-hf --use-supabase --dialect hunza --output data/manifests/train_hunza_mt.jsonl
```

---

## Evaluation

The evaluation scripts run after a checkpoint exists. They do not contain fixed results or placeholder scores.

ASR is evaluated with:

- `WER`: word error rate
- `CER`: character error rate

CER is especially important here because Burushaski spelling is not fully standardized, and character-level mistakes are more informative than only counting whole-word errors.

Text translation and speech-to-English outputs are evaluated with:

- `chrF++`
- `BLEU`

`chrF++` is useful for low-resource and spelling-variable settings because it gives credit for character n-gram overlap, while BLEU remains a common comparison point in MT and speech-translation papers.

The cascade evaluator reports two systems on the same test set:

```text
gold Burushaski transcript -> mT5 -> English
XLS-R predicted transcript -> mT5 -> English
```

This shows how much performance is lost because of ASR errors.

---

## Known Limitations

- **HuggingFace private access** - set `HF_TOKEN` before using `--use-hf`.
- **MDC download requires Terms acceptance** - visit your dataset page on `mozilladatacollective.com` while logged in and accept the terms before the MDC source will work.
- **Supabase `--use-supabase` flag** - use a service role key in `.env`. Do not paste it into notebooks or logs.
- **Splits still need tightening** - the next step is speaker/prompt-aware splitting. Do not treat random clip-level splits as final research results.
- **Current implementation is training/evaluation-ready, not result-ready** - run smoke tests on GPU first, then report only the metrics produced from trained checkpoints.
