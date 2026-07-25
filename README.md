# Burushaski Model Training - Project Yaraan

Training code for the Burushaski speech and translation models used in Project Yaraan. Burushaski is an endangered low-resource language spoken in northern Pakistan. The current training path is deliberately modular: train ASR and text translation as separate pieces so that errors can be inspected instead of hidden inside one large black box.

Training data comes from linked project sources:
- **Hugging Face** - versioned consolidated datasets under the `Yaraan` organization
- **Supabase** - live PWA database and storage for newly collected recordings

Training runs on RunPod cloud GPUs.

![Project Yaraan training and evaluation flow](assets/pipeline_block_diagram.png)

---

## Status

This branch has the first runnable Hunza-focused XLS-R/mT5 pipeline scaffold:

- XLS-R ASR training: implemented in `train/xlsr.py`
- mT5 text translation training: implemented in `train/mt5.py`
- XLS-R, mT5, and cascade evaluation scripts: implemented in `evaluate/`
- shared data loading from HF/Supabase: implemented in `data/loader.py`

Still pending after the first GPU runs:

- stricter speaker/prompt-aware split generation
- final trained checkpoints and real evaluation scores

TTS, Whisper, MMS, and SeamlessM4T are outside this branch's current training run.

---

## Current Model Plan

Main speech-to-English pipeline:

```text
Burushaski audio -> XLS-R ASR -> Burushaski text -> mT5 -> English text
```

Reverse text pipeline for now:

```text
English text -> mT5 -> Burushaski text
```

Whisper, MMS, SeamlessM4T, and TTS are outside the current run. They can be evaluated separately later, but the main system here is the interpretable XLS-R -> mT5 cascade.

The first training phase focuses on Hunza because the consolidated dataset currently has the strongest Hunza coverage. The translation model still uses dialect tags in its prompts so Nagar and Yasin can be added later without changing the training script.

For the current PI-requested run, training uses the Hugging Face Hunza training split only. Evaluation uses the held-out Hugging Face test split plus all complete Hunza rows currently available from Supabase.

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
SUPABASE_SERVICE_ROLE_KEY=  # Supabase service role key, preferred
SUPABASE_KEY=       # Alternative name for the same service role key
WANDB_API_KEY=      # Weights & Biases API key
WANDB_PROJECT=      # Weights & Biases project name (e.g. yaraan-hunza-xlsr-mt5)
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

**5. Check data access before training**
```bash
python test_run_loader.py --task asr --split train --use-hf --dialect hunza
python test_run_loader.py --task mt --split train --use-hf --dialect hunza
python test_run_loader.py --task asr --split test --use-hf --use-supabase --dialect hunza
python test_run_loader.py --task mt --split test --use-hf --use-supabase --dialect hunza
```

**6. Run training**
```bash
# Train on the Hugging Face Hunza training split only
python train/xlsr.py --config configs/asr_xlsr.yaml --use-hf --fp16
python train/mt5.py --config configs/mt5.yaml --use-hf --fp16
```

Both configs currently filter to the Hunza dialect through `target_dialects: [hunza]`.

**7. Evaluate**
```bash
# XLS-R ASR
python evaluate/xlsr.py --model-path outputs/asr-xlsr_final --use-hf --use-supabase

# mT5 text translation
python evaluate/mt5.py --model-path outputs/mt5-bsk-eng_final --use-hf --use-supabase

# Full cascade: XLS-R transcript -> mT5 English
python evaluate/pipeline.py --asr-model-path outputs/asr-xlsr_final --mt-model-path outputs/mt5-bsk-eng_final --use-hf --use-supabase
```

Evaluation writes detailed predictions plus summary CSV, JSON, and short readable TXT files under `outputs/.../results`. If `WANDB_API_KEY` is set, the summary metrics and result files are also attached to the W&B run.

**8. Upload final checkpoints to Hugging Face**
```bash
huggingface-cli upload Yaraan/xlsr-hunza-asr-v1 outputs/asr-xlsr_final .
huggingface-cli upload Yaraan/mt5-hunza-bsk-eng-v1 outputs/mt5-bsk-eng_final .
```

For a fresh run, the training scripts can also upload automatically after training:

```bash
python train/xlsr.py --config configs/asr_xlsr.yaml --use-hf --fp16 --push-to-hub
python train/mt5.py --config configs/mt5.yaml --use-hf --fp16 --push-to-hub
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

data/
  loader.py            # entry point: load_dataset(...)
  audio_io.py          # loads local path audio and HF Audio dictionaries
  manifest.py          # builds canonical JSONL manifests when needed
  normalization.py     # light Burushaski/English text normalization
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
  xlsr.py              # fine-tune XLS-R CTC ASR
  mt5.py               # fine-tune mT5 text translation
evaluate/
  xlsr.py              # compute ASR CER/WER
  mt5.py               # compute MT chrF++/BLEU
  pipeline.py          # compare gold transcript MT vs XLS-R -> mT5 cascade

```

---

## Data

The normal training path is linked mode. Hugging Face supplies the 80/20 training/test split, while Supabase is used as an additional Hunza evaluation source.

```bash
python train/xlsr.py --use-hf --fp16
python train/mt5.py --use-hf --fp16
```

The live app stores recordings in Supabase. The training loader reads from `active_recordings` and only keeps rows that have the fields needed for the current task:

- ASR: `audio_path` + `transcript`
- MT: `transcript` + `english_translation`

Audio files are stored in Supabase Storage under `dialect/participant_id/module_id/` paths and cached locally under `cache/audio/` (or `CACHE_DIR` if set). Older app recordings may be `.m4a`; newer browser-friendly recordings may be `.webm`.

The Hugging Face consolidated dataset is configured in `configs/datasets.yaml`. Supabase data is not used for training in the current run. It is added during evaluation so the final test report covers the HF held-out split and all complete Hunza app rows.

The old translation workbook is text-only. If those rows are needed for RunPod training, fold them into the Hugging Face consolidated dataset first rather than reading an `.xlsx` from a local machine.

To export a task-specific manifest:

```bash
python data/manifest.py --task mt --split train --use-hf --use-supabase --dialect hunza --output data/manifests/train_hunza_mt.jsonl
```

---

## Evaluation

The evaluation scripts run after a checkpoint exists. They do not contain fixed results or placeholder scores.

ASR is evaluated with both raw and normalized:

- `WER`: word error rate
- `CER`: character error rate

CER is especially important here because Burushaski spelling is not fully standardized, and character-level mistakes are more informative than only counting whole-word errors. The evaluator also writes group summaries by dialect, participant, and gender.

Text translation and speech-to-English outputs are evaluated with:

- `chrF++`
- `BLEU`

`chrF++` is useful for low-resource and spelling-variable settings because it gives credit for character n-gram overlap, while BLEU remains a common comparison point in MT and speech-translation papers.

The mT5 evaluator writes summaries by translation direction, dialect, and source.

The cascade evaluator reports two systems on the same test set:

```text
gold Burushaski transcript -> mT5 -> English
XLS-R predicted transcript -> mT5 -> English
```

This shows how much performance is lost because of ASR errors.

The cascade evaluator also writes an ASR summary with normalized WER/CER for the XLS-R transcript used inside the pipeline.

---

## Known Limitations

- **HuggingFace private access** - set `HF_TOKEN` before using `--use-hf`.
- **Supabase `--use-supabase` flag** - use a service role key in `.env`. Do not paste it into notebooks or logs.
- **Splits still need tightening** - the next step is speaker/prompt-aware splitting. Do not treat random clip-level splits as final research results.
- **Current implementation is training/evaluation-ready, not result-ready** - run smoke tests on GPU first, then report only the metrics produced from trained checkpoints.
