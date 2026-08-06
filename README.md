# Burushaski Model Training - Project Yaraan

Training code for the Burushaski speech and translation models used in Project Yaraan. The current training path keeps ASR and translation separate so each part can be checked properly.

Training data comes from linked project sources:
- **Hugging Face** - versioned consolidated datasets under the `Yaraan` organization
- **Supabase** - live PWA database and storage for newly collected recordings

Training runs on RunPod.

![Project Yaraan training and evaluation flow](assets/pipeline_block_diagram.png)

---

## Scope

This branch contains the Hunza-focused ASR, translation, and cascade evaluation pipeline:

- XLS-R ASR training: implemented in `train/xlsr.py`
- mT5 text translation training: implemented in `train/mt5.py`
- Whisper direct speech-to-English training: implemented in `train/whisper_st.py`
- ASR comparison for XLS-R, MMS, and Whisper checkpoints: implemented in `evaluate/asr_all.py`
- mT5 and cascade evaluation scripts: implemented in `evaluate/`
- shared data loading from HF/Supabase: implemented in `data/loader.py`

TTS and SeamlessM4T are outside this branch's current training run.

---

## Current Model Plan

Main speech-to-English cascade:

```text
Burushaski audio -> XLS-R/MMS/Whisper ASR -> Burushaski text -> mT5 -> English text
```

Direct speech-to-English baseline:

```text
Burushaski audio -> Whisper -> Burushaski transcript + English text
```

Text translation:

```text
Burushaski text -> mT5 -> English text
```

Whisper is used with audio input. It is not used as a Burushaski-text-to-English text model.

The first training phase focuses on Hunza because that is where we currently have the strongest coverage. The translation model still keeps dialect tags in the prompts.

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

To also cache the trained ASR checkpoints used during comparison:

```bash
python setup/download_models.py --include-trained
```

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
python test_run_loader.py --task asr --split test --use-supabase --dialect hunza --source supabase --decode-audio
```

**6. Run short checks first**
```bash
python train/xlsr.py --config configs/asr_xlsr_micro.yaml --use-hf --fp16
python train/mt5.py --config configs/mt5_micro.yaml --use-hf
python test_translation_smoke.py --config configs/mt5_short.yaml --use-hf --use-supabase --forward
python test_whisper_st_smoke.py --config configs/whisper_st_micro.yaml --use-hf --forward
```

These commands only run a couple of optimizer steps. Use them to check that training, saving, and reloading work before starting the full run. For mT5, use the default precision first unless you have already tested another precision mode on the target GPU.

After the translation models are saved, these two commands check the comparison scripts on a tiny sample before running the full evaluation:

```bash
python evaluate/asr_all.py --use-hf --use-supabase --max-samples 2
python evaluate/cascade_all.py --use-hf --use-supabase --max-samples 2
```

**7. Train text and direct speech translation**
```bash
python train/mt5.py --config configs/mt5_short.yaml --use-hf
python train/whisper_st.py --config configs/whisper_st_short.yaml --use-hf --fp16
```

The mT5 short config trains Burushaski text to English for three epochs. The Whisper short config continues the speech-to-English checkpoint and trains it to output both Burushaski transcript and English translation.

**8. Evaluate**
```bash
# Single-model checks
python evaluate/xlsr.py --model-path outputs/asr-xlsr_final --use-hf --use-supabase
python evaluate/mt5.py --model-path outputs/mt5-bsk-eng-short_final --config configs/mt5_short.yaml --use-hf --use-supabase

# Model comparison
python evaluate/asr_all.py --use-hf --use-supabase
python evaluate/cascade_all.py --use-hf --use-supabase
```

Evaluation writes detailed predictions plus summary CSV, JSON, and short readable TXT files under `outputs/.../results`. If `WANDB_API_KEY` is set, the summary metrics and result files are also attached to the W&B run.

Saved prediction CSVs can be rescored later without retraining:

```bash
python evaluate/saved_predictions.py --task asr --predictions outputs/asr-xlsr/results/xlsr_predictions.csv --output outputs/recomputed_metrics/xlsr
python evaluate/saved_predictions.py --task mt --predictions outputs/mt5-bsk-eng/results/mt5_predictions.csv --output outputs/recomputed_metrics/mt5
```

**9. Upload final checkpoints to Hugging Face**
```bash
hf upload Yaraan/mt5-hunza-bsk-eng-v1 outputs/mt5-bsk-eng-short_final .
hf upload Yaraan/whisper-hunza-bsk-eng-v1 outputs/whisper-bsk-eng-short_final .
hf upload Yaraan/mt5-hunza-bsk-eng-v1 outputs/mt5-bsk-eng/results results
hf upload Yaraan/whisper-hunza-bsk-eng-v1 outputs/cascade-all/results cascade-results
hf upload Yaraan/yaraan-hunza-asr-comparison-results outputs/asr-all/results .
hf upload Yaraan/yaraan-hunza-cascade-results outputs/cascade-all/results .
```

For a fresh run, the training scripts can also upload automatically after training:

```bash
python train/mt5.py --config configs/mt5_short.yaml --use-hf --push-to-hub
python train/whisper_st.py --config configs/whisper_st_short.yaml --use-hf --fp16 --push-to-hub
```

---

## Hugging Face Access

Private Hugging Face datasets cannot be cloned with your account password. Use a user access token.

On RunPod, the easiest path is usually:

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
  model_registry.yaml  # ASR/translation checkpoints used in evaluation
  asr_xlsr.yaml        # XLS-R ASR settings
  mt5.yaml             # mT5 translation settings
  mt5_short.yaml       # practical first mT5 run
  whisper_st_short.yaml  # practical first Whisper speech-translation run

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
  whisper_st.py        # fine-tune Whisper for speech-to-English
evaluate/
  xlsr.py              # compute ASR CER/WER
  asr_all.py           # compare XLS-R, MMS, and Whisper ASR checkpoints
  mt5.py               # compute MT chrF++/BLEU
  pipeline.py          # XLS-R -> mT5 cascade evaluator
  cascade_all.py       # compare ASR -> mT5 cascades

```

---

## Data

The normal training path is linked mode. Hugging Face supplies the 80/20 training/test split, while Supabase is used as an additional Hunza evaluation source.

```bash
python train/mt5.py --config configs/mt5_short.yaml --use-hf
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

The evaluation scripts run after a checkpoint exists and write fresh metrics from the selected test data.

ASR is evaluated with both raw and normalized:

- `WER`: word error rate
- `CER`: character error rate
- word accuracy
- character accuracy
- sentence error rate
- exact match rate
- empty prediction rate
- average reference/prediction length

CER is especially important here because Burushaski spelling is not fully standardized, and character-level mistakes are more informative than only counting whole-word errors. The evaluator also writes group summaries by dialect, participant, and gender.

Text translation and speech-to-English outputs are evaluated with:

- `chrF++`
- `BLEU`
- `TER`
- exact match rate
- empty prediction rate
- average length ratio

`chrF++` is useful for low-resource and spelling-variable settings because it gives credit for character n-gram overlap, while BLEU remains a common comparison point in MT and speech-translation papers.

The mT5 evaluator writes summaries by translation direction, dialect, and source.

The cascade evaluator compares the available ASR-to-mT5 systems on the same test set:

```text
XLS-R predicted transcript -> mT5 -> English
MMS predicted transcript -> mT5 -> English
Whisper predicted transcript -> mT5 -> English
Whisper direct speech translation -> English
```

This shows how much translation quality changes depending on which ASR transcript is fed into the text translator.

---

## Known Limitations

- **HuggingFace private access** - set `HF_TOKEN` before using `--use-hf`.
- **Supabase `--use-supabase` flag** - use a service role key in `.env`. Do not paste it into notebooks or logs.
- **HF/Supabase split policy** - training uses the HF Hunza train split. Evaluation uses the HF Hunza test split plus complete Hunza rows from Supabase.
- **Speaker and prompt overlap** - before publication-style reporting, confirm whether the HF split is speaker/prompt independent.
