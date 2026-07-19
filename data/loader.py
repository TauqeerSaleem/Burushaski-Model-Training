from datasets import concatenate_datasets

from data.sources.hf import load_hf_dataset
from data.sources.mdc import load_mdc_dataset
from data.sources.supabase import load_supabase_dataset
from data.normalization import normalize_burushaski_text


def _value(row, *keys):
  for key in keys:
    value = row.get(key)
    if value is not None and str(value).strip() != "":
      return value
  return None


def _complete_for_task(row, task):
  audio = _value(row, "audio", "audio_path")
  transcript = _value(row, "transcript", "bsk_text_raw", "bsk_text_normalized", "burushaski")
  english = _value(row, "english_translation", "english_text", "translation")

  if task == "asr":
    return audio is not None and transcript is not None
  if task == "mt":
    return transcript is not None and english is not None
  if task == "s2tt":
    return audio is not None and english is not None
  if task == "tts":
    return audio is not None and transcript is not None
  return True


def _standardize(dataset, task):
  def convert(row):
    transcript = _value(row, "transcription", "transcript", "bsk_text_raw", "burushaski")
    normalized = _value(row, "bsk_text_normalized") or normalize_burushaski_text(transcript)
    return {
      "id": str(_value(row, "id", "sample_id", "filename") or ""),
      "audio": _value(row, "audio", "audio_path") or "",
      "transcript": transcript or "",
      "bsk_text_normalized": normalized,
      "english_translation": _value(row, "english_translation", "english_text", "translation") or "",
      "dialect": str(_value(row, "dialect") or "unknown").lower(),
      "participant_id": str(_value(row, "participant_id", "speaker_id") or "unknown"),
      "gender": str(_value(row, "gender") or "unknown").lower(),
      "source": str(_value(row, "source") or "unknown"),
      "domain": str(_value(row, "domain") or "unknown"),
      "filename": str(_value(row, "filename") or _value(row, "id", "sample_id") or ""),
    }

  dataset = dataset.map(convert, remove_columns=dataset.column_names)
  return dataset.filter(lambda row: _complete_for_task(row, task))

def load_dataset(
        task: str = "mt",
        split: str = "train",
        use_hf: bool = True,
        use_supabase: bool = True,
        use_mdc: bool = True,
):
  datasets = []
  if use_hf:
    hf_dataset = load_hf_dataset(
        task=task,
        split=split,
    )
    datasets.append(_standardize(hf_dataset, task))
  if use_supabase and split == "train":
    supabase_dataset = load_supabase_dataset(task=task)
    datasets.append(_standardize(supabase_dataset, task))
  if use_mdc:
    mdc_dataset = load_mdc_dataset(split=split)
    datasets.append(_standardize(mdc_dataset, task))

  if len(datasets) == 0:
    raise ValueError("No dataset sources selected")

  if len(datasets) == 1:
    return datasets[0]

  return concatenate_datasets(datasets)

