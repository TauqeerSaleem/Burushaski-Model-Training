import csv
import os
import zipfile
from pathlib import Path

from datacollective import download_dataset
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

MDC_DATASET_ID = os.getenv("MDC_DATASET_ID")
MDC_DOWNLOAD_PATH = os.getenv("CACHE_DIR", "cache/mdc")

def extract_archive(archive_path: Path) -> Path:
    extract_dir = archive_path.parent / archive_path.stem
    if extract_dir.exists():
        return extract_dir
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(archive_path.parent)
    return extract_dir

def load_participant_metadata(dataset_root: Path) -> dict:
    metadata = {}
    csv_path = dataset_root / "Participant_meta_data.csv"
    if not csv_path.exists():
        return metadata
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            participant_id = row.get("PARTICIPANT ID", "").strip()
            if participant_id:
                metadata[participant_id] = row
    return metadata

def extract_participant_id(stem: str) -> str | None:
    if stem.startswith("P-"):
        return stem.split("_")[0]
    return None

def load_mdc_dataset(split: str = "train") -> Dataset:
    archive_path = download_dataset(
        MDC_DATASET_ID,
        download_directory=MDC_DOWNLOAD_PATH,
    )

    dataset_root = extract_archive(archive_path)
    metadata = load_participant_metadata(dataset_root)

    audio_dir = dataset_root / split / "AUDIO"
    txt_dir = dataset_root / split / "TXT"

    dataset = []
    for txt_file in sorted(txt_dir.iterdir()):
        if not txt_file.is_file():
            continue

        stem = txt_file.stem
        english_translation = txt_file.read_text(encoding="utf-8").strip()
        if not english_translation:
            continue

        audio_file = next(audio_dir.glob(f"{stem}.*"), None)
        if audio_file is None:
            continue

        participant_id = extract_participant_id(stem)
        participant = metadata.get(participant_id, {}) if participant_id else {}

        example = {
            "id": stem,
            "audio": str(audio_file),
            "transcript": None,
            "english_translation": english_translation,
            "dialect": participant.get("Dialect") or None,
            "participant_id": participant_id,
            "age_group": participant.get("Age Group") or None,
            "gender": participant.get("Gender") or None,
        }
        dataset.append(example)

    return Dataset.from_list(dataset)
