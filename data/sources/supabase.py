import os

from datasets import Dataset
from dotenv import load_dotenv
from supabase import create_client

from data.storage.audio import get_cached_audio

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

def _has_text(row, key):
    value = row.get(key)
    return value is not None and str(value).strip() != ""

def load_supabase_dataset(task: str = "mt"):
    if supabase is None:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set before loading Supabase data")

    response = (
        supabase.table("active_recordings")
        .select("*")
        .execute()
    )
    rows = response.data

    users_response = (
        supabase.table("app_users")
        .select("participant_id, dialect, gender")
        .execute()
    )
    users_by_participant = {
        row["participant_id"]: row
        for row in users_response.data
        if row.get("participant_id")
    }

    filtered_rows = []
    for row in rows:
        if task in {"asr", "s2tt", "tts"} and not _has_text(row, "audio_path"):
            continue
        if task in {"asr", "mt", "tts"} and not _has_text(row, "transcript"):
            continue
        if task in {"mt", "s2tt"} and not _has_text(row, "english_translation"):
            continue
        filtered_rows.append(row)

    dataset = []
    for row in filtered_rows:
        local_audio = get_cached_audio(row["audio_path"]) if _has_text(row, "audio_path") else ""
        user = users_by_participant.get(row.get("participant_id"), {})

        example = {
            "id" : row["id"],
            "audio" : str(local_audio),
            "transcript" : row["transcript"],
            "english_translation" : row["english_translation"],
            "dialect" : row.get("dialect") or user.get("dialect"),
            "gender" : row.get("gender") or user.get("gender"),
            "participant_id" : row.get("participant_id"),
            "source": "supabase",
            "domain": row.get("module_id"),
        }

        dataset.append(example)

    return Dataset.from_list(dataset)
