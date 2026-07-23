import os

from datasets import Dataset
from dotenv import load_dotenv
from supabase import create_client

from data.storage.audio import get_cached_audio

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

def _has_text(row, key):
    value = row.get(key)
    return value is not None and str(value).strip() != ""

def _fetch_all(table_name, select_columns="*"):
    rows = []
    page_size = 1000
    start = 0
    while True:
        response = (
            supabase.table(table_name)
            .select(select_columns)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows


def load_supabase_dataset(task: str = "mt", dialects: list[str] | None = None):
    if supabase is None:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY must be set before loading Supabase data")

    rows = _fetch_all("active_recordings")
    users_response = _fetch_all("app_users", "participant_id, dialect, dialects, gender")
    users_by_participant = {
        row["participant_id"]: row
        for row in users_response
        if row.get("participant_id")
    }

    wanted_dialects = {dialect.lower() for dialect in dialects or []}
    filtered_rows = []
    for row in rows:
        user = users_by_participant.get(row.get("participant_id"), {})
        dialect = (row.get("dialect") or user.get("dialect") or "").lower()
        if wanted_dialects and dialect not in wanted_dialects:
            continue
        if task == "asr" and not _has_text(row, "audio_path"):
            continue
        if task in {"asr", "mt"} and not _has_text(row, "transcript"):
            continue
        if task == "mt" and not _has_text(row, "english_translation"):
            continue
        filtered_rows.append(row)

    dataset = []
    for row in filtered_rows:
        local_audio = get_cached_audio(row["audio_path"]) if _has_text(row, "audio_path") else ""
        user = users_by_participant.get(row.get("participant_id"), {})

        example = {
            "id" : row.get("id") or row.get("recording_id") or row.get("audio_path"),
            "audio" : str(local_audio),
            "transcript" : row.get("transcript") or "",
            "english_translation" : row.get("english_translation") or "",
            "dialect" : row.get("dialect") or user.get("dialect"),
            "gender" : row.get("gender") or user.get("gender"),
            "participant_id" : row.get("participant_id"),
            "source": "supabase",
            "domain": row.get("module_id") or row.get("prompt_type") or "prompted",
            "filename": row.get("audio_path") or "",
        }

        dataset.append(example)

    return Dataset.from_list(dataset)
