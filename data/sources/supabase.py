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


def _clean_text(value):
    return str(value).strip() if value is not None and str(value).strip() else ""

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


def _prompt_lookup():
    prompts = _fetch_all("prompt_bank", "prompt_id,module_id,dialect,english,transliteration,active")
    return {
        (row.get("module_id"), row.get("prompt_id")): row
        for row in prompts
        if row.get("module_id") and row.get("prompt_id")
    }


def _usable_task(row):
    status = str(row.get("status") or "").lower()
    return status in {"done", "review"}


def _task_lookup():
    try:
        tasks = _fetch_all(
            "research_tasks",
            "id,recording_id,status,requested_outputs,transcript,translation,updated_at,completed_at,created_at",
        )
    except Exception:
        return {}

    by_recording = {}
    for row in tasks:
        if not _usable_task(row):
            continue
        if not (_clean_text(row.get("transcript")) or _clean_text(row.get("translation"))):
            continue
        rec_key = row.get("recording_id")
        if rec_key:
            by_recording.setdefault(rec_key, []).append(row)
    return by_recording


def _task_value(tasks, key):
    for status in ["done", "review"]:
        for row in sorted(tasks, key=lambda item: str(item.get("completed_at") or item.get("updated_at") or item.get("created_at") or ""), reverse=True):
            if str(row.get("status") or "").lower() == status and _clean_text(row.get(key)):
                return _clean_text(row.get(key))
    return ""


def _resolve_task(row, tasks_by_recording):
    return tasks_by_recording.get(row.get("id"), [])


def load_supabase_dataset(task: str = "mt", dialects: list[str] | None = None):
    if supabase is None:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY must be set before loading Supabase data")

    rows = _fetch_all("active_recordings")
    users_response = _fetch_all("app_users", "participant_id, dialect, dialects, gender")
    prompts_by_key = _prompt_lookup()
    tasks_by_recording = _task_lookup()
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
        prompt = prompts_by_key.get((row.get("module_id"), row.get("sentence_id")), {})
        tasks = _resolve_task(row, tasks_by_recording)
        resolved_transcript = _clean_text(row.get("transcript")) or _task_value(tasks, "transcript")
        resolved_english = (
            _clean_text(row.get("english_translation"))
            or _task_value(tasks, "translation")
            or _clean_text(prompt.get("english"))
        )
        if wanted_dialects and dialect not in wanted_dialects:
            continue
        if task == "asr" and not _has_text(row, "audio_path"):
            continue
        if task in {"asr", "mt"} and not resolved_transcript:
            continue
        if task == "mt" and not resolved_english:
            continue
        filtered_rows.append((row, prompt, resolved_transcript, resolved_english))

    dataset = []
    for row, prompt, resolved_transcript, resolved_english in filtered_rows:
        local_audio = get_cached_audio(row["audio_path"]) if _has_text(row, "audio_path") else ""
        user = users_by_participant.get(row.get("participant_id"), {})

        example = {
            "id" : row.get("id") or row.get("recording_id") or row.get("audio_path"),
            "audio" : str(local_audio),
            "transcript" : resolved_transcript,
            "english_translation" : resolved_english,
            "dialect" : row.get("dialect") or user.get("dialect"),
            "gender" : row.get("gender") or user.get("gender"),
            "participant_id" : row.get("participant_id"),
            "source": "supabase",
            "domain": row.get("module_id") or row.get("prompt_type") or "prompted",
            "filename": row.get("audio_path") or "",
            "prompt_id": row.get("sentence_id") or row.get("prompt_id") or "",
            "prompt_english": prompt.get("english") or "",
        }

        dataset.append(example)

    return Dataset.from_list(dataset)
