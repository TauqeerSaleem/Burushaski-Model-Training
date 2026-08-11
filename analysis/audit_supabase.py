import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def has_text(value):
    return value is not None and str(value).strip() != ""


def clean_text(value):
    return str(value).strip() if has_text(value) else ""


def fetch_all(client, table_name, select_columns="*"):
    rows = []
    page_size = 1000
    start = 0
    while True:
        response = (
            client.table(table_name)
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


def make_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY must be set")
    return create_client(url, key)


def user_lookup(users):
    return {
        row.get("participant_id"): row
        for row in users
        if row.get("participant_id")
    }


def normalized_dialect(row, users_by_participant):
    user = users_by_participant.get(row.get("participant_id"), {})
    return str(row.get("dialect") or user.get("dialect") or "unknown").strip().lower()


def normalized_gender(row, users_by_participant):
    user = users_by_participant.get(row.get("participant_id"), {})
    return str(row.get("gender") or user.get("gender") or "unknown").strip().lower()


def prompt_lookup(prompts):
    return {
        (row.get("module_id"), row.get("prompt_id")): row
        for row in prompts
        if row.get("module_id") and row.get("prompt_id")
    }


def usable_task(row):
    return str(row.get("status") or "").lower() in {"done", "review"}


def task_lookups(tasks):
    by_recording = {}
    for row in tasks:
        if not usable_task(row):
            continue
        if not (clean_text(row.get("transcript")) or clean_text(row.get("translation"))):
            continue
        if row.get("recording_id"):
            by_recording.setdefault(row.get("recording_id"), []).append(row)
    return by_recording


def related_tasks(row, by_recording):
    return by_recording.get(row.get("id"), [])


def task_value(tasks, key):
    ordered = sorted(
        tasks,
        key=lambda item: str(item.get("completed_at") or item.get("updated_at") or item.get("created_at") or ""),
        reverse=True,
    )
    for status in ["done", "review"]:
        for row in ordered:
            if str(row.get("status") or "").lower() == status and clean_text(row.get(key)):
                return clean_text(row.get(key)), status
    return "", ""


def build_frame(recordings, users_by_participant, prompts_by_key, tasks_by_recording):
    rows = []
    for row in recordings:
        recording_transcript = clean_text(row.get("transcript"))
        recording_english = clean_text(row.get("english_translation"))
        prompt = prompts_by_key.get((row.get("module_id"), row.get("sentence_id")), {})
        prompt_english = clean_text(prompt.get("english"))
        tasks = related_tasks(row, tasks_by_recording)
        task_transcript, task_transcript_status = task_value(tasks, "transcript")
        task_translation, task_translation_status = task_value(tasks, "translation")
        transcript = recording_transcript or task_transcript
        english = recording_english or task_translation or prompt_english
        audio_path = row.get("audio_path")
        rows.append({
            "id": row.get("id") or row.get("recording_id") or audio_path,
            "participant_id": row.get("participant_id") or "unknown",
            "dialect": normalized_dialect(row, users_by_participant),
            "gender": normalized_gender(row, users_by_participant),
            "module_id": row.get("module_id") or row.get("prompt_type") or "unknown",
            "prompt_id": row.get("prompt_id") or row.get("sentence_id") or "",
            "audio_path": audio_path or "",
            "has_audio": has_text(audio_path),
            "has_transcript": has_text(transcript),
            "has_english_translation": has_text(english),
            "recording_has_transcript": has_text(recording_transcript),
            "task_has_transcript": has_text(task_transcript),
            "task_transcript_status": task_transcript_status,
            "recording_has_english_translation": has_text(recording_english),
            "task_has_translation": has_text(task_translation),
            "task_translation_status": task_translation_status,
            "prompt_has_english": has_text(prompt_english),
            "english_source": (
                "recording_english_translation"
                if recording_english
                else "research_task_translation"
                if task_translation
                else "prompt_bank_english"
                if prompt_english
                else ""
            ),
            "transcript": transcript or "",
            "english_translation": english or "",
        })
    return pd.DataFrame(rows)


def counts_by(frame, columns):
    if frame.empty:
        return pd.DataFrame(columns=columns + ["rows"])
    return frame.groupby(columns, dropna=False).size().reset_index(name="rows")


def task_counts(frame):
    rows = []
    for dialect, group in frame.groupby("dialect", dropna=False):
        rows.append({
            "dialect": dialect,
            "total_rows": len(group),
            "asr_ready_rows": int((group["has_audio"] & group["has_transcript"]).sum()),
            "mt_ready_rows": int((group["has_transcript"] & group["has_english_translation"]).sum()),
            "full_pipeline_ready_rows": int((group["has_audio"] & group["has_transcript"] & group["has_english_translation"]).sum()),
            "recording_transcript_rows": int(group["recording_has_transcript"].sum()),
            "task_transcript_rows": int(group["task_has_transcript"].sum()),
            "recording_translation_rows": int(group["recording_has_english_translation"].sum()),
            "task_translation_rows": int(group["task_has_translation"].sum()),
            "prompt_english_rows": int(group["prompt_has_english"].sum()),
            "missing_audio": int((~group["has_audio"]).sum()),
            "missing_transcript": int((~group["has_transcript"]).sum()),
            "missing_english_translation": int((~group["has_english_translation"]).sum()),
        })
    return pd.DataFrame(rows)


def duplicate_candidates(frame):
    keys = ["participant_id", "prompt_id"]
    if "prompt_id" not in frame or frame["prompt_id"].astype(str).str.len().sum() == 0:
        keys = ["participant_id", "transcript"]
    grouped = frame.groupby(keys, dropna=False).size().reset_index(name="rows")
    return grouped[grouped["rows"] > 1].sort_values("rows", ascending=False)


def write_summary(frame, output_dir, dialect):
    scoped = frame if dialect == "all" else frame[frame["dialect"] == dialect]
    lines = [
        "Supabase audit summary",
        "",
        f"Dialect scope: {dialect}",
        f"Total rows: {len(scoped)}",
        f"ASR-ready rows: {int((scoped['has_audio'] & scoped['has_transcript']).sum()) if not scoped.empty else 0}",
        f"MT-ready rows: {int((scoped['has_transcript'] & scoped['has_english_translation']).sum()) if not scoped.empty else 0}",
        f"Full pipeline-ready rows: {int((scoped['has_audio'] & scoped['has_transcript'] & scoped['has_english_translation']).sum()) if not scoped.empty else 0}",
        "",
        "Resolved transcript sources:",
        f"- transcript already on recording: {int(scoped['recording_has_transcript'].sum()) if not scoped.empty else 0}",
        f"- transcript available from done/review RA task: {int(scoped['task_has_transcript'].sum()) if not scoped.empty else 0}",
        "",
        "Resolved English sources:",
        f"- English translation already on recording: {int(scoped['recording_has_english_translation'].sum()) if not scoped.empty else 0}",
        f"- English translation available from done/review RA task: {int(scoped['task_has_translation'].sum()) if not scoped.empty else 0}",
        f"- English prompt available from prompt bank: {int(scoped['prompt_has_english'].sum()) if not scoped.empty else 0}",
        "",
        f"Missing audio: {int((~scoped['has_audio']).sum()) if not scoped.empty else 0}",
        f"Missing transcript: {int((~scoped['has_transcript']).sum()) if not scoped.empty else 0}",
        f"Missing English translation: {int((~scoped['has_english_translation']).sum()) if not scoped.empty else 0}",
    ]
    (output_dir / "supabase_audit_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialect", default="hunza")
    parser.add_argument("--output-dir", default="analysis_outputs/supabase_audit")
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = make_client()
    recordings = fetch_all(client, "active_recordings")
    users = fetch_all(client, "app_users", "participant_id, dialect, dialects, gender")
    prompts = fetch_all(client, "prompt_bank", "prompt_id,module_id,dialect,english,transliteration,active")
    try:
        tasks = fetch_all(
            client,
            "research_tasks",
            "id,recording_id,status,requested_outputs,transcript,translation,updated_at,completed_at,created_at",
        )
    except Exception:
        tasks = []
    tasks_by_recording = task_lookups(tasks)
    frame = build_frame(recordings, user_lookup(users), prompt_lookup(prompts), tasks_by_recording)

    scoped = frame if args.dialect == "all" else frame[frame["dialect"] == args.dialect.lower()]
    task_counts(frame).to_csv(output_dir / "supabase_task_counts_by_dialect.csv", index=False)
    counts_by(scoped, ["dialect", "gender"]).to_csv(output_dir / "supabase_counts_by_gender.csv", index=False)
    counts_by(scoped, ["dialect", "module_id"]).to_csv(output_dir / "supabase_counts_by_module.csv", index=False)
    duplicate_candidates(scoped).to_csv(output_dir / "supabase_duplicate_candidates.csv", index=False)
    scoped[
        ~scoped["has_audio"] | ~scoped["has_transcript"] | ~scoped["has_english_translation"]
    ].to_csv(output_dir / "supabase_missing_rows.csv", index=False)
    scoped.head(args.sample_size).to_csv(output_dir / "supabase_sample_rows.csv", index=False)
    write_summary(frame, output_dir, args.dialect.lower())

    print(f"Supabase audit saved to {output_dir}")


if __name__ == "__main__":
    main()
