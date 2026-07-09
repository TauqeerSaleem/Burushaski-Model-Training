import os

from datasets import Dataset
from dotenv import load_dotenv
from supabase import create_client

from data.storage.audio import get_cached_audio

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
)

def load_supabase_dataset():
    response = (
        supabase.table("recordings")
        .select("*")
        .execute()
    )

    rows = response.data

    filtered_rows = []

    for row in rows:
        #if not row["audio_path"]:
        #    continue

        #if not row["transcript"]:
        #    continue

        #if not row["english_translation"]:
        #    continue

        filtered_rows.append(row)

    dataset = []
    for row in filtered_rows:
        local_audio = get_cached_audio(
            row["audio_path"]
        )

        example = {
            "id" : row["id"],
            "audio" : str(local_audio),
            "transcript" : row["transcript"],
            "english_translation" : row["english_translation"],
            "dialect" : row["dialect"],
            "participant_id" : row["participant_id"],
        }

        dataset.append(example)

    return Dataset.from_list(dataset)
