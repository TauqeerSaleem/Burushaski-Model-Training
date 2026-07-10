from pathlib import Path
import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
)

BUCKET_NAME = "audio-recordings"
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache/audio"))

CACHE_DIR.mkdir(
    parents = True,
    exist_ok = True,
)

def get_cached_audio(audio_path : str) -> Path:
    local_path = CACHE_DIR / audio_path

    if local_path.exists():
        return local_path
    
    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_bytes = (
        supabase.storage
        .from_(BUCKET_NAME)
        .download(audio_path)
    )

    with open(local_path, "wb") as f:
        f.write(file_bytes)

    return local_path
