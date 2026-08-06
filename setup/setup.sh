#!/bin/bash
set -e
echo "++++++++++++++++++++++++"
echo "Project Yaraan - RunPod setup"
echo "++++++++++++++++++++++++"

apt-get update
apt-get install -y git curl wget unzip ffmpeg
cd "$(dirname "$0")"/..
python -m pip install --upgrade pip

# Install PyTorch with CUDA support first (RunPod provides CUDA 12.1 by default)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from template. Fill in the values before running again."
  exit 1
fi

set -a
source .env
set +a

required_vars=(
  HF_TOKEN
  SUPABASE_URL
  WANDB_API_KEY
)

for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "ERROR: $var is not set in .env"
    exit 1
  fi
done

if [ -z "$SUPABASE_SERVICE_ROLE_KEY" ] && [ -z "$SUPABASE_KEY" ]; then
  echo "ERROR: set SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY in .env"
  exit 1
fi

echo "All required environment variables are set."
echo "Setup complete."
