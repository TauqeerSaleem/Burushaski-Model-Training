#!/bin/bash
set -e
echo "++++++++++++++++++++++++"
echo "Project Yaraan - RunPod Setting"
echo "++++++++++++++++++++++++"

apt-get update
apt-get install -y git curl wget unzip
cd "$(dirname "$0")"/..
python -m pip install --upgrade pip

# Install PyTorch with CUDA support first (RunPod provides CUDA 12.1 by default)
pip install torch --index-url https://download.pytorch.org/whl/cu121

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
  SUPABASE_KEY
  WANDB_API_KEY
)

for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "ERROR: $var is not set in .env"
    exit 1
  fi
done

echo "All required environment variables are set."
echo "Setup complete."
