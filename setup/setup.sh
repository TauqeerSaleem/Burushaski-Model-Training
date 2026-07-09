#!/bin/bash
set -e
echo "++++++++++++++++++++++++"
echo "Project Yaraan - RunPod Setting"
echo "++++++++++++++++++++++++"

apt-get update
apt-get install -y git curl wget unzip
cd "$(dirname "$0")"/..
python -m pip install --upgrade pip
pip install -r requirements.txt
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from template."
  echo "Please edit .env before training."
fi

set -a
source .env
set +a

required_vars = (
  HF_TOKEN
  SUPABASE_URL
  SUPABASE_KEY
  WANDB_API_KEY
)

for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "ERROR : $var is missing from .env"
    exit 1
  fi
done
