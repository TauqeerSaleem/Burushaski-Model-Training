import os

from datasets import load_dataset
import yaml

def load_dataset_config():
    with open("configs/datasets.yaml") as f:
        return yaml.safe_load(f)

def load_hf_dataset(
        task, 
        split = "train"
):
    config = load_dataset_config()
    repos = [repo for repo in config.get(task, {}).get("hf", []) if repo]
    if not repos:
        raise ValueError(f"No Hugging Face repo configured for task '{task}'")

    repo = repos[0]
    token = os.getenv("HF_TOKEN") or None
    return load_dataset(repo, split=split, token=token)
