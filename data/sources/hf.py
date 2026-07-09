from datasets import load_dataset
import yaml

def load_dataset_config():
    with open("configs/datasets.yaml") as f:
        return yaml.safe_load(f)

def load_hf_dataset(task, split = "train"):
    config = load_dataset_config()
    repo = config[task]["hf"][0]

    dataset = load_dataset(repo, split=split)
    return dataset
