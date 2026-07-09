from datasets import concatenate_datasets

from data.sources.hf import load_hf_dataset
from data.sources.supabase import load_supabase_dataset

def load_dataset(
        #task: str, 
        #split: str = "train", 
        #use_hf: bool = False, 
        use_supabase: bool = True
):
  datasets = []
  #if use_hf:
  #  hf_dataset = load_hf_dataset(
  #    task=task,
  #    split=split,
  #  )
  #  datasets.append(hf_dataset)
  if use_supabase:
    supabase_dataset = load_supabase_dataset(
      #task = task,
      #split = split,
    )
    datasets.append(supabase_dataset)

  if len(datasets) == 0:
    raise ValueError("No dataset sources selected")

  if len(datasets) == 1:
    return datasets[0]

  return concatenate_datasets(datasets)

