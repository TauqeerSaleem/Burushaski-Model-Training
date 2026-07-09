from data.loader import load_dataset

ds = load_dataset(use_supabase=True)

print(ds)
if len(ds) > 0:
    print(ds[0])
