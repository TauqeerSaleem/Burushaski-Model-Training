from data.sources.mdc import load_mdc_dataset

print("=" * 40)
print("Testing MDC")
print("=" * 40)
mdc_ds = load_mdc_dataset(split="train")
print(f"Rows: {len(mdc_ds)}")
if len(mdc_ds) > 0:
    print(f"First row: {mdc_ds[0]}")
