"""Inspect dataset - non-streaming mode, handle nested data."""
from datasets import load_dataset
import json

# Load non-streaming with a small slice
print("Loading dataset (non-streaming, first 5 rows)...")
ds = load_dataset("ai4bharat/MSMARCO-XI", "default", split="train[:5]")

print(f"\nDataset: {ds}")
print(f"\n--- Features ---")
print(ds.features)

print(f"\n--- Sample row 0 ---")
sample = ds[0]

def truncate(obj, maxlen=300):
    if isinstance(obj, str) and len(obj) > maxlen:
        return obj[:maxlen] + "..."
    if isinstance(obj, dict):
        return {k: truncate(v, maxlen) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 5:
            return [truncate(x, maxlen) for x in obj[:5]] + [f"... ({len(obj)} total)"]
        return [truncate(x, maxlen) for x in obj]
    return obj

print(json.dumps(truncate(sample), indent=2, ensure_ascii=False, default=str))

# Inspect passages structure in detail
passages = sample["passages"]
print(f"\n--- Passages detail ---")
print(f"Type: {type(passages).__name__}")
if isinstance(passages, dict):
    for k, v in passages.items():
        print(f"  '{k}': type={type(v).__name__}, len={len(v)}")
        if isinstance(v, list) and len(v) > 0:
            print(f"    [0] type={type(v[0]).__name__}, value={str(v[0])[:150]}")

# Check unique languages
print(f"\n--- Checking languages across rows ---")
for i in range(min(5, len(ds))):
    row = ds[i]
    print(f"  Row {i}: source_lang={row['source_lang']}, target_lang={row['target_lang']}")

# Try loading different per-language configs as the original prompt suggested
from datasets import get_dataset_config_names
configs = get_dataset_config_names("ai4bharat/MSMARCO-XI")
print(f"\nAll configs: {configs}")
