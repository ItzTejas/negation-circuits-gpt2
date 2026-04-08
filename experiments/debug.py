import warnings
import os
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
model.eval()

# Test with one example
prompt = "The mother tongue of Danielle Darrieux is"
tokens = model.to_tokens(prompt)

with torch.no_grad():
    _, cache = model.run_with_cache(tokens)

# Print all available cache keys
print("Available cache keys (first 20):")
keys = list(cache.keys())
for k in keys[:20]:
    print(f"  {k} — shape: {cache[k].shape}")