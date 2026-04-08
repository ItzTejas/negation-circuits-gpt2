from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
print("Model loaded successfully")
print(f"Number of layers: {model.cfg.n_layers}")
print(f"Number of heads: {model.cfg.n_heads}")
print(f"Model device: {model.cfg.device}")