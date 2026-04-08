# ============================================================
# Suppress warnings
# ============================================================
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ============================================================
# SECTION 1 — Imports
# ============================================================
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from transformer_lens import HookedTransformer

# ============================================================
# SECTION 2 — Load Model and Dataset
# ============================================================
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
print("✓ Model loaded\n")

df = pd.read_csv("negation_dataset.csv")
print(f"✓ Dataset loaded — {len(df)} examples\n")

# ============================================================
# SECTION 3 — Define Key Parameters
# ============================================================
# GPT-2 Small has:
# - 12 layers (numbered 0-11)
# - 12 attention heads per layer (numbered 0-11)
# - 144 total attention heads to test
#
# For each head we'll store one number — the "patching effect"
# This measures how much that head contributes to negation

N_LAYERS = model.cfg.n_layers   # 12
N_HEADS = model.cfg.n_heads     # 12

print(f"Model architecture:")
print(f"  Layers: {N_LAYERS}")
print(f"  Heads per layer: {N_HEADS}")
print(f"  Total heads to test: {N_LAYERS * N_HEADS}\n")

# ============================================================
# SECTION 4 — Activation Patching Function
# ============================================================
# This is the core of the entire experiment.
# For a single example, we:
# 1. Run control prompt → cache all head activations
# 2. For each head in each layer:
#    a. Run negation prompt
#    b. At that specific head — swap in the control activation
#    c. Measure the logit difference
#    d. Record how much it changed
#
# The change in logit difference IS the patching effect

def patch_single_example(control_prompt, negation_prompt,
                          correct_token, false_token):
    """
    Runs activation patching on one example.
    Returns a (n_layers, n_heads) matrix of patching effects.
    """
    # Verify both tokens are single tokens
    correct_tokens = model.to_tokens(correct_token, prepend_bos=False)
    false_tokens = model.to_tokens(" " + false_token, prepend_bos=False)

    if correct_tokens.shape[1] != 1 or false_tokens.shape[1] != 1:
        raise ValueError(f"Multi-token answer detected")

    correct_id = model.to_single_token(correct_token)
    false_id = false_tokens[0, 0].item()

    # Tokenize both prompts
    control_tokens = model.to_tokens(control_prompt)
    negation_tokens = model.to_tokens(negation_prompt)

    # Run control prompt and cache activations
    with torch.no_grad():
        _, control_cache = model.run_with_cache(control_tokens)

    # Get baseline logit difference on negation (no patching)
    with torch.no_grad():
        negation_logits = model(negation_tokens)
    last_neg = negation_logits[0, -1, :]
    baseline_ld = (last_neg[correct_id] - last_neg[false_id]).item()

    # Get ideal logit difference on control
    with torch.no_grad():
        control_logits = model(control_tokens)
    last_ctrl = control_logits[0, -1, :]
    ideal_ld = (last_ctrl[correct_id] - last_ctrl[false_id]).item()

    if abs(ideal_ld - baseline_ld) < 0.01:
        raise ValueError("No meaningful logit difference")

    # Patch each head one at a time
    results = np.zeros((N_LAYERS, N_HEADS))

    for layer in range(N_LAYERS):
        # Store cached values for this layer
        # We copy to avoid any reference issues
        cached_v = control_cache[
            f"blocks.{layer}.attn.hook_v"
        ].detach().clone()

        for head in range(N_HEADS):
            # Store head index as local variable
            # to avoid closure scoping issues
            head_idx = head
            cached_v_copy = cached_v.clone()

            # Define hook using a class to avoid closure issues
            class PatchHook:
                def __init__(self, cache, h):
                    self.cache = cache
                    self.h = h

                def __call__(self, value, hook):
                    # Patch only the last token position
                    # to handle different sequence lengths
                    value[:, -1:, self.h, :] = \
                        self.cache[:, -1:, self.h, :]
                    return value

            hook_fn = PatchHook(cached_v_copy, head_idx)
            hook_name = f"blocks.{layer}.attn.hook_v"

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    negation_tokens,
                    fwd_hooks=[(hook_name, hook_fn)]
                )

            last_patch = patched_logits[0, -1, :]
            patched_ld = (last_patch[correct_id] -
                         last_patch[false_id]).item()

            effect = (patched_ld - baseline_ld) / \
                     (ideal_ld - baseline_ld)
            results[layer][head] = effect

    return results

# ============================================================
# SECTION 5 — Run Patching Across All Examples
# ============================================================
# We run patch_single_example() on every example in our dataset
# and average the results.
#
# Averaging across 107 examples is crucial — it means our
# findings aren't just true for one sentence, they're a
# general property of how GPT-2 handles negation
#
# This is what makes it a publishable finding rather than
# an interesting observation

print("Running activation patching...")
print("(This will take several minutes — 107 examples × 144 heads)\n")

all_results = np.zeros((N_LAYERS, N_HEADS))
successful = 0
errors = []

for i, row in tqdm(df.iterrows(), total=len(df),
                   desc="Patching examples"):
    try:
        result = patch_single_example(
            row["control_prompt"],
            row["negation_prompt"],
            row["expected_token"],
            row["target_false"]
        )
        all_results += result
        successful += 1
    except Exception as e:
        errors.append(str(e))
        continue

print(f"\n✓ Patching complete — {successful} examples processed")
print(f"✗ Skipped: {len(errors)} examples")

# Show most common errors so we can debug
if errors:
    common_errors = Counter(errors).most_common(3)
    print("\nMost common errors:")
    for err, count in common_errors:
        print(f"  {count}x — {err}")

# Average across successful examples
if successful > 0:
    avg_results = all_results / successful
else:
    print("\n❌ No examples processed successfully — check errors above")
    exit()

# ============================================================
# SECTION 6 — Save Results
# ============================================================
np.save("activation_patching_out/patching_results.npy", avg_results)
print("\n✓ Results saved to patching_results.npy\n")

# ============================================================
# SECTION 7 — Identify Important Heads
# ============================================================
# Find the top heads — the ones with the highest patching effect
# These are the candidates for the negation circuit

print("=== Top 10 Most Important Heads For Negation ===\n")

flat_results = avg_results.flatten()
top_indices = np.argsort(flat_results)[::-1][:10]

for rank, idx in enumerate(top_indices):
    layer = idx // N_HEADS
    head = idx % N_HEADS
    effect = flat_results[idx]
    print(f"  Rank {rank+1}: Layer {layer}, Head {head} "
          f"— patching effect: {effect:.4f}")

# ============================================================
# SECTION 8 — Plot The Heatmap
# ============================================================
# This is Figure 2 of your paper.
# Each cell = one attention head
# Color intensity = how important that head is for negation
# Bright red cells = the negation circuit

print("\nGenerating heatmap...")

fig, ax = plt.subplots(figsize=(14, 8))

sns.heatmap(
    avg_results,
    ax=ax,
    cmap="RdBu_r",        # Red = positive effect, Blue = negative
    center=0,             # White = no effect
    annot=True,           # Show numbers in each cell
    fmt=".2f",            # 2 decimal places
    linewidths=0.5,
    xticklabels=[f"H{i}" for i in range(N_HEADS)],
    yticklabels=[f"L{i}" for i in range(N_LAYERS)]
)

ax.set_title(
    "Activation Patching Results — Negation Circuit in GPT-2\n"
    "Higher values (red) = head is important for processing negation",
    fontsize=13, pad=15
)
ax.set_xlabel("Attention Head", fontsize=11)
ax.set_ylabel("Layer", fontsize=11)

plt.tight_layout()
plt.savefig("patching_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Heatmap saved to patching_heatmap.png")

# ============================================================
# SECTION 9 — Summary For Paper
# ============================================================
top_layer = top_indices[0] // N_HEADS
top_head = top_indices[0] % N_HEADS
top_effect = flat_results[top_indices[0]]

print(f"\n=== Key Finding ===")
print(f"Most important head: Layer {top_layer}, Head {top_head}")
print(f"Patching effect: {top_effect:.4f}")
print(f"\nInterpretation:")
print(f"  Patching L{top_layer}H{top_head} recovers "
      f"{top_effect*100:.1f}% of the performance gap")
print(f"  between negation and control prompts.")
print(f"  This head is a strong candidate for the negation circuit.")