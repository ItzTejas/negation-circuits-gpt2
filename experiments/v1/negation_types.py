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
from transformer_lens import HookedTransformer

# ============================================================
# SECTION 2 — Load Model and Dataset
# ============================================================
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
print("✓ Model loaded\n")

df = pd.read_csv("negation_dataset.csv")
print(f"✓ Base dataset loaded — {len(df)} examples\n")

# ============================================================
# SECTION 3 — Define Negation Types
# ============================================================
# For each negation type we define a function that takes
# the filled prompt and false target and returns the
# negation prompt in that syntactic form
#
# Example inputs:
#   filled_prompt = "The mother tongue of Danielle Darrieux is"
#   false_target  = "English"
#
# Example outputs per type:
#   standard:     "...is not English, it is"
#   contraction:  "...isn't English, it is"
#   never:        "...has never been English, it is"
#   cannot:       "...cannot be English, it is"

NEGATION_TYPES = {
    "standard": lambda prompt, false: \
        prompt + f" not {false}, it is",

    "contraction": lambda prompt, false: \
        prompt + f"n't {false}, it is",

    "never": lambda prompt, false: \
        prompt.rstrip() + f" has never been {false}, it is",

    "cannot": lambda prompt, false: \
        prompt.rstrip() + f" cannot be {false}, it is",

    "without_negation": lambda prompt, false: \
        prompt,  # control — no negation at all
}

print("Negation types to test:")
for ntype in NEGATION_TYPES:
    print(f"  {ntype}")
print()

# ============================================================
# SECTION 4 — Build Dataset For Each Type
# ============================================================
# For each negation type we construct prompts from our
# existing verified dataset and check how many examples
# the model handles correctly

print("Building prompts for each negation type...")

type_datasets = {}

for ntype, prompt_fn in NEGATION_TYPES.items():
    examples = []
    for _, row in df.iterrows():
        neg_prompt = prompt_fn(
            row["control_prompt"],
            row["target_false"]
        )
        examples.append({
            "negation_prompt": neg_prompt,
            "control_prompt": row["control_prompt"],
            "expected_token": row["expected_token"],
            "target_false": row["target_false"],
            "subject": row["subject"],
            "negation_type": ntype,
        })
    type_datasets[ntype] = examples
    print(f"  {ntype}: {len(examples)} examples built")

print()

# ============================================================
# SECTION 5 — Behavioral Analysis Per Type
# ============================================================
# First we measure how well GPT-2 handles each negation type
# This tells us which types are harder vs easier

print("Running behavioral analysis per negation type...")
print("(Measuring success rates)\n")

def get_top5_tokens(prompt):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    last_logits = logits[0, -1, :]
    top5_ids = torch.topk(last_logits, 5).indices
    return [model.to_string(t) for t in top5_ids]

def get_probability(prompt, expected_token):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    try:
        expected_id = model.to_single_token(expected_token)
        return probs[expected_id].item()
    except:
        return 0.0

behavioral_results = {}

for ntype, examples in type_datasets.items():
    successes = 0
    probs = []

    for ex in examples:
        top5 = get_top5_tokens(ex["negation_prompt"])
        if ex["expected_token"] in top5:
            successes += 1
        prob = get_probability(
            ex["negation_prompt"],
            ex["expected_token"]
        )
        probs.append(prob)

    success_rate = successes / len(examples) * 100
    avg_prob = np.mean(probs)

    behavioral_results[ntype] = {
        "success_rate": success_rate,
        "avg_prob": avg_prob,
        "n": len(examples)
    }

    print(f"{ntype}:")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Avg probability: {avg_prob:.4f}")
    print()

# ============================================================
# SECTION 6 — Activation Patching Per Type
# ============================================================
# Now the key experiment — we run activation patching for
# each negation type and compare which heads light up
#
# If the same heads appear across all types → shared circuit
# If different heads appear → type-specific processing

N_LAYERS = model.cfg.n_layers
N_HEADS = model.cfg.n_heads

def patch_single_example(control_prompt, negation_prompt,
                          correct_token, false_token):
    """Activation patching for one example."""
    # Verify single tokens
    correct_tokens = model.to_tokens(
        correct_token, prepend_bos=False)
    false_tokens = model.to_tokens(
        " " + false_token, prepend_bos=False)

    if correct_tokens.shape[1] != 1 or \
       false_tokens.shape[1] != 1:
        raise ValueError("Multi-token answer")

    correct_id = model.to_single_token(correct_token)
    false_id = false_tokens[0, 0].item()

    control_tokens = model.to_tokens(control_prompt)
    negation_tokens = model.to_tokens(negation_prompt)

    with torch.no_grad():
        _, control_cache = model.run_with_cache(control_tokens)

    with torch.no_grad():
        neg_logits = model(negation_tokens)
    last_neg = neg_logits[0, -1, :]
    baseline_ld = (last_neg[correct_id] -
                   last_neg[false_id]).item()

    with torch.no_grad():
        ctrl_logits = model(control_tokens)
    last_ctrl = ctrl_logits[0, -1, :]
    ideal_ld = (last_ctrl[correct_id] -
                last_ctrl[false_id]).item()

    if abs(ideal_ld - baseline_ld) < 0.01:
        raise ValueError("No meaningful difference")

    results = np.zeros((N_LAYERS, N_HEADS))

    for layer in range(N_LAYERS):
        cached_v = control_cache[
            f"blocks.{layer}.attn.hook_v"
        ].detach().clone()

        for head in range(N_HEADS):
            class PatchHook:
                def __init__(self, cache, h):
                    self.cache = cache
                    self.h = h
                def __call__(self, value, hook):
                    value[:, -1:, self.h, :] = \
                        self.cache[:, -1:, self.h, :]
                    return value

            hook_fn = PatchHook(cached_v.clone(), head)
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

print("Running activation patching per negation type...")
print("(This will take 10-20 minutes)\n")

# Skip without_negation — it has no negation to patch
patch_types = [t for t in NEGATION_TYPES
               if t != "without_negation"]

all_type_results = {}

for ntype in patch_types:
    print(f"Patching {ntype}...")
    examples = type_datasets[ntype]
    type_results = np.zeros((N_LAYERS, N_HEADS))
    successful = 0

    for ex in tqdm(examples,
                   desc=f"  {ntype}",
                   leave=False):
        try:
            result = patch_single_example(
                ex["control_prompt"],
                ex["negation_prompt"],
                ex["expected_token"],
                ex["target_false"]
            )
            type_results += result
            successful += 1
        except:
            continue

    if successful > 0:
        all_type_results[ntype] = type_results / successful
        print(f"  ✓ {successful} examples processed")
    else:
        print(f"  ✗ No examples processed")

np.save("negation_types_out/negation_types_results.npy",
        all_type_results, allow_pickle=True)
print("\n✓ All patching complete\n")

# ============================================================
# SECTION 7 — Compare Top Heads Across Types
# ============================================================
print("=== Top 5 Heads Per Negation Type ===\n")

top_heads_per_type = {}

for ntype, results in all_type_results.items():
    flat = results.flatten()
    top5_idx = np.argsort(flat)[::-1][:5]
    top_heads = [(idx // N_HEADS, idx % N_HEADS,
                  flat[idx]) for idx in top5_idx]
    top_heads_per_type[ntype] = top_heads

    print(f"{ntype}:")
    for layer, head, effect in top_heads:
        # Flag if this is one of our original circuit heads
        flag = " ← CIRCUIT HEAD" if (layer, head) in \
               [(7, 3), (11, 3), (6, 5)] else ""
        print(f"  L{layer}H{head}: {effect:.4f}{flag}")
    print()

# ============================================================
# SECTION 8 — Visualize All Types Side By Side
# ============================================================
print("Generating comparison heatmaps...")

n_types = len(all_type_results)
fig, axes = plt.subplots(
    1, n_types,
    figsize=(6 * n_types, 8)
)
fig.suptitle(
    "Activation Patching Across Negation Types\n"
    "Do the same heads handle all forms of negation?",
    fontsize=13
)

for idx, (ntype, results) in \
        enumerate(all_type_results.items()):
    ax = axes[idx]
    vmax = max(r.max() for r in all_type_results.values())

    im = ax.imshow(
        results,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax
    )
    ax.set_xticks(range(N_HEADS))
    ax.set_yticks(range(N_LAYERS))
    ax.set_xticklabels(
        [f"H{i}" for i in range(N_HEADS)],
        fontsize=7
    )
    ax.set_yticklabels(
        [f"L{i}" for i in range(N_LAYERS)],
        fontsize=7
    )
    ax.set_title(ntype.replace("_", " ").title(),
                 fontsize=10)
    ax.set_xlabel("Head", fontsize=8)
    if idx == 0:
        ax.set_ylabel("Layer", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig("negation_types_heatmaps.png",
            dpi=120, bbox_inches="tight")
plt.show()
print("✓ Heatmaps saved to negation_types_heatmaps.png")

# ============================================================
# SECTION 9 — Behavioral Comparison Bar Chart
# ============================================================
print("Generating behavioral comparison chart...")

fig, ax = plt.subplots(figsize=(10, 5))

types = list(behavioral_results.keys())
success_rates = [behavioral_results[t]["success_rate"]
                 for t in types]
avg_probs = [behavioral_results[t]["avg_prob"]
             for t in types]

colors = ["#e74c3c", "#e67e22", "#f1c40f",
          "#2ecc71", "#3498db"]
bars = ax.bar(types, success_rates,
              color=colors, alpha=0.85,
              edgecolor="white")

ax.set_xlabel("Negation Type")
ax.set_ylabel("Success Rate (%)")
ax.set_title(
    "GPT-2 Success Rate Across Negation Types\n"
    "(correct answer in top-5 predictions)"
)
ax.set_xticklabels(
    [t.replace("_", "\n") for t in types],
    fontsize=9
)

for bar, rate in zip(bars, success_rates):
    ax.annotate(
        f"{rate:.1f}%",
        xy=(bar.get_x() + bar.get_width()/2,
            bar.get_height()),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center", fontsize=9
    )

plt.tight_layout()
plt.savefig("negation_types_behavior.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✓ Chart saved to negation_types_behavior.png")

# ============================================================
# SECTION 10 — Circuit Overlap Analysis
# ============================================================
# How much do the top heads overlap across negation types?
# High overlap → shared circuit
# Low overlap → type-specific processing

print("\n=== Circuit Overlap Analysis ===\n")

# Get top 10 heads for each type as sets
top10_sets = {}
for ntype, results in all_type_results.items():
    flat = results.flatten()
    top10_idx = np.argsort(flat)[::-1][:10]
    top10_sets[ntype] = set(
        [(idx // N_HEADS, idx % N_HEADS)
         for idx in top10_idx]
    )

# Compare overlap between all pairs
type_names = list(top10_sets.keys())
print("Overlap in top-10 heads between negation types:")
print("(out of 10 heads — higher = more similar circuit)\n")

for i, t1 in enumerate(type_names):
    for t2 in type_names[i+1:]:
        overlap = len(
            top10_sets[t1] & top10_sets[t2]
        )
        print(f"  {t1} ∩ {t2}: {overlap}/10 heads overlap")

print()

# Check how many of our original circuit heads
# appear in top-10 for each type
original_circuit = {(7, 3), (11, 3), (6, 5)}
print("Original circuit heads in top-10 per type:")
for ntype, top10 in top10_sets.items():
    found = original_circuit & top10
    print(f"  {ntype}: {len(found)}/3 circuit heads "
          f"— {found if found else 'none'}")