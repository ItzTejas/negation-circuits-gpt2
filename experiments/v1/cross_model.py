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
from tqdm import tqdm
from transformer_lens import HookedTransformer

# ============================================================
# SECTION 2 — Load Dataset
# ============================================================
df = pd.read_csv("negation_dataset.csv")
print(f"✓ Dataset loaded — {len(df)} examples\n")

# ============================================================
# SECTION 3 — What Is Cross-Model Comparison?
# ============================================================
# We've mapped the negation circuit in GPT-2 Small (117M).
# Now we ask: do bigger models handle negation better?
# Do the same heads specialize across model sizes?
#
# We test 3 models:
# - GPT-2 Small  (117M params, 12 layers, 12 heads)
# - GPT-2 Medium (345M params, 24 layers, 16 heads)
# - GPT-2 Large  (774M params, 36 layers, 20 heads)
#
# For each model we measure:
# 1. Behavioral: negation success rate + avg probability
# 2. Activation patching: which heads matter most
#
# Key questions:
# - Do larger models handle negation better? (scaling)
# - Do the same relative positions show up? (universality)
# - Does the circuit become more localized? (specialization)

MODELS = {
    "gpt2":        "GPT-2 Small  (117M)",
    "gpt2-medium": "GPT-2 Medium (345M)",
    "gpt2-large":  "GPT-2 Large  (774M)",
}

# ============================================================
# SECTION 4 — Helper Functions
# ============================================================

def get_correct_prob(model, prompt, expected_token):
    """Get probability assigned to expected token."""
    try:
        tokens = model.to_tokens(prompt)
        expected_id = model.to_single_token(expected_token)
        with torch.no_grad():
            logits = model(tokens)
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        top5_ids = torch.topk(probs, 5).indices.tolist()
        prob = probs[expected_id].item()
        in_top5 = expected_id in top5_ids
        return prob, in_top5
    except:
        return 0.0, False

def run_behavioral_analysis(model, model_name, df):
    """
    Measures negation success rate and avg probability
    for a given model across all examples.
    """
    print(f"  Running behavioral analysis...")
    neg_probs = []
    ctrl_probs = []
    neg_successes = 0
    ctrl_successes = 0

    for _, row in tqdm(df.iterrows(),
                       total=len(df), leave=False):
        neg_prob, neg_success = get_correct_prob(
            model,
            row["negation_prompt"],
            row["expected_token"]
        )
        ctrl_prob, ctrl_success = get_correct_prob(
            model,
            row["control_prompt"],
            row["expected_token"]
        )
        neg_probs.append(neg_prob)
        ctrl_probs.append(ctrl_prob)
        if neg_success:
            neg_successes += 1
        if ctrl_success:
            ctrl_successes += 1

    return {
        "model": model_name,
        "neg_success_rate": neg_successes / len(df) * 100,
        "ctrl_success_rate": ctrl_successes / len(df) * 100,
        "avg_neg_prob": np.mean(neg_probs),
        "avg_ctrl_prob": np.mean(ctrl_probs),
        "prob_drop": np.mean(ctrl_probs) - np.mean(neg_probs),
    }

def run_activation_patching(model, df, n_examples=50):
    """
    Runs activation patching on a model.
    Uses n_examples for speed on larger models.
    Returns normalized (n_layers, n_heads) patching matrix.
    """
    N_LAYERS = model.cfg.n_layers
    N_HEADS = model.cfg.n_heads
    all_results = np.zeros((N_LAYERS, N_HEADS))
    successful = 0

    # Sample subset for speed on larger models
    sample_df = df.sample(
        min(n_examples, len(df)),
        random_state=42
    )

    for _, row in tqdm(sample_df.iterrows(),
                       total=len(sample_df), leave=False):
        try:
            correct_tokens = model.to_tokens(
                row["expected_token"], prepend_bos=False)
            false_tokens = model.to_tokens(
                " " + row["target_false"], prepend_bos=False)

            if correct_tokens.shape[1] != 1 or \
               false_tokens.shape[1] != 1:
                continue

            correct_id = model.to_single_token(
                row["expected_token"])
            false_id = false_tokens[0, 0].item()

            ctrl_tokens = model.to_tokens(
                row["control_prompt"])
            neg_tokens = model.to_tokens(
                row["negation_prompt"])

            with torch.no_grad():
                _, ctrl_cache = model.run_with_cache(
                    ctrl_tokens)

            with torch.no_grad():
                neg_logits = model(neg_tokens)
            baseline_ld = (
                neg_logits[0, -1, correct_id] -
                neg_logits[0, -1, false_id]
            ).item()

            with torch.no_grad():
                ctrl_logits = model(ctrl_tokens)
            ideal_ld = (
                ctrl_logits[0, -1, correct_id] -
                ctrl_logits[0, -1, false_id]
            ).item()

            if abs(ideal_ld - baseline_ld) < 0.01:
                continue

            results = np.zeros((N_LAYERS, N_HEADS))

            for layer in range(N_LAYERS):
                cached_v = ctrl_cache[
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

                    with torch.no_grad():
                        patched = model.run_with_hooks(
                            neg_tokens,
                            fwd_hooks=[(
                                f"blocks.{layer}.attn.hook_v",
                                PatchHook(cached_v.clone(), head)
                            )]
                        )

                    patched_ld = (
                        patched[0, -1, correct_id] -
                        patched[0, -1, false_id]
                    ).item()

                    effect = (patched_ld - baseline_ld) / \
                             (ideal_ld - baseline_ld)
                    results[layer][head] = effect

            all_results += results
            successful += 1

        except:
            continue

    if successful > 0:
        return all_results / successful, successful
    return None, 0

# ============================================================
# SECTION 5 — Run All Models
# ============================================================
behavioral_results = []
patching_results = {}

for model_id, model_name in MODELS.items():
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")

    # Load model
    print(f"  Loading {model_id}...")
    try:
        model = HookedTransformer.from_pretrained(model_id)
        model.eval()
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        print(f"  ✓ Loaded — {n_layers} layers, "
              f"{n_heads} heads")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        continue

    # Behavioral analysis
    beh = run_behavioral_analysis(model, model_name, df)
    behavioral_results.append(beh)
    print(f"  Negation success rate: "
          f"{beh['neg_success_rate']:.1f}%")
    print(f"  Control success rate:  "
          f"{beh['ctrl_success_rate']:.1f}%")
    print(f"  Probability drop:      "
          f"{beh['prob_drop']:.4f}")

    # Activation patching
    # Use fewer examples for larger models
    n_examples = 50 if "small" in model_name.lower() \
        else 30
    print(f"  Running activation patching "
          f"({n_examples} examples)...")
    patch_matrix, n_successful = run_activation_patching(
        model, df, n_examples=n_examples)

    if patch_matrix is not None:
        patching_results[model_id] = {
            "matrix": patch_matrix,
            "name": model_name,
            "n_layers": n_layers,
            "n_heads": n_heads,
        }
        print(f"  ✓ Patching complete — "
              f"{n_successful} examples")

        # Print top 5 heads
        flat = patch_matrix.flatten()
        top5_idx = np.argsort(flat)[::-1][:5]
        print(f"  Top 5 heads:")
        for idx in top5_idx:
            l = idx // n_heads
            h = idx % n_heads
            # Normalize position to 0-1 scale
            # for cross-model comparison
            l_norm = l / (n_layers - 1)
            print(f"    L{l}H{h} "
                  f"(depth {l_norm:.2f}): "
                  f"{flat[idx]:.4f}")

    # Free GPU memory before loading next model
    del model
    torch.cuda.empty_cache()

# ============================================================
# SECTION 6 — Behavioral Comparison
# ============================================================
print("\n\n=== Behavioral Comparison Across Models ===\n")

beh_df = pd.DataFrame(behavioral_results)
beh_df.to_csv("cross_model_behavioral.csv", index=False)

print(f"{'Model':<25} {'Neg Success':>12} "
      f"{'Ctrl Success':>13} {'Prob Drop':>10}")
print("-" * 65)
for _, row in beh_df.iterrows():
    print(f"{row['model']:<25} "
          f"{row['neg_success_rate']:>11.1f}% "
          f"{row['ctrl_success_rate']:>12.1f}% "
          f"{row['prob_drop']:>10.4f}")

# ============================================================
# SECTION 7 — Visualize Behavioral Results
# ============================================================
print("\nGenerating behavioral comparison chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Cross-Model Comparison — Does Negation Improve "
    "With Scale?",
    fontsize=13
)

model_labels = [r["model"].split("(")[0].strip()
                for r in behavioral_results]
neg_rates = [r["neg_success_rate"]
             for r in behavioral_results]
ctrl_rates = [r["ctrl_success_rate"]
              for r in behavioral_results]
prob_drops = [r["prob_drop"]
              for r in behavioral_results]

x = np.arange(len(model_labels))
width = 0.35
colors = ["#e74c3c", "#3498db", "#2ecc71"]

# Plot 1 — Success rates
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, neg_rates, width,
                label="Negation prompts",
                color=colors, alpha=0.85,
                edgecolor="white")
bars2 = ax1.bar(x + width/2, ctrl_rates, width,
                label="Control prompts",
                color=colors, alpha=0.4,
                edgecolor="white")

ax1.set_xticks(x)
ax1.set_xticklabels(model_labels, fontsize=9)
ax1.set_ylabel("Success Rate (%)")
ax1.set_title("Negation vs Control Success Rate\n"
              "Across Model Sizes")
ax1.legend()

for bar in bars1:
    ax1.annotate(
        f"{bar.get_height():.1f}%",
        xy=(bar.get_x() + bar.get_width()/2,
            bar.get_height()),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center", fontsize=8
    )

# Plot 2 — Probability drop
ax2 = axes[1]
bars3 = ax2.bar(model_labels, prob_drops,
                color=colors, alpha=0.85,
                edgecolor="white")
ax2.set_ylabel("Probability Drop (Control − Negation)")
ax2.set_title("Confidence Drop Due To Negation\n"
              "Smaller = model handles negation better")

for bar in bars3:
    ax2.annotate(
        f"{bar.get_height():.4f}",
        xy=(bar.get_x() + bar.get_width()/2,
            bar.get_height()),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center", fontsize=9
    )

plt.tight_layout()
plt.savefig("cross_model_behavior.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved cross_model_behavior.png")

# ============================================================
# SECTION 8 — Patching Heatmaps Side By Side
# ============================================================
if len(patching_results) > 0:
    print("\nGenerating patching heatmaps...")

    n_models = len(patching_results)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(7 * n_models, 10)
    )
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        "Activation Patching Across Model Sizes\n"
        "Do the same circuits emerge at larger scale?",
        fontsize=13
    )

    # Use same color scale across all models
    vmax = max(
        r["matrix"].max()
        for r in patching_results.values()
    )

    for idx, (model_id, result) in \
            enumerate(patching_results.items()):
        ax = axes[idx]
        im = ax.imshow(
            result["matrix"],
            cmap="RdBu_r",
            aspect="auto",
            vmin=-vmax,
            vmax=vmax
        )
        ax.set_title(
            result["name"].split("(")[0].strip(),
            fontsize=10
        )
        ax.set_xlabel("Head", fontsize=8)
        if idx == 0:
            ax.set_ylabel("Layer", fontsize=8)

        n_l = result["n_layers"]
        n_h = result["n_heads"]
        ax.set_xticks(range(n_h))
        ax.set_yticks(range(n_l))
        ax.set_xticklabels(
            [f"H{i}" for i in range(n_h)],
            fontsize=6
        )
        ax.set_yticklabels(
            [f"L{i}" for i in range(n_l)],
            fontsize=6
        )
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig("cross_model_patching.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("✓ Saved cross_model_patching.png")

# ============================================================
# SECTION 9 — Summary For Paper
# ============================================================
print("\n=== Key Findings For Paper ===\n")

if len(behavioral_results) >= 2:
    small = behavioral_results[0]
    medium = behavioral_results[1]
    print(f"Scaling from Small → Medium:")
    print(f"  Negation success: "
          f"{small['neg_success_rate']:.1f}% → "
          f"{medium['neg_success_rate']:.1f}%")
    print(f"  Prob drop: "
          f"{small['prob_drop']:.4f} → "
          f"{medium['prob_drop']:.4f}")

    if medium["neg_success_rate"] > \
       small["neg_success_rate"]:
        print(f"\n✓ SCALING HELPS: Larger models handle "
              f"negation better")
    else:
        print(f"\n→ SCALING NEUTRAL: Negation difficulty "
              f"persists across model sizes")
        print(f"  This strengthens the distributed "
              f"processing hypothesis")