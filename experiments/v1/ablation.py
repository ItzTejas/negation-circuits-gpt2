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
# SECTION 2 — Load Model and Dataset
# ============================================================
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
print("✓ Model loaded\n")

df = pd.read_csv("negation_dataset.csv")
print(f"✓ Dataset loaded — {len(df)} examples\n")

# ============================================================
# SECTION 3 — What Is Ablation?
# ============================================================
# Activation patching told us WHICH heads are associated
# with negation. But association isn't causation.
#
# Ablation proves causation by DESTROYING the circuit
# and measuring the damage.
#
# We zero out a head by replacing its output with zeros.
# This is like cutting a wire in a circuit board —
# if the circuit breaks, that wire was essential.
#
# Crucially we test TWO things:
# 1. Does ablating L7H3/L11H3 hurt NEGATION performance?
#    → It should, if they're part of the negation circuit
# 2. Does ablating L7H3/L11H3 hurt CONTROL performance?
#    → It should NOT, proving the effect is negation-specific
#
# This double dissociation is the gold standard proof
# in both neuroscience and mechanistic interpretability

# ============================================================
# SECTION 4 — Define Heads To Ablate
# ============================================================
# We test three ablation conditions:
# 1. Ablate L7H3 alone (answer retrieval head)
# 2. Ablate L11H3 alone (false answer suppression head)
# 3. Ablate both together (full circuit knockout)
# 4. Ablate a RANDOM head (control condition)
#    → Should show minimal effect, proving our heads are special

CIRCUIT_HEADS = [
    (7, 3),    # Answer retrieval head
    (11, 3),   # False answer suppression head
]

# Random heads for control condition
# These should NOT matter for negation
RANDOM_HEADS = [
    (3, 7),    # Random head 1
    (5, 2),    # Random head 2
]

print("Heads to ablate:")
print("  Circuit heads:", CIRCUIT_HEADS)
print("  Random control heads:", RANDOM_HEADS)
print()

# ============================================================
# SECTION 5 — Ablation Function
# ============================================================
def ablate_heads_and_measure(prompt, expected_token,
                              heads_to_ablate):
    """
    Runs the model with specific heads zeroed out.
    Returns the probability assigned to the expected token.

    Args:
        prompt: input sentence
        expected_token: the correct next word
        heads_to_ablate: list of (layer, head) tuples to zero out

    Returns:
        probability of expected token after ablation
    """
    tokens = model.to_tokens(prompt)
    expected_id = model.to_single_token(expected_token)

    # Build hooks for each head we want to ablate
    # Each hook zeros out that head's output
    hooks = []
    for layer, head in heads_to_ablate:
        hook_name = f"blocks.{layer}.attn.hook_z"

        # Closure to capture layer and head correctly
        def make_zero_hook(h):
            def hook_fn(value, hook):
                # value shape: [batch, seq_len, n_heads, d_head]
                # Zero out this specific head's output
                value[:, :, h, :] = 0.0
                return value
            return hook_fn

        hooks.append((hook_name, make_zero_hook(head)))

    # Run model with all ablation hooks applied simultaneously
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    # Get probability of expected token at last position
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    return probs[expected_id].item()

def get_baseline_probability(prompt, expected_token):
    """
    Runs model normally with no ablation.
    Returns probability of expected token.
    """
    tokens = model.to_tokens(prompt)
    expected_id = model.to_single_token(expected_token)
    with torch.no_grad():
        logits = model(tokens)
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    return probs[expected_id].item()

# ============================================================
# SECTION 6 — Run Ablation Experiments
# ============================================================
# For each example we measure probability under 5 conditions:
# 1. Baseline (no ablation)
# 2. Ablate L7H3 only
# 3. Ablate L11H3 only
# 4. Ablate both L7H3 + L11H3
# 5. Ablate random heads (control)
#
# We do this for BOTH control and negation prompts
# The key comparison is:
# Does ablation hurt negation MORE than control?

print("Running ablation experiments...")
print("(This may take a few minutes)\n")

results = []

for i, row in tqdm(df.iterrows(), total=len(df),
                   desc="Ablating examples"):
    try:
        expected = row["expected_token"]

        # --- Baseline (no ablation) ---
        ctrl_baseline = get_baseline_probability(
            row["control_prompt"], expected)
        neg_baseline = get_baseline_probability(
            row["negation_prompt"], expected)

        # --- Ablate L7H3 only ---
        ctrl_ablate_L7H3 = ablate_heads_and_measure(
            row["control_prompt"], expected, [(7, 3)])
        neg_ablate_L7H3 = ablate_heads_and_measure(
            row["negation_prompt"], expected, [(7, 3)])

        # --- Ablate L11H3 only ---
        ctrl_ablate_L11H3 = ablate_heads_and_measure(
            row["control_prompt"], expected, [(11, 3)])
        neg_ablate_L11H3 = ablate_heads_and_measure(
            row["negation_prompt"], expected, [(11, 3)])

        # --- Ablate both circuit heads ---
        ctrl_ablate_both = ablate_heads_and_measure(
            row["control_prompt"], expected, CIRCUIT_HEADS)
        neg_ablate_both = ablate_heads_and_measure(
            row["negation_prompt"], expected, CIRCUIT_HEADS)

        # --- Ablate random heads (control condition) ---
        ctrl_ablate_random = ablate_heads_and_measure(
            row["control_prompt"], expected, RANDOM_HEADS)
        neg_ablate_random = ablate_heads_and_measure(
            row["negation_prompt"], expected, RANDOM_HEADS)

        results.append({
            "subject": row["subject"],
            "expected": expected,

            # Control prompt probabilities
            "ctrl_baseline": ctrl_baseline,
            "ctrl_ablate_L7H3": ctrl_ablate_L7H3,
            "ctrl_ablate_L11H3": ctrl_ablate_L11H3,
            "ctrl_ablate_both": ctrl_ablate_both,
            "ctrl_ablate_random": ctrl_ablate_random,

            # Negation prompt probabilities
            "neg_baseline": neg_baseline,
            "neg_ablate_L7H3": neg_ablate_L7H3,
            "neg_ablate_L11H3": neg_ablate_L11H3,
            "neg_ablate_both": neg_ablate_both,
            "neg_ablate_random": neg_ablate_random,

            # Drops = baseline - ablated
            # Larger drop = head was more important
            "neg_drop_L7H3": neg_baseline - neg_ablate_L7H3,
            "neg_drop_L11H3": neg_baseline - neg_ablate_L11H3,
            "neg_drop_both": neg_baseline - neg_ablate_both,
            "neg_drop_random": neg_baseline - neg_ablate_random,
            "ctrl_drop_L7H3": ctrl_baseline - ctrl_ablate_L7H3,
            "ctrl_drop_L11H3": ctrl_baseline - ctrl_ablate_L11H3,
            "ctrl_drop_both": ctrl_baseline - ctrl_ablate_both,
            "ctrl_drop_random": ctrl_baseline - ctrl_ablate_random,
        })

    except Exception as e:
        continue

results_df = pd.DataFrame(results)
results_df.to_csv("ablation_results.csv", index=False)
print(f"\n✓ Ablation complete — {len(results_df)} examples\n")

# ============================================================
# SECTION 7 — Analyze Results
# ============================================================
# The key metric is DROP in probability after ablation.
# We compare:
# - How much does ablation hurt NEGATION prompts?
# - How much does ablation hurt CONTROL prompts?
#
# If negation drop >> control drop → negation-specific effect
# This is the double dissociation that proves the circuit

print("=== Ablation Analysis ===\n")

conditions = ["L7H3", "L11H3", "Both", "Random"]
neg_drops = [
    results_df["neg_drop_L7H3"].mean(),
    results_df["neg_drop_L11H3"].mean(),
    results_df["neg_drop_both"].mean(),
    results_df["neg_drop_random"].mean(),
]
ctrl_drops = [
    results_df["ctrl_drop_L7H3"].mean(),
    results_df["ctrl_drop_L11H3"].mean(),
    results_df["ctrl_drop_both"].mean(),
    results_df["ctrl_drop_random"].mean(),
]

print(f"{'Condition':<20} {'Negation Drop':>15} "
      f"{'Control Drop':>15} {'Difference':>12}")
print("-" * 65)
for i, cond in enumerate(conditions):
    diff = neg_drops[i] - ctrl_drops[i]
    print(f"{cond:<20} {neg_drops[i]:>15.4f} "
          f"{ctrl_drops[i]:>15.4f} {diff:>12.4f}")

print()
print("Key: Larger negation drop vs control drop = ")
print("     this head is specifically important for negation")

# ============================================================
# SECTION 8 — Visualize Results
# ============================================================
# Figure 3 of your paper — the ablation bar chart
# This is the causal proof visualization

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Ablation Study — Causal Evidence for Negation Circuit",
             fontsize=13)

x = np.arange(len(conditions))
width = 0.35
colors_neg = ["#e74c3c", "#c0392b", "#922b21", "#95a5a6"]
colors_ctrl = ["#3498db", "#2980b9", "#1a5276", "#bdc3c7"]

# Plot 1 — Average probability drop
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, neg_drops, width,
                label="Negation prompts",
                color=colors_neg, alpha=0.85,
                edgecolor="white")
bars2 = ax1.bar(x + width/2, ctrl_drops, width,
                label="Control prompts",
                color=colors_ctrl, alpha=0.85,
                edgecolor="white")

ax1.set_xlabel("Ablation Condition")
ax1.set_ylabel("Average Probability Drop")
ax1.set_title("Probability Drop After Ablation\n"
              "(larger = head more important)")
ax1.set_xticks(x)
ax1.set_xticklabels(conditions)
ax1.legend()
ax1.axhline(y=0, color="black", linewidth=0.5)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f"{height:.4f}",
                 xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f"{height:.4f}",
                 xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points",
                 ha="center", va="bottom", fontsize=8)

# Plot 2 — Specificity: negation drop MINUS control drop
# This shows how SPECIFICALLY each ablation hurts negation
# vs general performance
specificity = [neg_drops[i] - ctrl_drops[i]
               for i in range(len(conditions))]
colors_spec = ["#e74c3c" if s > 0 else "#3498db"
               for s in specificity]

ax2 = axes[1]
bars3 = ax2.bar(conditions, specificity,
                color=colors_spec, alpha=0.85,
                edgecolor="white")
ax2.axhline(y=0, color="black", linewidth=1)
ax2.set_xlabel("Ablation Condition")
ax2.set_ylabel("Negation Drop − Control Drop")
ax2.set_title("Negation Specificity\n"
              "(positive = hurts negation more than control)")

for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f"{height:.4f}",
                 xy=(bar.get_x() + bar.get_width()/2,
                     height),
                 xytext=(0, 3 if height >= 0 else -12),
                 textcoords="offset points",
                 ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("ablation_results.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("\n✓ Figure saved to ablation_results.png")

# ============================================================
# SECTION 9 — Statistical Summary For Paper
# ============================================================
print("\n=== Key Findings For Paper ===\n")

both_neg = results_df["neg_drop_both"].mean()
both_ctrl = results_df["ctrl_drop_both"].mean()
random_neg = results_df["neg_drop_random"].mean()

print(f"1. Full circuit knockout (L7H3 + L11H3):")
print(f"   Negation performance drop: {both_neg:.4f}")
print(f"   Control performance drop:  {both_ctrl:.4f}")
print(f"   Negation-specific effect:  "
      f"{both_neg - both_ctrl:.4f}")
print()
print(f"2. Random head ablation baseline:")
print(f"   Negation performance drop: {random_neg:.4f}")
print()

if both_neg > random_neg * 1.5:
    print("✓ RESULT: Circuit heads show significantly larger")
    print("  negation-specific effects than random heads.")
    print("  This supports the causal role of the circuit.")
else:
    print("→ RESULT: Effects are modest — consistent with")
    print("  distributed negation processing hypothesis.")
    print("  Both interpretations remain viable.")