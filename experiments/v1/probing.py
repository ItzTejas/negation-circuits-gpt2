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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from transformer_lens import HookedTransformer
from tqdm import tqdm

# ============================================================
# SECTION 2 — Load Model and Dataset
# ============================================================
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
print("✓ Model loaded\n")

df = pd.read_csv("negation_dataset.csv")
print(f"✓ Dataset loaded — {len(df)} examples\n")

# ============================================================
# SECTION 3 — What Is A Probing Classifier?
# ============================================================
# Activation patching told us WHERE negation is processed.
# Attention visualization told us WHAT those heads attend to.
# Ablation told us WHICH heads are causally involved.
#
# Probing classifiers tell us WHEN — at which layer does
# GPT-2 first internally "know" that negation is present?
#
# The idea:
# 1. Run a negation prompt through GPT-2
#    → Save the hidden state at the "not" token position
# 2. Run a control prompt through GPT-2
#    → Save the hidden state at the last token position
# 3. At each layer, train a tiny linear classifier:
#    "Given this hidden state vector, is this negation?"
# 4. Measure accuracy at each layer
#
# We probe at the "not" token position specifically because:
# - It appears in every negation prompt
# - It is causally downstream of the negation structure
# - It is the token the model must process to handle negation
# - It avoids causal masking issues (subject token can't
#   see words that come after it)

N_LAYERS = model.cfg.n_layers  # 12

# ============================================================
# SECTION 4 — Hidden State Extraction Functions
# ============================================================

def get_hidden_states_at_not_token(prompt):
    """
    Runs a negation prompt through GPT-2 and returns
    hidden states at the "not" token position.

    The "not" token is the key token — it appears in every
    negation prompt and is causally downstream of the negation
    structure. Any layer that processes "not" meaningfully
    will show up here.

    Returns dict: layer_idx -> hidden state vector (768-dim)
    OR None if negation token not found in prompt
    """
    tokens = model.to_tokens(prompt)
    token_strs = [model.to_string(tokens[0, i])
                  for i in range(tokens.shape[1])]

    # Find position of negation token
    # We check for all negation forms we tested:
    # "not", "n't", "never", "cannot"
    not_pos = None
    for i, tok in enumerate(token_strs):
        if tok.strip().lower() in ["not", "n't",
                                    "never", "cannot"]:
            not_pos = i
            break

    # Skip this example if negation token not found
    if not_pos is None:
        return None

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name:
                "hook_resid_post" in name
        )

    hidden_states = {}
    for layer in range(N_LAYERS):
        key = f"blocks.{layer}.hook_resid_post"
        # Extract at the negation token position
        # This is where the model processes "not"
        hidden_states[layer] = \
            cache[key][0, not_pos, :].cpu().numpy()

    return hidden_states

def get_hidden_states_at_last_token(prompt):
    """
    Runs a control prompt through GPT-2 and returns
    hidden states at the last token position.

    The last token "is" in control prompts is the closest
    equivalent position to where "not" appears in negation
    prompts — both are at the boundary of where the
    factual completion begins.

    Returns dict: layer_idx -> hidden state vector (768-dim)
    """
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name:
                "hook_resid_post" in name
        )

    hidden_states = {}
    for layer in range(N_LAYERS):
        key = f"blocks.{layer}.hook_resid_post"
        # Last token of control prompt
        hidden_states[layer] = \
            cache[key][0, -1, :].cpu().numpy()

    return hidden_states

# ============================================================
# SECTION 5 — Extract Hidden States
# ============================================================
print("Extracting hidden states...")
print("  Negation prompts → at 'not' token position")
print("  Control prompts  → at last token position\n")

all_hidden_states = {layer: [] for layer in range(N_LAYERS)}
all_labels = {layer: [] for layer in range(N_LAYERS)}
skipped = 0

# Negation prompts — extract at "not" token position
print("Processing negation prompts...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    hs = get_hidden_states_at_not_token(
        row["negation_prompt"])
    if hs is None:
        skipped += 1
        continue
    for layer in range(N_LAYERS):
        all_hidden_states[layer].append(hs[layer])
        all_labels[layer].append(1)  # 1 = negation

# Control prompts — extract at last token position
print("Processing control prompts...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        hs = get_hidden_states_at_last_token(
            row["control_prompt"])
        for layer in range(N_LAYERS):
            all_hidden_states[layer].append(hs[layer])
            all_labels[layer].append(0)  # 0 = control
    except:
        skipped += 1
        continue

print(f"\n✓ Extracted hidden states")
print(f"  Samples per layer: {len(all_labels[0])}")
print(f"  Skipped: {skipped} examples")
print(f"  Hidden state dimension: "
      f"{all_hidden_states[0][0].shape[0]}\n")

# ============================================================
# SECTION 6 — Train Probing Classifiers
# ============================================================
# At each layer we train a logistic regression classifier.
# Logistic regression is intentionally simple — we want to
# know if the information EXISTS in the representation,
# not if a complex classifier can extract it.
#
# We use 5-fold cross-validation to get reliable accuracy
# estimates and avoid overfitting.
#
# Key insight: if a SIMPLE linear classifier can predict
# "is this negation?" from the hidden state alone,
# that means negation is LINEARLY ENCODED at that layer —
# the model has explicitly represented negation as a
# direction in its activation space.

print("Training probing classifiers at each layer...")
print("(5-fold cross-validation)\n")

probe_results = []

for layer in range(N_LAYERS):
    X = np.stack(all_hidden_states[layer])
    y = np.array(all_labels[layer])

    # Standardize — important for logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    cv_scores = cross_val_score(
        clf, X_scaled, y,
        cv=5, scoring="accuracy"
    )

    mean_acc = cv_scores.mean()
    std_acc = cv_scores.std()

    probe_results.append({
        "layer": layer,
        "accuracy": mean_acc,
        "std": std_acc,
    })

    # Visual bar showing distance above chance
    above_chance = mean_acc - 0.5
    bar = "█" * int(max(0, above_chance) * 100)
    print(f"  Layer {layer:2d}: {mean_acc:.3f} ± "
          f"{std_acc:.3f}  {bar}")

probe_df = pd.DataFrame(probe_results)
probe_df.to_csv("probing_results.csv", index=False)
print(f"\n✓ Saved to probing_results.csv\n")

# ============================================================
# SECTION 7 — Key Findings
# ============================================================
print("=== Key Findings ===\n")

max_layer = probe_df.loc[
    probe_df["accuracy"].idxmax(), "layer"]
max_acc = probe_df["accuracy"].max()
chance = 0.5

# Find first layer exceeding meaningful thresholds
threshold_60 = probe_df[probe_df["accuracy"] > 0.60]
threshold_70 = probe_df[probe_df["accuracy"] > 0.70]
threshold_80 = probe_df[probe_df["accuracy"] > 0.80]

first_60 = threshold_60.iloc[0]["layer"] \
    if len(threshold_60) > 0 else None
first_70 = threshold_70.iloc[0]["layer"] \
    if len(threshold_70) > 0 else None
first_80 = threshold_80.iloc[0]["layer"] \
    if len(threshold_80) > 0 else None

print(f"Chance accuracy:        {chance:.3f}")
print(f"Peak probe accuracy:    {max_acc:.3f} "
      f"at Layer {max_layer}")
print(f"First layer >60%:       Layer {first_60}")
print(f"First layer >70%:       Layer {first_70}")
print(f"First layer >80%:       Layer {first_80}")
print()

if max_acc > 0.85:
    print("✓ STRONG: Negation is clearly linearly encoded.")
    print(f"  Model first encodes negation at Layer {first_60}")
elif max_acc > 0.70:
    print("→ MODERATE: Negation is partially linearly encoded.")
    print(f"  Meaningful encoding begins at Layer {first_60}")
elif max_acc > 0.60:
    print("→ WEAK: Some negation signal present but subtle.")
    print(f"  First signal at Layer {first_60}")
else:
    print("→ NEAR CHANCE: Negation not linearly encoded")
    print("  in the residual stream at these positions.")
    print("  Suggests nonlinear or highly distributed encoding.")

# ============================================================
# SECTION 8 — Visualize
# ============================================================
print("\nGenerating probing accuracy plot...")

fig, ax = plt.subplots(figsize=(10, 6))

layers = probe_df["layer"].values
accuracies = probe_df["accuracy"].values
stds = probe_df["std"].values

# Main accuracy curve
ax.plot(layers, accuracies,
        color="#e74c3c", linewidth=2.5,
        marker="o", markersize=6,
        label="Probe accuracy")

# Error band
ax.fill_between(layers,
                accuracies - stds,
                accuracies + stds,
                alpha=0.2, color="#e74c3c")

# Reference lines
ax.axhline(y=0.5, color="gray", linestyle="--",
           alpha=0.7, label="Chance (0.5)")
ax.axhline(y=0.7, color="orange", linestyle="--",
           alpha=0.7, label="70% threshold")
ax.axhline(y=0.8, color="green", linestyle="--",
           alpha=0.5, label="80% threshold")

# Mark peak
ax.axvline(x=max_layer, color="#e74c3c",
           linestyle=":", alpha=0.5,
           label=f"Peak: Layer {max_layer}")

# Mark circuit head layers for reference
for layer, name in [(6, "L6H5"), (7, "L7H3"),
                     (8, "L8H10"), (11, "L11H3")]:
    ax.axvline(x=layer, color="steelblue",
               linestyle=":", alpha=0.3)
    ax.text(layer + 0.1, 0.52, name,
            fontsize=7, color="steelblue", alpha=0.8)

ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Probe Accuracy", fontsize=12)
ax.set_title(
    "Probing Classifier Accuracy Across Layers\n"
    "When does GPT-2 first linearly encode negation?",
    fontsize=13
)
ax.set_xticks(range(N_LAYERS))
ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)])
ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Annotate peak
ax.annotate(
    f"Peak: {max_acc:.3f}\n(Layer {max_layer})",
    xy=(max_layer, max_acc),
    xytext=(max_layer - 3, max_acc + 0.03),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=9
)

plt.tight_layout()
plt.savefig("probing_results.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("✓ Figure saved to probing_results.png")

# ============================================================
# SECTION 9 — Layer by Layer Change
# ============================================================
print("\n=== Layer-by-Layer Accuracy Change ===\n")

for i in range(1, N_LAYERS):
    delta = accuracies[i] - accuracies[i-1]
    bar = "█" * int(abs(delta) * 300)
    direction = "↑" if delta > 0 else "↓"
    print(f"  L{i-1}→L{i}: {direction} {delta:+.4f}  {bar}")

biggest_jump = np.argmax(np.diff(accuracies)) + 1
print(f"\n✓ Biggest accuracy jump: "
      f"Layer {biggest_jump - 1} → Layer {biggest_jump}")
print(f"  This is where the most negation-relevant")
print(f"  computation occurs in the forward pass.")