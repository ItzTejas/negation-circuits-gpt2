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
# torch: tensor operations and GPU computation
# pandas: loading our saved dataset CSV
# matplotlib: plotting graphs of our results
# numpy: numerical operations on arrays
# HookedTransformer: our wiretapped version of GPT-2
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformer_lens import HookedTransformer

# ============================================================
# SECTION 2 — Load Model and Dataset
# ============================================================
# We load the same GPT-2 Small we used in dataset.py
# We also load our verified 107 negation/control pairs
# that we saved to CSV in the last script
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
print("✓ Model loaded\n")

df = pd.read_csv("negation_dataset.csv")
print(f"✓ Dataset loaded — {len(df)} verified examples\n")


# ============================================================
# SECTION 3 — Understanding Logits and Probability
# ============================================================
# When GPT-2 processes a prompt it outputs a number called
# a LOGIT for every word in its vocabulary (~50,000 words)
# The higher the logit for a word, the more confident GPT-2
# is that word comes next.
#
# We convert logits to PROBABILITIES using softmax:
# probability = exp(logit) / sum(exp(all logits))
# This gives us a number between 0 and 1 for each word
#
# For our research we specifically track:
# - The probability GPT-2 assigns to the CORRECT answer
#   on CONTROL prompts vs NEGATION prompts
# - If negation is handled correctly, both should be high
# - If GPT-2 struggles with negation, the negation probability
#   will be lower than the control probability

def get_correct_token_probability(prompt, expected_token):
    """
    Runs a prompt through GPT-2 and returns the probability
    it assigns to the expected next token.

    Args:
        prompt: the input sentence string
        expected_token: the word we expect GPT-2 to predict

    Returns:
        probability: float between 0 and 1
        rank: where the expected token ranks in top predictions
        top5: list of top 5 predicted tokens
    """
    # Convert prompt string to token IDs
    tokens = model.to_tokens(prompt)

    # Run forward pass through GPT-2
    # no_grad() saves memory since we're not training
    with torch.no_grad():
        logits = model(tokens)

    # Get logits at the last position — this is the prediction
    # for the NEXT token after our prompt
    # Shape: [vocab_size] — one number per word in vocabulary
    last_logits = logits[0, -1, :]

    # Convert logits to probabilities using softmax
    probabilities = torch.softmax(last_logits, dim=-1)

    # Get the token ID for our expected answer
    expected_id = model.to_single_token(expected_token)

    # Get the probability assigned to our expected token
    probability = probabilities[expected_id].item()

    # Get the rank of our expected token
    # rank 1 = model's top prediction
    # rank 10 = model's 10th best prediction
    sorted_ids = torch.argsort(probabilities, descending=True)
    rank = (sorted_ids == expected_id).nonzero().item() + 1

    # Get top 5 predictions for inspection
    top5_ids = torch.topk(probabilities, 5).indices
    top5 = [model.to_string(t) for t in top5_ids]

    return probability, rank, top5


# ============================================================
# SECTION 4 — Run Baseline Experiments
# ============================================================
# We now run EVERY example in our dataset through GPT-2
# recording probabilities for both control and negation prompts
#
# This is our BASELINE — it tells us:
# 1. How well does GPT-2 handle factual recall normally?
# 2. How much does adding negation affect its confidence?
# 3. Which examples does it handle negation well vs poorly?
#
# This section alone is publishable as a behavioral analysis

print("Running baseline experiments...")
print("(This may take a few minutes)\n")

results = []

for i, row in df.iterrows():
    # Get probability for control prompt
    # e.g. "The mother tongue of Danielle Darrieux is"
    control_prob, control_rank, control_top5 = get_correct_token_probability(
        row["control_prompt"],
        row["expected_token"]
    )

    # Get probability for negation prompt
    # e.g. "The mother tongue of Danielle Darrieux is not English, it is"
    negation_prob, negation_rank, negation_top5 = get_correct_token_probability(
        row["negation_prompt"],
        row["expected_token"]
    )

    # Calculate the difference in probability
    # Positive = negation prompt is MORE confident (good)
    # Negative = negation prompt is LESS confident (model struggles)
    prob_difference = negation_prob - control_prob

    results.append({
        "subject": row["subject"],
        "expected": row["expected_token"],
        "target_false": row["target_false"],
        "control_prompt": row["control_prompt"],
        "negation_prompt": row["negation_prompt"],
        "control_prob": control_prob,
        "negation_prob": negation_prob,
        "prob_difference": prob_difference,
        "control_rank": control_rank,
        "negation_rank": negation_rank,
        "negation_handled": negation_rank <= 5  # True if correct answer in top 5
    })

    # Print progress every 20 examples
    if (i + 1) % 20 == 0:
        print(f"  Processed {i + 1}/{len(df)} examples...")

results_df = pd.DataFrame(results)
results_df.to_csv("baseline_results.csv", index=False)
print(f"\n✓ Baseline experiments complete\n")

# ============================================================
# SECTION 5 — Analyze Results
# ============================================================
# Now we look at what the numbers actually tell us.
# This analysis directly feeds into your paper's
# "Results" and "Analysis" sections.

print("=== Baseline Analysis ===\n")

# How often does GPT-2 correctly handle negation?
negation_success_rate = results_df["negation_handled"].mean() * 100
print(f"Negation success rate: {negation_success_rate:.1f}%")
print(f"(% of time correct answer is in top 5 for negation prompts)\n")

# Average probabilities
avg_control_prob = results_df["control_prob"].mean()
avg_negation_prob = results_df["negation_prob"].mean()
print(f"Average probability on control prompts:  {avg_control_prob:.4f}")
print(f"Average probability on negation prompts: {avg_negation_prob:.4f}")
print(f"Average drop in probability:             {avg_control_prob - avg_negation_prob:.4f}\n")

# Best and worst negation examples
best_idx = results_df["prob_difference"].idxmax()
worst_idx = results_df["prob_difference"].idxmin()

print("--- Best Negation Example (model handles it well) ---")
print(f"Prompt:   {results_df.loc[best_idx, 'negation_prompt']}")
print(f"Expected: {results_df.loc[best_idx, 'expected']}")
print(f"Control prob:  {results_df.loc[best_idx, 'control_prob']:.4f}")
print(f"Negation prob: {results_df.loc[best_idx, 'negation_prob']:.4f}\n")

print("--- Worst Negation Example (model struggles) ---")
print(f"Prompt:   {results_df.loc[worst_idx, 'negation_prompt']}")
print(f"Expected: {results_df.loc[worst_idx, 'expected']}")
print(f"Control prob:  {results_df.loc[worst_idx, 'control_prob']:.4f}")
print(f"Negation prob: {results_df.loc[worst_idx, 'negation_prob']:.4f}\n")

# ============================================================
# SECTION 6 — Visualize Results
# ============================================================
# We plot the probability distributions for control vs negation
# This becomes FIGURE 1 in your paper
# A good figure tells the story before the reader reads a word

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("GPT-2 Negation Handling — Baseline Analysis", fontsize=14)

# Plot 1 — Probability comparison scatter plot
# Each dot is one example
# X axis = control probability
# Y axis = negation probability
# Dots ABOVE the diagonal line = negation helped
# Dots BELOW the diagonal line = negation hurt
ax1 = axes[0]
ax1.scatter(results_df["control_prob"],
            results_df["negation_prob"],
            alpha=0.6, color="steelblue", s=40)

# Draw diagonal line — points on this line have equal probability
max_prob = max(results_df["control_prob"].max(),
               results_df["negation_prob"].max())
ax1.plot([0, max_prob], [0, max_prob],
         "r--", alpha=0.5, label="Equal probability")

ax1.set_xlabel("Control Prompt Probability")
ax1.set_ylabel("Negation Prompt Probability")
ax1.set_title("Control vs Negation Probability\n(dots below line = model struggles with negation)")
ax1.legend()

# Plot 2 — Distribution of probability differences
# Shows overall whether negation helps or hurts
ax2 = axes[1]
ax2.hist(results_df["prob_difference"],
         bins=20, color="steelblue",
         edgecolor="white", alpha=0.8)
ax2.axvline(x=0, color="red", linestyle="--",
            alpha=0.7, label="No difference")
ax2.axvline(x=results_df["prob_difference"].mean(),
            color="green", linestyle="-",
            alpha=0.7, label=f'Mean: {results_df["prob_difference"].mean():.4f}')

ax2.set_xlabel("Probability Difference (Negation - Control)")
ax2.set_ylabel("Number of Examples")
ax2.set_title("Distribution of Probability Differences\n(negative = negation hurts model confidence)")
ax2.legend()

plt.tight_layout()
plt.savefig("baseline_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Figure saved to baseline_results.png")