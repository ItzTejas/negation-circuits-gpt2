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
# We're visualizing what the top heads are actually "looking at"
# when they process negation vs control prompts
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# SECTION 3 — What Is Attention Pattern Visualization?
# ============================================================
# Every attention head produces an "attention pattern" —
# a matrix that shows which tokens it pays attention to.
#
# For a sentence like:
# "The mother tongue of Danielle Darrieux is not English, it is"
# Each head assigns weights to every pair of tokens:
# - How much does "is" attend to "not"?
# - How much does "it" attend to "English"?
# - How much does the last token attend to "Darrieux"?
#
# By visualizing these patterns for our top heads (L11H3, L6H5)
# we can see EXACTLY what those heads are doing:
# - Are they tracking the negation word "not"?
# - Are they suppressing the false answer "English"?
# - Are they boosting attention to the subject?
#
# This turns our patching finding from "this head matters"
# into "this head matters BECAUSE it does X"

# ============================================================
# SECTION 4 — Our Top Heads From Patching
# ============================================================
# These are the heads we identified as most important
# We'll visualize all of them for both control and negation

TOP_HEADS = [
    (11, 3),   # Rank 1 — strongest positive effect
    (6, 5),    # Rank 2 — almost equal to rank 1
    (7, 3),    # Rank 3
    (6, 6),    # Most negative effect — inhibitory head
]

print(f"Visualizing attention patterns for {len(TOP_HEADS)} heads:")
for layer, head in TOP_HEADS:
    print(f"  Layer {layer}, Head {head}")
print()

# ============================================================
# SECTION 5 — Pick Representative Examples
# ============================================================
# We'll visualize 3 examples:
# 1. Best negation case — model handles it well
# 2. Worst negation case — model struggles
# 3. Average case — typical behavior
#
# This gives a complete picture for the paper

def get_correct_token_probability(prompt, expected_token):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    expected_id = model.to_single_token(expected_token)
    return probs[expected_id].item()

print("Finding representative examples...")

prob_diffs = []
for i, row in df.iterrows():
    ctrl_prob = get_correct_token_probability(
        row["control_prompt"], row["expected_token"])
    neg_prob = get_correct_token_probability(
        row["negation_prompt"], row["expected_token"])
    prob_diffs.append(neg_prob - ctrl_prob)

df["prob_diff"] = prob_diffs

# Pick best, worst, and median examples
best_idx = df["prob_diff"].idxmax()
worst_idx = df["prob_diff"].idxmin()
median_idx = df.iloc[(df["prob_diff"] - df["prob_diff"].median()).abs()
                      .argsort()[:1]].index[0]

examples = {
    "Best (model handles negation well)": df.loc[best_idx],
    "Median (typical behavior)": df.loc[median_idx],
    "Worst (model struggles with negation)": df.loc[worst_idx],
}

print(f"✓ Selected 3 representative examples\n")

# ============================================================
# SECTION 6 — Attention Pattern Extraction Function
# ============================================================
def get_attention_pattern(prompt, layer, head):
    """
    Runs the model on a prompt and extracts the attention
    pattern for a specific layer and head.

    Returns:
        pattern: (seq_len, seq_len) attention weight matrix
        tokens: list of token strings for axis labels
    """
    tokens = model.to_tokens(prompt)

    # Get token strings for labeling the axes
    token_strs = [model.to_string(tokens[0, i])
                  for i in range(tokens.shape[1])]

    # Run model and cache attention patterns
    # hook_pattern contains the softmaxed attention weights
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=f"blocks.{layer}.attn.hook_pattern"
        )

    # Extract pattern for our specific head
    # Shape: [batch, n_heads, seq_len, seq_len]
    # We want [seq_len, seq_len] for head h
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
    pattern = pattern[0, head, :, :].cpu().numpy()

    return pattern, token_strs

# ============================================================
# SECTION 7 — Visualize Attention Patterns
# ============================================================
# For each top head we create a figure showing:
# LEFT: attention pattern on CONTROL prompt
# RIGHT: attention pattern on NEGATION prompt
#
# The difference between left and right IS the head's
# response to negation — this is what we publish

print("Generating attention visualizations...")

for layer, head in TOP_HEADS:
    fig, axes = plt.subplots(
        len(examples), 2,
        figsize=(16, 5 * len(examples))
    )
    fig.suptitle(
        f"Attention Patterns — Layer {layer}, Head {head}\n"
        f"Left: Control Prompt | Right: Negation Prompt",
        fontsize=13, y=1.01
    )

    for row_idx, (example_name, example) in enumerate(examples.items()):
        # Get attention patterns for both prompts
        ctrl_pattern, ctrl_tokens = get_attention_pattern(
            example["control_prompt"], layer, head)
        neg_pattern, neg_tokens = get_attention_pattern(
            example["negation_prompt"], layer, head)

        # Plot control attention pattern
        ax_ctrl = axes[row_idx, 0]
        im = ax_ctrl.imshow(ctrl_pattern, cmap="Blues",
                            vmin=0, vmax=1, aspect="auto")
        ax_ctrl.set_xticks(range(len(ctrl_tokens)))
        ax_ctrl.set_yticks(range(len(ctrl_tokens)))
        ax_ctrl.set_xticklabels(ctrl_tokens, rotation=45,
                                ha="right", fontsize=8)
        ax_ctrl.set_yticklabels(ctrl_tokens, fontsize=8)
        ax_ctrl.set_title(f"{example_name}\nControl",
                          fontsize=9)
        plt.colorbar(im, ax=ax_ctrl, fraction=0.046)

        # Plot negation attention pattern
        ax_neg = axes[row_idx, 1]
        im2 = ax_neg.imshow(neg_pattern, cmap="Blues",
                             vmin=0, vmax=1, aspect="auto")
        ax_neg.set_xticks(range(len(neg_tokens)))
        ax_neg.set_yticks(range(len(neg_tokens)))
        ax_neg.set_xticklabels(neg_tokens, rotation=45,
                               ha="right", fontsize=8)
        ax_neg.set_yticklabels(neg_tokens, fontsize=8)
        ax_neg.set_title(f"{example_name}\nNegation",
                         fontsize=9)
        plt.colorbar(im2, ax=ax_neg, fraction=0.046)

    plt.tight_layout()
    filename = f"attention_L{layer}H{head}.png"
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved {filename}")

# ============================================================
# SECTION 8 — Last Token Attention Analysis
# ============================================================
# The last token position is where the prediction happens.
# We specifically look at what the last token attends to
# in our top heads — this is the most informative slice
# because it directly influences what word gets predicted next

print("\n=== Last Token Attention Analysis ===")
print("(What does the prediction position attend to?)\n")

for layer, head in TOP_HEADS:
    print(f"--- Layer {layer}, Head {head} ---")

    for example_name, example in examples.items():
        # Get negation pattern
        neg_pattern, neg_tokens = get_attention_pattern(
            example["negation_prompt"], layer, head)

        # Last token row = what the prediction attends to
        last_token_attn = neg_pattern[-1, :]

        # Find top 3 attended tokens
        top3_idx = np.argsort(last_token_attn)[::-1][:3]

        print(f"  {example_name}:")
        for idx in top3_idx:
            token = neg_tokens[idx]
            weight = last_token_attn[idx]
            # Flag if attending to negation-relevant tokens
            flag = ""
            if token.strip().lower() in ["not", "no",
                                          "never", "n't"]:
                flag = " ← NEGATION TOKEN"
            elif token == example["expected_token"].strip():
                flag = " ← CORRECT ANSWER"
            elif token == example["target_false"]:
                flag = " ← FALSE ANSWER"
            print(f"    '{token}' — {weight:.3f}{flag}")
        print()

print("✓ Analysis complete")
print("\nKey question: Do top heads attend to 'not' more")
print("in negation prompts than control prompts?")
print("If yes → these heads are detecting negation.")
print("If no  → they may be doing something more subtle.")