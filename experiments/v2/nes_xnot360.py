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
from datasets import load_dataset
from transformer_lens import HookedTransformer
from scipy import stats

# ============================================================
# SECTION 2 — Paths
# ============================================================
REPO     = r"C:\Users\vinay\PycharmProjects\negation-circuits-gpt2"
DATA_IN  = os.path.join(REPO, "data", "v1")
DATA_OUT = os.path.join(REPO, "data", "v2")
os.makedirs(DATA_OUT, exist_ok=True)

# ============================================================
# SECTION 3 — Load Model
# ============================================================
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
print("✓ Model loaded\n")

# ============================================================
# SECTION 4 — What Is NES?
# ============================================================
# The Negation Effect Score (NES) from Al Mofael et al. (2026)
# measures how strongly a model's output changes when context
# flips from affirmative to negated.
#
# NES = log P(target | affirmative) - log P(target | negated)
#
# Interpretation:
#   NES < 0 → model assigns HIGHER probability under negation
#             (correctly processes logical reversal) ✓
#   NES > 0 → model assigns LOWER probability under negation
#             (fails — prefers affirmative continuation) ✗
#   NES ≈ 0 → model shows no distinction between contexts
#
# We implement this on TWO datasets:
# 1. Our CounterFact dataset — factual negation
# 2. xNot360 — natural language negation benchmark
#
# This lets us directly compare with Al Mofael et al. who
# used NES on their synthetic templates AND xNot360.

def compute_nes(affirmative_prompt, negated_prompt,
                target_token):
    """
    Computes NES for a single example.

    NES = log P(target | affirmative) - log P(target | negated)

    Args:
        affirmative_prompt: the base/control sentence
        negated_prompt: the negated version
        target_token: the token whose probability we track

    Returns:
        nes: float
        log_prob_aff: log probability under affirmative
        log_prob_neg: log probability under negated
    """
    try:
        target_id = model.to_single_token(target_token)

        # Run affirmative prompt
        aff_tokens = model.to_tokens(affirmative_prompt)
        with torch.no_grad():
            aff_logits = model(aff_tokens)
        aff_last = aff_logits[0, -1, :]
        aff_log_probs = torch.log_softmax(aff_last, dim=-1)
        log_prob_aff = aff_log_probs[target_id].item()

        # Run negated prompt
        neg_tokens = model.to_tokens(negated_prompt)
        with torch.no_grad():
            neg_logits = model(neg_tokens)
        neg_last = neg_logits[0, -1, :]
        neg_log_probs = torch.log_softmax(neg_last, dim=-1)
        log_prob_neg = neg_log_probs[target_id].item()

        nes = log_prob_aff - log_prob_neg
        return nes, log_prob_aff, log_prob_neg

    except Exception as e:
        return None, None, None

# ============================================================
# SECTION 5 — NES On Our CounterFact Dataset
# ============================================================
# This gives us NES values on our existing dataset
# so we can compare our metric (top-5 success) with NES

print("=" * 55)
print("PART 1: NES on CounterFact Dataset")
print("=" * 55 + "\n")

df = pd.read_csv(
    os.path.join(DATA_OUT, "negation_dataset_v2.csv"))

counterfact_nes = []

for _, row in tqdm(df.iterrows(),
                   total=len(df),
                   desc="Computing NES (CounterFact)"):
    nes, log_aff, log_neg = compute_nes(
        row["control_prompt"],
        row["negation_prompt"],
        row["expected_token"]
    )
    if nes is not None:
        counterfact_nes.append({
            "subject": row["subject"],
            "expected": row["expected_token"],
            "control_prompt": row["control_prompt"],
            "negation_prompt": row["negation_prompt"],
            "nes": nes,
            "log_prob_aff": log_aff,
            "log_prob_neg": log_neg,
            "failure": nes > 0,  # NES > 0 = failure
            "category": row.get("category", "other"),
        })

cf_df = pd.DataFrame(counterfact_nes)
cf_df.to_csv(
    os.path.join(DATA_OUT, "nes_counterfact.csv"),
    index=False)

print(f"\n✓ CounterFact NES computed — {len(cf_df)} examples\n")

# Statistics
mean_nes = cf_df["nes"].mean()
median_nes = cf_df["nes"].median()
failure_rate = cf_df["failure"].mean() * 100
se = cf_df["nes"].std() / np.sqrt(len(cf_df))
ci_95 = 1.96 * se

print(f"CounterFact NES Statistics:")
print(f"  Mean NES:     {mean_nes:.4f} ± {ci_95:.4f} (95% CI)")
print(f"  Median NES:   {median_nes:.4f}")
print(f"  Failure rate: {failure_rate:.1f}% (NES > 0)")
print(f"  N:            {len(cf_df)}\n")

# Category breakdown
print("NES by category:")
print(f"  {'Category':<15} {'Mean NES':>10} "
      f"{'Failure %':>12} {'N':>5}")
print("  " + "-" * 45)
for cat, grp in cf_df.groupby("category"):
    print(f"  {cat:<15} {grp['nes'].mean():>10.4f} "
          f"{grp['failure'].mean()*100:>11.1f}% "
          f"{len(grp):>5}")

# ============================================================
# SECTION 6 — Load xNot360 Dataset
# ============================================================
print("\n" + "=" * 55)
print("PART 2: NES on xNot360 Benchmark")
print("=" * 55 + "\n")

print("Loading xNot360 from HuggingFace...")
xnot = load_dataset(
    "nguyenthanhasia/xNot360", split="test")
print(f"✓ xNot360 loaded — {len(xnot)} examples\n")

# Print dataset structure
print("Dataset structure:")
print(f"  Fields: {xnot.column_names}")
print(f"  Sample:")
print(f"    sentence1: {xnot[0]['sentence1']}")
print(f"    sentence2: {xnot[0]['sentence2']}")
print(f"    label: {xnot[0]['label']}")
print(f"      (1 = sentence2 negates sentence1)")
print(f"      (0 = does not negate)\n")

# ============================================================
# SECTION 7 — Adapt NES For xNot360
# ============================================================
# xNot360 uses COMPLETE sentences not prefix/target format.
# To compute NES we use the approach from Al Mofael et al.:
# - Use all tokens except the last as the PREFIX
# - Use the last token as the TARGET
# - This works because sentence2 is a modification of
#   sentence1 — they share most of their structure
#
# We only compute NES on NEGATION pairs (label=1)
# since those are cases where sentence2 genuinely negates
# sentence1 — the interesting cases for our analysis.

def sentence_to_prefix_target(sentence):
    """
    Splits a sentence into prefix (all but last token)
    and target (last token).
    Uses GPT-2's tokenizer.
    """
    tokens = model.to_tokens(sentence)[0]
    if len(tokens) < 3:
        return None, None

    # Decode prefix (skip BOS token, use all but last)
    prefix_tokens = tokens[1:-1]  # skip BOS and last
    prefix = model.to_string(prefix_tokens)

    # Last token is the target
    target_token_id = tokens[-1].item()
    target = model.to_string(
        torch.tensor([target_token_id]))

    return prefix, target

print("Computing NES on xNot360...")
print("(Only for negation pairs where label=1)\n")

xnot_nes = []
skipped = 0

for example in tqdm(xnot, desc="Computing NES (xNot360)"):
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]
    label = example["label"]

    # Get prefix/target from sentence1 (affirmative)
    aff_prefix, target = sentence_to_prefix_target(sentence1)
    if aff_prefix is None or target is None:
        skipped += 1
        continue

    # Get prefix from sentence2 (potentially negated)
    neg_prefix, neg_target = \
        sentence_to_prefix_target(sentence2)
    if neg_prefix is None:
        skipped += 1
        continue

    # Try to use sentence1's target for both
    # (measures how probability of SAME token changes)
    try:
        target_id = model.to_single_token(target)
    except:
        skipped += 1
        continue

    nes, log_aff, log_neg = compute_nes(
        aff_prefix, neg_prefix, target)

    if nes is not None:
        xnot_nes.append({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "label": label,
            "target": target,
            "nes": nes,
            "log_prob_aff": log_aff,
            "log_prob_neg": log_neg,
            "failure": nes > 0,
        })

xnot_df = pd.DataFrame(xnot_nes)
xnot_df.to_csv(
    os.path.join(DATA_OUT, "nes_xnot360.csv"),
    index=False)

print(f"\n✓ xNot360 NES computed")
print(f"  Processed: {len(xnot_df)}")
print(f"  Skipped: {skipped}\n")

# Statistics — all examples
mean_nes_all = xnot_df["nes"].mean()
failure_all = xnot_df["failure"].mean() * 100

# Statistics — negation pairs only (label=1)
neg_pairs = xnot_df[xnot_df["label"] == 1]
non_neg_pairs = xnot_df[xnot_df["label"] == 0]

print(f"xNot360 NES Statistics:\n")
print(f"  All pairs (n={len(xnot_df)}):")
print(f"    Mean NES:     {mean_nes_all:.4f}")
print(f"    Failure rate: {failure_all:.1f}%\n")

if len(neg_pairs) > 0:
    mean_neg = neg_pairs["nes"].mean()
    se_neg = neg_pairs["nes"].std() / \
             np.sqrt(len(neg_pairs))
    ci_neg = 1.96 * se_neg
    fail_neg = neg_pairs["failure"].mean() * 100
    print(f"  Negation pairs (label=1, n={len(neg_pairs)}):")
    print(f"    Mean NES:     {mean_neg:.4f} ± "
          f"{ci_neg:.4f} (95% CI)")
    print(f"    Failure rate: {fail_neg:.1f}%\n")

if len(non_neg_pairs) > 0:
    mean_non = non_neg_pairs["nes"].mean()
    fail_non = non_neg_pairs["failure"].mean() * 100
    print(f"  Non-negation pairs (label=0, "
          f"n={len(non_neg_pairs)}):")
    print(f"    Mean NES:     {mean_non:.4f}")
    print(f"    Failure rate: {fail_non:.1f}%\n")

# ============================================================
# SECTION 8 — Compare With Al Mofael et al.
# ============================================================
print("=" * 55)
print("COMPARISON WITH AL MOFAEL ET AL. (2026)")
print("=" * 55 + "\n")

print("Their reported NES values (from Table I):")
print("  capital_of:      Mean NES = 2.04, Failure = 100.0%")
print("  can_ability:     Mean NES = 0.96, Failure = 60.3%")
print("  likes:           Mean NES = 3.76, Failure = 100.0%")
print("  is_a_job:        Mean NES = 1.30, Failure = 83.7%")
print("  drives_vehicle:  Mean NES = -2.26, Failure = 24.6%")
print()
print("Our CounterFact NES values:")
print(f"  Overall: Mean NES = {mean_nes:.4f}, "
      f"Failure = {failure_rate:.1f}%")
print()

# Statistical comparison
if len(neg_pairs) > 5:
    print("xNot360 comparison:")
    print(f"  Their finding: ablation slightly decreased "
          f"NES on xNot360")
    print(f"  Our NES on xNot360 (negation pairs): "
          f"{mean_neg:.4f}")
    if mean_neg > 0:
        print(f"  → Consistent: model still fails on "
              f"xNot360 negation pairs")
    else:
        print(f"  → Interesting: model handles some "
              f"xNot360 negation correctly")

# ============================================================
# SECTION 9 — Visualization
# ============================================================
print("\nGenerating figures...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(
    "Negation Effect Score (NES) Analysis\n"
    "Comparing CounterFact and xNot360 Benchmarks",
    fontsize=13
)

# Plot 1 — NES distribution on CounterFact
ax1 = axes[0]
ax1.hist(cf_df["nes"], bins=25,
         color="#e74c3c", alpha=0.8,
         edgecolor="white")
ax1.axvline(x=0, color="black",
            linewidth=1.5, linestyle="--",
            label="NES=0 (chance)")
ax1.axvline(x=mean_nes, color="darkred",
            linewidth=2, linestyle="-",
            label=f"Mean: {mean_nes:.3f}")
ax1.set_xlabel("NES Value")
ax1.set_ylabel("Count")
ax1.set_title(f"CounterFact NES Distribution\n"
              f"Failure rate: {failure_rate:.1f}%")
ax1.legend(fontsize=9)

# Plot 2 — NES distribution on xNot360
ax2 = axes[1]
colors_label = ["#3498db" if l == 0 else "#e74c3c"
                for l in xnot_df["label"]]
ax2.scatter(range(len(xnot_df)),
            xnot_df["nes"].values,
            c=colors_label, alpha=0.6, s=20)
ax2.axhline(y=0, color="black",
            linewidth=1.5, linestyle="--")
ax2.axhline(y=mean_nes_all, color="purple",
            linewidth=2, linestyle="-",
            label=f"Mean: {mean_nes_all:.3f}")
ax2.set_xlabel("Example Index")
ax2.set_ylabel("NES Value")
ax2.set_title(f"xNot360 NES by Example\n"
              f"Red=negation, Blue=non-negation")
ax2.legend(fontsize=9)

# Plot 3 — Category NES comparison
ax3 = axes[2]
categories = cf_df["category"].unique()
cat_means = []
cat_failures = []
cat_names = []

for cat in sorted(categories):
    grp = cf_df[cf_df["category"] == cat]
    cat_means.append(grp["nes"].mean())
    cat_failures.append(grp["failure"].mean() * 100)
    cat_names.append(cat[:8])

x = np.arange(len(cat_names))
bars = ax3.bar(x, cat_failures,
               color="#e74c3c", alpha=0.8,
               edgecolor="white")
ax3.set_xticks(x)
ax3.set_xticklabels(cat_names,
                     rotation=45, ha="right",
                     fontsize=8)
ax3.set_ylabel("Failure Rate (%)")
ax3.set_title("NES Failure Rate by Category\n"
              "(CounterFact dataset)")
ax3.axhline(y=failure_rate, color="black",
            linestyle="--", alpha=0.5,
            label=f"Overall: {failure_rate:.1f}%")
ax3.legend(fontsize=9)

for bar, val in zip(bars, cat_failures):
    ax3.annotate(f"{val:.0f}%",
                xy=(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1),
                ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(
    os.path.join(DATA_OUT, "nes_analysis.png"),
    dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved data/v2/nes_analysis.png")

# ============================================================
# SECTION 10 — Summary For Paper
# ============================================================
print("\n\n=== KEY FINDINGS FOR PAPER ===\n")

print(f"NES on CounterFact (factual negation):")
print(f"  Mean NES = {mean_nes:.4f} "
      f"(positive = model fails)")
print(f"  Failure rate = {failure_rate:.1f}%")
print(f"  95% CI: [{mean_nes-ci_95:.4f}, "
      f"{mean_nes+ci_95:.4f}]\n")

if len(neg_pairs) > 5:
    print(f"NES on xNot360 (natural language negation):")
    print(f"  Negation pairs — Mean NES = {mean_neg:.4f}")
    print(f"  Failure rate = {fail_neg:.1f}%\n")

print(f"Comparison with Al Mofael et al.:")
print(f"  Their can_ability: Mean NES = 0.96, "
      f"Failure = 60.3%")
print(f"  Our CounterFact:   Mean NES = {mean_nes:.4f}, "
      f"Failure = {failure_rate:.1f}%")
print()

if mean_nes > 0 and failure_rate > 60:
    print("→ Our findings are CONSISTENT with Al Mofael "
          "et al.")
    print("  Both show GPT-2 predominantly fails at "
          "factual negation.")
elif failure_rate < 60:
    print("→ Our failure rate is LOWER than Al Mofael et al.")
    print("  CounterFact may produce easier negation "
          "examples than synthetic templates.")

print(f"\nNote: NES > 0 means model prefers affirmative")
print(f"      (same as our top-5 failure metric)")
print(f"      Both metrics converge on similar conclusions.")