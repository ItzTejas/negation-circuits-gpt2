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
from scipy import stats
from transformer_lens import HookedTransformer

# ============================================================
# SECTION 2 — Paths
# ============================================================
# All inputs come from data/v1
# All outputs go to data/v2

REPO = r"C:\Users\vinay\PycharmProjects\negation-circuits-gpt2"
DATA_IN  = os.path.join(REPO, "data", "v1")
DATA_OUT = os.path.join(REPO, "data", "v2")
os.makedirs(DATA_OUT, exist_ok=True)

# ============================================================
# SECTION 3 — Load Cross-Model Dataset
# ============================================================
print("Loading cross-model dataset...")
df = pd.read_csv(
    os.path.join(DATA_IN, "negation_dataset_crossmodel.csv"))
df = df.sample(200, random_state=42).reset_index(drop=True)
print(f"✓ Dataset loaded — {len(df)} examples\n")

MODELS = {
    "gpt2":        "GPT-2 Small  (117M)",
    "gpt2-medium": "GPT-2 Medium (345M)",
    "gpt2-large":  "GPT-2 Large  (774M)",
}

# ============================================================
# SECTION 4 — Helper Functions
# ============================================================

def get_logits_and_probs(model, prompt,
                          correct_token, false_token):
    """
    Returns full logit information for a prompt.
    Tracks prob(correct), prob(false), logit_diff, entropy.
    """
    try:
        tokens = model.to_tokens(prompt)
        correct_id = model.to_single_token(correct_token)
        false_tok = model.to_tokens(
            " " + false_token, prepend_bos=False)
        if false_tok.shape[1] != 1:
            return None
        false_id = false_tok[0, 0].item()
        with torch.no_grad():
            logits = model(tokens)
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        prob_correct = probs[correct_id].item()
        prob_false = probs[false_id].item()
        logit_diff = (last_logits[correct_id] -
                     last_logits[false_id]).item()
        top5_ids = torch.topk(probs, 5).indices.tolist()
        in_top5 = correct_id in top5_ids
        entropy = -(probs * torch.log(probs + 1e-10)
                   ).sum().item()
        return {
            "prob_correct": prob_correct,
            "prob_false": prob_false,
            "logit_diff": logit_diff,
            "in_top5": in_top5,
            "entropy": entropy,
        }
    except:
        return None

# ============================================================
# SECTION 5 — Run All Models
# ============================================================

all_model_results = {}

for model_id, model_name in MODELS.items():
    print(f"\n{'='*50}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*50}")

    model = HookedTransformer.from_pretrained(model_id)
    model.eval()
    print(f"✓ Loaded — {model.cfg.n_layers} layers, "
          f"{model.cfg.n_heads} heads")

    results = []

    for _, row in tqdm(df.iterrows(),
                       total=len(df),
                       desc="  Analyzing"):
        ctrl = get_logits_and_probs(
            model, row["control_prompt"],
            row["expected_token"], row["target_false"])
        neg = get_logits_and_probs(
            model, row["negation_prompt"],
            row["expected_token"], row["target_false"])

        if ctrl is None or neg is None:
            continue

        results.append({
            "subject": row["subject"],
            "expected": row["expected_token"],
            "false": row["target_false"],
            "category": row.get("category", "other"),
            "ctrl_prob_correct": ctrl["prob_correct"],
            "ctrl_prob_false": ctrl["prob_false"],
            "ctrl_logit_diff": ctrl["logit_diff"],
            "ctrl_in_top5": ctrl["in_top5"],
            "ctrl_entropy": ctrl["entropy"],
            "neg_prob_correct": neg["prob_correct"],
            "neg_prob_false": neg["prob_false"],
            "neg_logit_diff": neg["logit_diff"],
            "neg_in_top5": neg["in_top5"],
            "neg_entropy": neg["entropy"],
            "prob_drop": ctrl["prob_correct"] -
                        neg["prob_correct"],
            "false_boost": neg["prob_false"] -
                          ctrl["prob_false"],
            "entropy_change": neg["entropy"] -
                             ctrl["entropy"],
            "logit_diff_change": neg["logit_diff"] -
                                ctrl["logit_diff"],
        })

    results_df = pd.DataFrame(results)
    all_model_results[model_id] = {
        "name": model_name,
        "data": results_df,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
    }

    print(f"\n  Control success:     "
          f"{results_df['ctrl_in_top5'].mean()*100:.1f}%")
    print(f"  Negation success:    "
          f"{results_df['neg_in_top5'].mean()*100:.1f}%")
    print(f"  Avg false boost:     "
          f"{results_df['false_boost'].mean():.4f}")
    print(f"  Avg entropy change:  "
          f"{results_df['entropy_change'].mean():.4f}")
    print(f"  N examples:          {len(results_df)}")

    del model
    torch.cuda.empty_cache()

# ============================================================
# SECTION 6 — Hypothesis Testing
# ============================================================
print("\n\n" + "="*60)
print("HYPOTHESIS TESTING — WHY DOES MEDIUM BEAT LARGE?")
print("="*60)

small_data  = all_model_results["gpt2"]["data"]
medium_data = all_model_results["gpt2-medium"]["data"]
large_data  = all_model_results["gpt2-large"]["data"]

common_subjects = set(small_data["subject"]) & \
                  set(medium_data["subject"]) & \
                  set(large_data["subject"])

small_common  = small_data[
    small_data["subject"].isin(common_subjects)
].set_index("subject")
medium_common = medium_data[
    medium_data["subject"].isin(common_subjects)
].set_index("subject")
large_common  = large_data[
    large_data["subject"].isin(common_subjects)
].set_index("subject")

print(f"\nCommon examples across all models: "
      f"{len(common_subjects)}\n")

# Hypothesis A — False Answer Interference
print("--- Hypothesis A: False Answer Interference ---\n")
for model_id, result in all_model_results.items():
    data = result["data"]
    print(f"  {result['name']:<25} "
          f"false boost: {data['false_boost'].mean():+.4f} "
          f"± {data['false_boost'].std():.4f}")

if len(small_common) > 10 and len(large_common) > 10:
    common_idx = list(common_subjects)[:50]
    s_boost = small_common.loc[
        small_common.index.isin(common_idx),
        "false_boost"].values
    l_boost = large_common.loc[
        large_common.index.isin(common_idx),
        "false_boost"].values
    if len(s_boost) > 5 and len(l_boost) > 5:
        min_len = min(len(s_boost), len(l_boost))
        t_stat, p_val = stats.ttest_rel(
            l_boost[:min_len], s_boost[:min_len])
        print(f"\n  T-test (Large vs Small): "
              f"t={t_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            print(f"  ✓ SIGNIFICANT")
        else:
            print(f"  → Not significant at p<0.05")

# Hypothesis B — Entropy
print("\n--- Hypothesis B: Uncertainty Under Negation ---\n")
for model_id, result in all_model_results.items():
    data = result["data"]
    print(f"  {result['name']:<25} "
          f"entropy change: "
          f"{data['entropy_change'].mean():+.4f} "
          f"(ctrl: {data['ctrl_entropy'].mean():.3f} → "
          f"neg: {data['neg_entropy'].mean():.3f})")

# Hypothesis C — Logit Diff Recovery
print("\n--- Hypothesis C: Answer Confidence Recovery ---\n")
for model_id, result in all_model_results.items():
    data = result["data"]
    avg_ctrl = data["ctrl_logit_diff"].mean()
    avg_neg = data["neg_logit_diff"].mean()
    recovery = (avg_neg / avg_ctrl * 100
                if avg_ctrl != 0 else 0)
    print(f"  {result['name']:<25} "
          f"logit diff: {avg_ctrl:.3f} → "
          f"{avg_neg:.3f} ({recovery:.1f}% recovery)")

# Hypothesis D — Category Breakdown
print("\n--- Hypothesis D: Category-Level Breakdown ---\n")
categories = df["category"].unique()
print(f"{'Category':<15} {'Small':>10} "
      f"{'Medium':>10} {'Large':>10}")
print("-" * 50)

category_results = {}
for cat in sorted(categories):
    row_data = []
    for model_id, result in all_model_results.items():
        cat_data = result["data"][
            result["data"]["category"] == cat]
        rate = cat_data["neg_in_top5"].mean() * 100 \
            if len(cat_data) > 0 else 0.0
        row_data.append(rate)
    category_results[cat] = row_data
    print(f"  {cat:<15} "
          f"{row_data[0]:>9.1f}% "
          f"{row_data[1]:>9.1f}% "
          f"{row_data[2]:>9.1f}%")

# ============================================================
# SECTION 7 — Bootstrap Confidence Intervals
# ============================================================
print("\n\n=== Bootstrap Confidence Intervals (95%) ===\n")

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for mean."""
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1-ci)/2 * 100)
    upper = np.percentile(means, (1+ci)/2 * 100)
    return np.mean(data), lower, upper

print(f"{'Model':<25} {'Success Rate':>15} {'95% CI':>20}")
print("-" * 65)

bootstrap_results = {}
for model_id, result in all_model_results.items():
    data = result["data"]["neg_in_top5"].astype(
        float).values
    mean, lower, upper = bootstrap_ci(data)
    bootstrap_results[model_id] = (mean, lower, upper)
    print(f"  {result['name']:<23} "
          f"{mean*100:>13.1f}% "
          f"[{lower*100:.1f}%, {upper*100:.1f}%]")

if "gpt2-medium" in bootstrap_results and \
   "gpt2-large" in bootstrap_results:
    med_arr = all_model_results[
        "gpt2-medium"]["data"]["neg_in_top5"].astype(
        float).values
    large_arr = all_model_results[
        "gpt2-large"]["data"]["neg_in_top5"].astype(
        float).values
    min_len = min(len(med_arr), len(large_arr))
    t_stat, p_val = stats.ttest_ind(
        med_arr[:min_len], large_arr[:min_len])
    print(f"\n  Medium vs Large: t={t_stat:.3f}, "
          f"p={p_val:.4f}")
    if p_val < 0.05:
        print(f"  ✓ Medium significantly outperforms Large")
    else:
        print(f"  → Not significant at p<0.05")

# ============================================================
# SECTION 8 — Visualization
# ============================================================
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Why Does GPT-2 Medium Outperform Large?\n"
    "Mechanistic Analysis of Non-Monotonic Scaling",
    fontsize=13
)

model_names = ["Small\n(117M)",
               "Medium\n(345M)",
               "Large\n(774M)"]
colors = ["#e74c3c", "#3498db", "#2ecc71"]

# Plot 1 — Success rates with CI
ax1 = axes[0, 0]
means = [bootstrap_results[m][0]*100
         for m in MODELS.keys()]
lowers = [bootstrap_results[m][1]*100
          for m in MODELS.keys()]
uppers = [bootstrap_results[m][2]*100
          for m in MODELS.keys()]
yerr_lo = [m-l for m, l in zip(means, lowers)]
yerr_hi = [u-m for m, u in zip(means, uppers)]
bars = ax1.bar(model_names, means,
               color=colors, alpha=0.85,
               edgecolor="white")
ax1.errorbar(range(len(means)), means,
             yerr=[yerr_lo, yerr_hi],
             fmt="none", color="black",
             capsize=5, linewidth=2)
ax1.set_ylabel("Negation Success Rate (%)")
ax1.set_title("Success Rates with 95% CI")
for bar, mean in zip(bars, means):
    ax1.annotate(f"{mean:.1f}%",
                xy=(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+1),
                ha="center", fontsize=9)

# Plot 2 — False answer boost
ax2 = axes[0, 1]
false_boosts = [all_model_results[m]["data"][
    "false_boost"].mean() for m in MODELS.keys()]
false_stds = [all_model_results[m]["data"][
    "false_boost"].std() for m in MODELS.keys()]
bars2 = ax2.bar(model_names, false_boosts,
                color=colors, alpha=0.85,
                edgecolor="white")
ax2.errorbar(range(len(false_boosts)), false_boosts,
             yerr=false_stds, fmt="none",
             color="black", capsize=5, linewidth=2)
ax2.axhline(y=0, color="black", linewidth=0.5)
ax2.set_ylabel("Probability Boost to False Answer")
ax2.set_title("False Answer Interference (Hypothesis A)")
for bar, val in zip(bars2, false_boosts):
    ax2.annotate(f"{val:+.4f}",
                xy=(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.0002),
                ha="center", fontsize=9)

# Plot 3 — Entropy change
ax3 = axes[1, 0]
entropy_changes = [all_model_results[m]["data"][
    "entropy_change"].mean() for m in MODELS.keys()]
entropy_stds = [all_model_results[m]["data"][
    "entropy_change"].std() for m in MODELS.keys()]
bars3 = ax3.bar(model_names, entropy_changes,
                color=colors, alpha=0.85,
                edgecolor="white")
ax3.errorbar(range(len(entropy_changes)), entropy_changes,
             yerr=entropy_stds, fmt="none",
             color="black", capsize=5, linewidth=2)
ax3.axhline(y=0, color="black", linewidth=0.5)
ax3.set_ylabel("Entropy Change (Negation - Control)")
ax3.set_title("Uncertainty Under Negation (Hypothesis B)")
for bar, val in zip(bars3, entropy_changes):
    ax3.annotate(f"{val:+.3f}",
                xy=(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.01),
                ha="center", fontsize=9)

# Plot 4 — Category breakdown
ax4 = axes[1, 1]
x = np.arange(len(category_results))
width = 0.25
cats = sorted(category_results.keys())
for i, (model_id, result) in \
        enumerate(all_model_results.items()):
    rates = [category_results[cat][i] for cat in cats]
    ax4.bar(x + i*width, rates, width,
            label=model_names[i].replace("\n", " "),
            color=colors[i], alpha=0.85,
            edgecolor="white")
ax4.set_xticks(x + width)
ax4.set_xticklabels([c[:8] for c in cats], fontsize=8)
ax4.set_ylabel("Success Rate (%)")
ax4.set_title("Category-Level Breakdown (Hypothesis D)")
ax4.legend(fontsize=8)

plt.tight_layout()
plt.savefig(
    os.path.join(DATA_OUT, "scaling_analysis.png"),
    dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved data/v2/scaling_analysis.png")

# ============================================================
# SECTION 9 — Save Results CSVs
# ============================================================
for model_id, result in all_model_results.items():
    fname = os.path.join(
        DATA_OUT,
        f"scaling_{model_id.replace('-', '_')}.csv"
    )
    result["data"].to_csv(fname, index=False)
    print(f"✓ Saved {fname}")

# ============================================================
# SECTION 10 — Summary For Paper
# ============================================================
print("\n\n=== KEY FINDINGS FOR PAPER ===\n")

small_neg  = all_model_results["gpt2"]["data"][
    "neg_in_top5"].mean() * 100
medium_neg = all_model_results["gpt2-medium"]["data"][
    "neg_in_top5"].mean() * 100
large_neg  = all_model_results["gpt2-large"]["data"][
    "neg_in_top5"].mean() * 100
small_fb   = all_model_results["gpt2"]["data"][
    "false_boost"].mean()
medium_fb  = all_model_results["gpt2-medium"]["data"][
    "false_boost"].mean()
large_fb   = all_model_results["gpt2-large"]["data"][
    "false_boost"].mean()
small_ec   = all_model_results["gpt2"]["data"][
    "entropy_change"].mean()
medium_ec  = all_model_results["gpt2-medium"]["data"][
    "entropy_change"].mean()
large_ec   = all_model_results["gpt2-large"]["data"][
    "entropy_change"].mean()

print(f"Non-monotonic scaling confirmed:")
print(f"  Small: {small_neg:.1f}% | "
      f"Medium: {medium_neg:.1f}% | "
      f"Large: {large_neg:.1f}%\n")

print(f"False answer interference (Hypothesis A):")
print(f"  Small: {small_fb:+.4f} | "
      f"Medium: {medium_fb:+.4f} | "
      f"Large: {large_fb:+.4f}")
if large_fb > small_fb:
    print(f"  → Large shows MORE interference — "
          f"supports Hypothesis A\n")
else:
    print(f"  → No clear interference pattern\n")

print(f"Uncertainty under negation (Hypothesis B):")
print(f"  Small: {small_ec:+.4f} | "
      f"Medium: {medium_ec:+.4f} | "
      f"Large: {large_ec:+.4f}")
if large_ec > small_ec:
    print(f"  → Large more uncertain — supports Hypothesis B")
else:
    print(f"  → No clear uncertainty pattern")