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
from transformer_lens import HookedTransformer
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# ============================================================
# SECTION 2 — Load Model
# ============================================================
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
print("✓ Model loaded\n")

# ============================================================
# SECTION 3 — Load Dataset
# ============================================================
print("Loading CounterFact dataset...")
dataset = load_dataset("azhx/counterfact", split="train")
print(f"✓ Dataset loaded — {len(dataset)} total examples\n")

# ============================================================
# SECTION 4 — Build Prompts
# ============================================================
# Same as before but we scan ALL 19,728 examples
# instead of capping at 500
# We also add fact category labels for stratified analysis

print("Building prompt pairs...")

CATEGORY_KEYWORDS = {
    "language":    ["tongue", "language", "speaks",
                    "spoken", "dialect"],
    "geography":   ["located", "capital", "country",
                    "city", "born", "headquartered"],
    "occupation":  ["occupation", "profession", "works",
                    "employed", "position"],
    "nationality": ["nationality", "citizen", "national"],
    "science":     ["element", "discovered", "invented",
                    "developed", "created"],
    "other":       []
}

def get_category(prompt_template):
    """Assign a fact category based on prompt template."""
    template_lower = prompt_template.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "other":
            continue
        for kw in keywords:
            if kw in template_lower:
                return category
    return "other"

raw_examples = []

for entry in tqdm(dataset, desc="Building prompts"):
    rr = entry["requested_rewrite"]
    prompt_template = rr["prompt"]
    subject = rr["subject"]
    target_true = rr["target_true"]["str"]
    target_false = rr["target_new"]["str"]

    filled_prompt = prompt_template.format(subject)
    negation_prompt = filled_prompt + \
        f" not {target_false}, it is"
    control_prompt = filled_prompt
    category = get_category(prompt_template)

    raw_examples.append({
        "negation_prompt": negation_prompt,
        "control_prompt": control_prompt,
        "expected_token": " " + target_true,
        "subject": subject,
        "target_true": target_true,
        "target_false": target_false,
        "category": category,
    })

print(f"✓ Built {len(raw_examples)} prompt pairs\n")

# ============================================================
# SECTION 5 — Single Token Filter
# ============================================================
print("Filtering for single-token answers...")

single_token = []
for item in tqdm(raw_examples,
                 desc="Token filtering"):
    try:
        correct_tokens = model.to_tokens(
            item["expected_token"], prepend_bos=False)
        false_tokens = model.to_tokens(
            " " + item["target_false"], prepend_bos=False)

        # Both correct AND false must be single tokens
        # We need this for cross-model patching later
        if correct_tokens.shape[1] == 1 and \
           false_tokens.shape[1] == 1:
            single_token.append(item)
    except:
        continue

print(f"✓ {len(single_token)} single-token examples\n")

# ============================================================
# SECTION 6 — Model Knowledge Verification
# ============================================================
# Scan ALL single-token examples to maximize yield
# Target: as close to 1000 as possible

print("Verifying model knowledge...")
print("(Scanning all examples — this takes ~10 minutes)\n")

verified = []

for item in tqdm(single_token,
                 desc="Verifying"):
    try:
        tokens = model.to_tokens(item["control_prompt"])
        with torch.no_grad():
            logits = model(tokens)
        last_logits = logits[0, -1, :]
        top5_ids = torch.topk(last_logits, 5).indices
        top5 = [model.to_string(t) for t in top5_ids]

        if item["expected_token"] in top5:
            verified.append(item)
    except:
        continue

print(f"\n✓ {len(verified)} verified examples\n")

# ============================================================
# SECTION 7 — Quality Filter
# ============================================================
print("Applying quality filter...")

df = pd.DataFrame(verified)
df_clean = df[
    df["negation_prompt"].str.contains("not") &
    (df["negation_prompt"].str.len() > 20) &
    ~df["negation_prompt"].str.startswith(",") &
    ~df["negation_prompt"].str.startswith(" ,") &
    (df["subject"].str.len() > 2)
].copy()

print(f"✓ After quality filter: {len(df_clean)} examples\n")

# ============================================================
# SECTION 8 — Category Distribution
# ============================================================
print("=== Category Distribution ===\n")
category_counts = df_clean["category"].value_counts()
for cat, count in category_counts.items():
    pct = count / len(df_clean) * 100
    print(f"  {cat:<15} {count:>4} ({pct:.1f}%)")

# ============================================================
# SECTION 9 — Cross-Model Compatibility Filter
# ============================================================
# Pre-filter for examples that are LIKELY to be known
# by all three GPT-2 variants
# Proxy: short prompt templates correspond to simpler,
# more well-known facts that all models likely know

print("\nApplying cross-model compatibility filter...")

df_clean["prompt_length"] = df_clean[
    "control_prompt"].str.len()

# Short prompts = simpler facts = more likely known by all
df_crossmodel = df_clean[
    df_clean["prompt_length"] <= 60
].copy()

print(f"✓ Cross-model compatible: "
      f"{len(df_crossmodel)} examples")

# ============================================================
# SECTION 10 — Save Both Datasets
# ============================================================

# Full dataset — for single-model experiments
df_clean.to_csv(
    "negation_dataset_v2.csv", index=False)
print(f"\n✓ Full dataset saved: "
      f"{len(df_clean)} examples "
      f"→ negation_dataset_v2.csv")

# Cross-model dataset — for scaling experiments
df_crossmodel.to_csv(
    "negation_dataset_crossmodel.csv", index=False)
print(f"✓ Cross-model dataset saved: "
      f"{len(df_crossmodel)} examples "
      f"→ negation_dataset_crossmodel.csv")

# ============================================================
# SECTION 11 — Summary
# ============================================================
print(f"\n=== Dataset Summary ===")
print(f"Total CounterFact entries:       {len(dataset)}")
print(f"After single-token filter:       {len(single_token)}")
print(f"After model verification:        {len(verified)}")
print(f"After quality filter:            {len(df_clean)}")
print(f"Cross-model compatible:          {len(df_crossmodel)}")
print(f"\nCategory breakdown (full dataset):")
for cat, count in category_counts.items():
    print(f"  {cat:<15} {count}")