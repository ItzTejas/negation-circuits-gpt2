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
# datasets: HuggingFace library for loading research datasets
# torch: for running the model and tensor operations
# pandas: for saving and organizing our data
# HookedTransformer: TransformerLens's version of GPT-2
import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
import pandas as pd

# ============================================================
# SECTION 2 — Load the Model
# ============================================================
# GPT-2 Small — 12 layers, 12 heads, 117M parameters
# .eval() disables dropout so results are deterministic
# meaning the same input always gives the same output
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
print("✓ Model loaded successfully\n")

# ============================================================
# SECTION 3 — Load the CounterFact Dataset
# ============================================================
# CounterFact was built by Meng et al. (2022) — the same team
# behind the famous ROME paper on model editing.
# It contains ~21,000 factual statements in this structure:
#
# "requested_rewrite" contains:
#   - prompt: e.g. "The Eiffel Tower is located in {}"
#   - subject: e.g. "The Eiffel Tower"
#   - target_true: the correct answer e.g. "Paris"
#   - target_new: a counterfactual answer e.g. "Berlin"
#
# This is perfect for us because:
# 1. We know the correct answer (target_true)
# 2. We know a plausible wrong answer (target_new)
# 3. We can construct negation prompts using both
#    e.g. "The Eiffel Tower is not in Berlin, it is in"
#         → expected answer: Paris

print("Loading CounterFact dataset from HuggingFace...")
dataset = load_dataset("azhx/counterfact", split="train")
print(f"✓ Dataset loaded — {len(dataset)} total examples\n")

# ============================================================
# SECTION 4 — Understand the Dataset Structure
# ============================================================
# Let's print one example so we can see exactly what fields
# are available before we start filtering

print("--- Sample Entry ---")
sample = dataset[0]
print(f"Subject: {sample['requested_rewrite']['subject']}")
print(f"Prompt template: {sample['requested_rewrite']['prompt']}")
print(f"True target: {sample['requested_rewrite']['target_true']['str']}")
print(f"False target: {sample['requested_rewrite']['target_new']['str']}")
print()

# ============================================================
# SECTION 5 — Build Negation Prompt Pairs
# ============================================================
# For each CounterFact entry we construct TWO prompts:
#
# NEGATION prompt:
# "The mother tongue of Danielle Darrieux is not English, it is"
# → expected: French
#
# CONTROL prompt (same fact, no negation):
# "The mother tongue of Danielle Darrieux is"
# → expected: French
#
# We need BOTH because our research compares what happens
# internally when negation is present vs absent
# That comparison is how we find the negation circuit

print("Building negation prompt pairs...")

raw_examples = []

for entry in dataset:
    rr = entry["requested_rewrite"]

    prompt_template = rr["prompt"]
    subject = rr["subject"]
    target_true = rr["target_true"]["str"]
    target_false = rr["target_new"]["str"]

    # Fill the template with the subject
    filled_prompt = prompt_template.format(subject)
    # e.g. "The mother tongue of Danielle Darrieux is"

    # Build clean negation prompt by appending to the filled template
    # e.g. "The mother tongue of Danielle Darrieux is not English, it is"
    negation_prompt = filled_prompt + f" not {target_false}, it is"

    # Control prompt is just the filled template as-is
    control_prompt = filled_prompt

    raw_examples.append({
        "negation_prompt": negation_prompt,
        "control_prompt": control_prompt,
        "expected_token": " " + target_true,
        "subject": subject,
        "target_true": target_true,
        "target_false": target_false,
    })

print(f"✓ Built {len(raw_examples)} prompt pairs\n")

# ============================================================
# SECTION 6 — Filter: Keep Only Single Token Answers
# ============================================================
# GPT-2 predicts one token at a time.
# If the correct answer is multiple tokens (e.g. "New York")
# we can't cleanly measure whether it got it right in one step.
# So we ONLY keep examples where the answer is a single token.
# This is standard practice in Mech Interp papers.

print("Filtering for single-token answers...")

single_token_examples = []

for item in raw_examples:
    # Convert expected answer to tokens
    # to_tokens() returns a tensor — we check its length
    expected_tokens = model.to_tokens(item["expected_token"],
                                      prepend_bos=False)

    # Keep only if answer is exactly 1 token
    if expected_tokens.shape[1] == 1:
        single_token_examples.append(item)

print(f"✓ {len(single_token_examples)} examples with single-token answers\n")

# ============================================================
# SECTION 7 — Verify: Model Must Get Control Prompt Right
# ============================================================
# We only want examples where GPT-2 actually knows the fact.
# If it doesn't know "Apple was founded in California" normally,
# it definitely can't handle the negation version.
# So we test every control prompt first and filter out failures.
# This is called "behavioral filtering" in research papers.

print("Verifying model knowledge on control prompts...")
print("(This may take a few minutes...)\n")

verified_examples = []

# We cap at 200 examples for speed — enough for solid research
# More examples = more compute time, 200 is standard for this
MAX_EXAMPLES = 200

for i, item in enumerate(single_token_examples[:500]):
    if len(verified_examples) >= MAX_EXAMPLES:
        break

    # Tokenize the control prompt
    control_tokens = model.to_tokens(item["control_prompt"])

    # Run model forward pass — no gradient needed
    with torch.no_grad():
        logits = model(control_tokens)

    # Get predictions at the last token position
    # logits shape: [1, sequence_length, vocab_size]
    last_logits = logits[0, -1, :]

    # Get top 5 predicted tokens
    top5_ids = torch.topk(last_logits, 5).indices
    top5_tokens = [model.to_string(t) for t in top5_ids]

    # Check if our expected answer is in top 5
    if item["expected_token"] in top5_tokens:
        verified_examples.append(item)

        # Print progress every 10 verified examples
        if len(verified_examples) % 10 == 0:
            print(f"  Verified {len(verified_examples)}/{MAX_EXAMPLES}...")

print(f"\n✓ {len(verified_examples)} verified examples where model knows the fact\n")

# ============================================================
# SECTION 8 — Quality Filter and Save
# ============================================================
# Remove any malformed prompts that slipped through
# A good prompt must:
# 1. Contain the word "not" (negation must be present)
# 2. Be at least 20 characters long (not too short)
# 3. Not start with a comma (broken template artifact)

df_clean = pd.DataFrame(verified_examples)
df_clean = df_clean[
    df_clean["negation_prompt"].str.contains("not") &
    (df_clean["negation_prompt"].str.len() > 20) &
    ~df_clean["negation_prompt"].str.startswith(",")
]

df_clean.to_csv("negation_dataset.csv", index=False)
print(f"✓ After quality filter: {len(df_clean)} examples")
print(f"✓ Saved to negation_dataset.csv")

# ============================================================
# SECTION 9 — Print Summary Statistics
# ============================================================
print(f"\n=== Dataset Summary ===")
print(f"Total CounterFact entries:        {len(dataset)}")
print(f"After single-token filter:        {len(single_token_examples)}")
print(f"After model verification:         {len(verified_examples)}")
print(f"After quality filter:             {len(df_clean)}")
print(f"\nSample verified negation prompt:")
print(f"  {df_clean.iloc[0]['negation_prompt']}")
print(f"  → Expected: {df_clean.iloc[0]['expected_token']}")
print(f"\nSample verified control prompt:")
print(f"  {df_clean.iloc[0]['control_prompt']}")
print(f"  → Expected: {df_clean.iloc[0]['expected_token']}")