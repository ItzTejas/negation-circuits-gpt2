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
# SECTION 2 — Paths
# ============================================================
REPO     = r"C:\Users\vinay\PycharmProjects\negation-circuits-gpt2"
DATA_IN  = os.path.join(REPO, "data", "v1")
DATA_OUT = os.path.join(REPO, "data", "v2")
os.makedirs(DATA_OUT, exist_ok=True)

# ============================================================
# SECTION 3 — Load Model and Dataset
# ============================================================
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
print("✓ Model loaded\n")

df = pd.read_csv(
    os.path.join(DATA_IN, "negation_dataset.csv"))
print(f"✓ Dataset loaded — {len(df)} examples\n")

N_LAYERS = model.cfg.n_layers   # 12
N_HEADS  = model.cfg.n_heads    # 12

# ============================================================
# SECTION 4 — What Is Complete Circuit Analysis?
# ============================================================
# Our previous activation patching only patched VALUE vectors
# (hook_v). A complete circuit analysis patches ALL components:
#
# 1. hook_q  — Query vectors
#    What position is each head "looking FROM"?
#    Queries determine which tokens attend to which.
#
# 2. hook_k  — Key vectors
#    What position is each head "looking AT"?
#    Keys determine which tokens are attended to.
#
# 3. hook_v  — Value vectors (already done)
#    What INFORMATION is passed along attention edges?
#    Values carry the actual content.
#
# 4. hook_mlp_out — MLP layer outputs
#    MLPs store factual knowledge (ROME paper finding).
#    Do MLPs also contribute to negation processing?
#
# By patching all four component types we get a complete
# picture of the circuit — which components are necessary
# and which are sufficient for negation processing.

# ============================================================
# SECTION 5 — Patching Function
# ============================================================

def patch_component(control_prompt, negation_prompt,
                    correct_token, false_token,
                    component_type):
    """
    Runs activation patching for one component type
    across all layers and heads (or just layers for MLPs).

    component_type: one of 'q', 'k', 'v', 'mlp'

    Returns:
        results: (n_layers, n_heads) array for attention
                 (n_layers, 1) array for MLP
    """
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

    ctrl_tokens = model.to_tokens(control_prompt)
    neg_tokens  = model.to_tokens(negation_prompt)

    # Cache control activations
    with torch.no_grad():
        _, ctrl_cache = model.run_with_cache(ctrl_tokens)

    # Baseline logit difference (no patching)
    with torch.no_grad():
        neg_logits = model(neg_tokens)
    baseline_ld = (neg_logits[0, -1, correct_id] -
                   neg_logits[0, -1, false_id]).item()

    # Ideal logit difference (control prompt)
    with torch.no_grad():
        ctrl_logits = model(ctrl_tokens)
    ideal_ld = (ctrl_logits[0, -1, correct_id] -
                ctrl_logits[0, -1, false_id]).item()

    if abs(ideal_ld - baseline_ld) < 0.01:
        raise ValueError("No meaningful difference")

    # ---- Attention head patching (Q, K, V) ----
    if component_type in ["q", "k", "v"]:
        hook_name_template = \
            "blocks.{}.attn.hook_{}"
        results = np.zeros((N_LAYERS, N_HEADS))

        for layer in range(N_LAYERS):
            hook_name = hook_name_template.format(
                layer, component_type)
            cached = ctrl_cache[hook_name].detach().clone()

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
                            hook_name,
                            PatchHook(cached.clone(), head)
                        )]
                    )

                patched_ld = (
                    patched[0, -1, correct_id] -
                    patched[0, -1, false_id]
                ).item()

                effect = (patched_ld - baseline_ld) / \
                         (ideal_ld - baseline_ld)
                results[layer][head] = effect

        return results

    # ---- MLP patching ----
    elif component_type == "mlp":
        results = np.zeros((N_LAYERS, 1))

        for layer in range(N_LAYERS):
            hook_name = f"blocks.{layer}.hook_mlp_out"
            cached = ctrl_cache[hook_name].detach().clone()

            class MLPPatchHook:
                def __init__(self, cache):
                    self.cache = cache
                def __call__(self, value, hook):
                    # Patch last token position
                    value[:, -1:, :] = \
                        self.cache[:, -1:, :]
                    return value

            with torch.no_grad():
                patched = model.run_with_hooks(
                    neg_tokens,
                    fwd_hooks=[(
                        hook_name,
                        MLPPatchHook(cached.clone())
                    )]
                )

            patched_ld = (
                patched[0, -1, correct_id] -
                patched[0, -1, false_id]
            ).item()

            effect = (patched_ld - baseline_ld) / \
                     (ideal_ld - baseline_ld)
            results[layer][0] = effect

        return results

    else:
        raise ValueError(
            f"Unknown component type: {component_type}")

# ============================================================
# SECTION 6 — Run Complete Circuit Analysis
# ============================================================
# We patch all 4 component types across all examples
# and average results to find the complete negation circuit

COMPONENTS = ["q", "k", "v", "mlp"]
COMPONENT_NAMES = {
    "q":   "Query Vectors",
    "k":   "Key Vectors",
    "v":   "Value Vectors",
    "mlp": "MLP Layers",
}

all_results = {}
for comp in COMPONENTS:
    if comp == "mlp":
        all_results[comp] = np.zeros((N_LAYERS, 1))
    else:
        all_results[comp] = np.zeros((N_LAYERS, N_HEADS))

successful = {comp: 0 for comp in COMPONENTS}

print("Running complete circuit analysis...")
print("(Patching Q, K, V, and MLP across all examples)\n")

for i, row in tqdm(df.iterrows(),
                   total=len(df),
                   desc="Analyzing examples"):
    for comp in COMPONENTS:
        try:
            result = patch_component(
                row["control_prompt"],
                row["negation_prompt"],
                row["expected_token"],
                row["target_false"],
                comp
            )
            all_results[comp] += result
            successful[comp] += 1
        except:
            continue

# Average results
avg_results = {}
for comp in COMPONENTS:
    if successful[comp] > 0:
        avg_results[comp] = \
            all_results[comp] / successful[comp]
    else:
        avg_results[comp] = all_results[comp]

print("\n✓ Complete circuit analysis done")
for comp in COMPONENTS:
    print(f"  {COMPONENT_NAMES[comp]}: "
          f"{successful[comp]} examples processed")

# Save results
np.save(
    os.path.join(DATA_OUT, "complete_circuit.npy"),
    avg_results,
    allow_pickle=True
)
print("\n✓ Saved complete_circuit.npy\n")

# ============================================================
# SECTION 7 — Identify Top Components Per Type
# ============================================================
print("=== Top Components Per Type ===\n")

top_components = {}

for comp in COMPONENTS:
    results = avg_results[comp]
    flat = results.flatten()
    top5_idx = np.argsort(flat)[::-1][:5]

    print(f"{COMPONENT_NAMES[comp]}:")
    top_list = []
    for idx in top5_idx:
        if comp == "mlp":
            layer = idx
            desc = f"  Layer {layer}"
        else:
            layer = idx // N_HEADS
            head  = idx % N_HEADS
            desc = f"  L{layer}H{head}"
        effect = flat[idx]

        # Flag our previously identified circuit heads
        is_circuit = comp == "v" and \
            (layer, head if comp != "mlp" else 0) \
            in [(7,3), (11,3), (6,5), (8,10)]
        flag = " ← CIRCUIT HEAD" if is_circuit else ""

        print(f"  {desc}: {effect:.4f}{flag}")
        top_list.append((layer,
                         head if comp != "mlp" else 0,
                         effect))

    top_components[comp] = top_list
    print()

# ============================================================
# SECTION 8 — Compare Component Importance
# ============================================================
print("=== Component Type Comparison ===\n")
print("Max patching effect per component type:")
print("(Higher = this component type more important)\n")

comp_max = {}
comp_mean_top5 = {}

for comp in COMPONENTS:
    flat = avg_results[comp].flatten()
    top5 = np.sort(flat)[::-1][:5]
    comp_max[comp] = flat.max()
    comp_mean_top5[comp] = top5.mean()
    print(f"  {COMPONENT_NAMES[comp]:<20} "
          f"max: {comp_max[comp]:.4f}  "
          f"top-5 mean: {comp_mean_top5[comp]:.4f}")

print()
dominant = max(comp_max, key=comp_max.get)
print(f"Most important component type: "
      f"{COMPONENT_NAMES[dominant]}")

# ============================================================
# SECTION 9 — Visualization
# ============================================================
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Complete Circuit Analysis — GPT-2 Negation\n"
    "Activation Patching Across All Component Types",
    fontsize=13
)

vmax = max(avg_results[c].max() for c in COMPONENTS)
vmin = min(avg_results[c].min() for c in COMPONENTS)
abs_max = max(abs(vmax), abs(vmin))

titles = {
    "q":   "Query Vectors\n(what positions attend FROM)",
    "k":   "Key Vectors\n(what positions are attended TO)",
    "v":   "Value Vectors\n(what information is passed)",
    "mlp": "MLP Layers\n(factual knowledge storage)",
}

positions = {"q": (0,0), "k": (0,1),
             "v": (1,0), "mlp": (1,1)}

for comp, (row_idx, col_idx) in positions.items():
    ax = axes[row_idx, col_idx]
    data = avg_results[comp]

    im = ax.imshow(
        data,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-abs_max,
        vmax=abs_max
    )

    ax.set_title(titles[comp], fontsize=10)
    ax.set_ylabel("Layer", fontsize=9)

    if comp == "mlp":
        ax.set_xlabel("MLP Layer", fontsize=9)
        ax.set_xticks([0])
        ax.set_xticklabels(["MLP"])
    else:
        ax.set_xlabel("Attention Head", fontsize=9)
        ax.set_xticks(range(N_HEADS))
        ax.set_xticklabels(
            [f"H{i}" for i in range(N_HEADS)],
            fontsize=7
        )

    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels(
        [f"L{i}" for i in range(N_LAYERS)],
        fontsize=7
    )
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig(
    os.path.join(DATA_OUT, "complete_circuit.png"),
    dpi=150, bbox_inches="tight"
)
plt.show()
print("✓ Saved data/v2/complete_circuit.png")

# ============================================================
# SECTION 10 — Summary For Paper
# ============================================================
print("\n\n=== KEY FINDINGS FOR PAPER ===\n")

print("Component importance ranking:")
ranked = sorted(comp_max.items(),
                key=lambda x: x[1], reverse=True)
for rank, (comp, max_val) in enumerate(ranked, 1):
    print(f"  {rank}. {COMPONENT_NAMES[comp]:<20} "
          f"max effect: {max_val:.4f}")

print()
v_max = comp_max.get("v", 0)
mlp_max = comp_max.get("mlp", 0)

if mlp_max > v_max:
    print("→ MLP layers show STRONGER effects than attention")
    print("  This connects to ROME's finding that facts")
    print("  are stored in MLP layers — negation may")
    print("  require accessing the same storage.")
elif v_max > mlp_max * 1.5:
    print("→ Attention value vectors dominate over MLPs")
    print("  Negation is primarily an attention phenomenon")
    print("  consistent with our activation patching results.")
else:
    print("→ Both attention and MLP contribute to negation")
    print("  The circuit involves both components.")

print()
q_max = comp_max.get("q", 0)
k_max = comp_max.get("k", 0)

if q_max > k_max:
    print("→ Query vectors more important than key vectors")
    print("  Negation processing is driven by WHERE")
    print("  tokens attend FROM rather than TO.")
else:
    print("→ Key vectors more important than query vectors")
    print("  Negation processing is driven by WHICH")
    print("  tokens are being attended TO.")