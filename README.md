# Negation Circuits in GPT-2
### A Mechanistic Interpretability Study

**Author:** Tejas  
**Institution:** Independent Researcher  
**Status:** Under preparation for conference submission  
**Code:** This repository  

---

## Overview

Large language models demonstrate surprisingly poor performance 
on negation despite near-perfect factual recall. This paper 
investigates the internal mechanisms by which GPT-2 processes 
negation using mechanistic interpretability techniques.

We find that negation processing is distributed across 
multiple attention heads with no single dominant component, 
and that scaling does not monotonically improve negation 
performance — GPT-2 Medium outperforms both smaller and 
larger variants.

---

## Key Findings

- GPT-2 Small handles negation correctly only 24.3% of the time
- Adding negation causes a 50% drop in model confidence
- Negation processing is distributed — no single dominant head
- Non-monotonic scaling: Medium (33.6%) outperforms both Small (24.3%) and Large (28.0%)
- Two-tier circuit: universal core (L11H3, L7H3) + type-specific components (L6H5, L8H10)
- Negation form matters: "never/cannot" (15.9%) harder than standard "not" (24.3%)

---

## Repository Structure

```
negation-circuits-gpt2/
├── experiments/
│   ├── dataset.py
│   ├── dataset_v2.py
│   ├── experiment.py
│   ├── activation_patching.py
│   ├── attention_viz.py
│   ├── ablation.py
│   ├── negation_types.py
│   ├── cross_model.py
│   └── probing.py
├── data/
│   ├── negation_dataset.csv
│   ├── negation_dataset_v2.csv
│   └── negation_dataset_crossmodel.csv
├── figures/
│   ├── baseline_results.png
│   ├── patching_heatmap.png
│   ├── ablation_results.png
│   ├── negation_types_behavior.png
│   ├── negation_types_heatmaps.png
│   ├── cross_model_behavior.png
│   └── cross_model_patching.png
└── paper/
    └── draft.md
```

---

## Setup

Clone the repository and install dependencies:

```
pip install transformer-lens transformers datasets
pip install pandas numpy matplotlib seaborn scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## Reproducing Results

Run experiments in this order:

```
1. python experiments/dataset_v2.py
2. python experiments/experiment.py
3. python experiments/activation_patching.py
4. python experiments/attention_viz.py
5. python experiments/ablation.py
6. python experiments/negation_types.py
7. python experiments/cross_model.py
```

---

## Compute Requirements

All experiments were run on a single NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM).

| Experiment | Runtime |
|---|---|
| Dataset construction | ~10 min |
| Behavioral baseline | ~2 min |
| Activation patching | ~5 min |
| Attention visualization | ~1 min |
| Ablation study | ~3 min |
| Negation types | ~20 min |
| Cross-model comparison | ~40 min |

---

## Results Summary

| Model | Negation Success | Control Success | Prob Drop |
|---|---|---|---|
| GPT-2 Small (117M) | 24.3% | 100.0% | 0.0470 |
| GPT-2 Medium (345M) | 33.6% | 86.9% | 0.0515 |
| GPT-2 Large (774M) | 28.0% | 88.8% | 0.1147 |

| Negation Type | Success Rate |
|---|---|
| Standard ("not") | 24.3% |
| Contraction ("isn't") | 28.0% |
| Never | 15.9% |
| Cannot | 15.9% |
| Control (no negation) | 100.0% |

---

## Citation

```
@article{tejas2026negation,
  title={Distributed Negation Processing in GPT-2: A Mechanistic Interpretability Analysis},
  author={Tejas},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT License — see LICENSE for details.

---

## Acknowledgements

This work uses the TransformerLens library by Neel Nanda 
and the CounterFact dataset by Meng et al. (2022).
