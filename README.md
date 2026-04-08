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

We find that negation processing is **distributed** across 
multiple attention heads with no single dominant component, 
and that scaling does not monotonically improve negation 
performance — GPT-2 Medium outperforms both smaller and 
larger variants.

---

## Key Findings

- GPT-2 Small handles negation correctly only **24.3%** of the time
- Adding negation causes a **50% drop** in model confidence
- Negation processing is **distributed** — no single dominant head
- **Non-monotonic scaling:** Medium (33.6%) outperforms both Small (24.3%) and Large (28.0%)
- **Two-tier circuit:** universal core (L11H3, L7H3) + type-specific components (L6H5, L8H10)
- Negation form matters: "never/cannot" (15.9%) harder than standard "not" (24.3%)

---

## Repository Structure
