# `steerlens` — Mechanistic Interpretability, SAE Probes & Activation Steering Parameters

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Rust Kernel](https://img.shields.io/badge/Rust-steerlens__core-orange.svg)](rust/src/lib.rs)

**`steerlens`** is Pillar 5 of the **Open, Decentralized Verifiable AI Stack**: a glass-box mechanistic interpretability and activation steering library that extracts **monosemantic Sparse Autoencoder (SAE) features** and **Contrastive Activation Addition (CAA) steering vectors** from verifier-guided tree search, then compiles them into **portable parameter deltas** (linear biases or rank-1 LoRA adapters).

---

## Why `steerlens`?

When a verifier-guided search engine (`formal-lean`, `verichip`, `sysguard`, `chemcircuit`) explores reasoning branches, it naturally produces **paired contrastive trajectories**:
- **Verified Winning Branches ($h^+$):** States that pass the Lean 4 kernel, Yosys equivalence checker, eBPF verifier, or Conservation-Law solver.
- **Refuted / Hallucinated Branches ($h^-$):** Dead-end states rejected by the formal oracle.

Instead of throwing away refuted branches or requiring expensive full-model backpropagation, **`steerlens`** mines the hidden layer activations directly:

1. **Sparse Autoencoder Probes (`SparseAutoencoder`):** Decomposes dense polysemantic activations $h \in \mathbb{R}^d$ into sparse monosemantic features $f \in \mathbb{R}^m$ ($m \gg d$) with exact Top-$K$ sparsity, identifying which interpretable features fire during deductive planning vs. ungrounded guessing.
2. **Contrastive Activation Addition (`compute_caa_vector`):** Computes the normalized causal steering direction:
   $$v_{\text{steer}} = \frac{\mathbb{E}[h^+] - \mathbb{E}[h^-]}{\|\mathbb{E}[h^+] - \mathbb{E}[h^-]\|_2}$$
   Injecting $h' = h + \alpha \cdot v_{\text{steer}}$ at inference time shifts model logits directly toward formally verifiable actions.
3. **Zero-Overhead Parameter Folding (`fold_steering_into_linear_bias`, `fold_steering_into_lora_delta`):** Folds runtime steering vectors directly into static model biases ($b' = b + \alpha W v_{\text{steer}}$) or rank-1 LoRA adapters ($\Delta W = \alpha v_{\text{steer}} k_{\text{trigger}}^\top$) so contributors can submit lightweight **Steering Parameter PRs** to `everbench`!

---

## Architecture

```
Verified Search Branches (h+) ──┐
                                ├──► Top-K Sparse Autoencoder (SAE) ──► Diagnostic Feature Telemetry
Refuted Search Branches  (h-) ──┤
                                └──► Contrastive Activation Addition (CAA)
                                              │
                                              ├──► Runtime Hook: h' = h + α · v_steer
                                              └──► Folded Parameter PR: Δb = α W v_steer / Rank-1 LoRA
```

---

## Quickstart

```python
from steerlens import (
    SparseAutoencoder,
    compute_caa_vector,
    fold_steering_into_linear_bias,
    ReasoningTelemetry,
)

# 1. Fit a Top-K Sparse Autoencoder on hidden states collected during tree search
sae = SparseAutoencoder(d_model=8, d_sae=24, top_k=3)
sae.fit(pos_acts + neg_acts, epochs=25, lr=0.06)

# 2. Identify monosemantic features that distinguish verified proofs from hallucinations
diagnostics = sae.top_diagnostic_features(pos_acts, neg_acts, top_n=3)
print("Top verified reasoning feature:", diagnostics[0])

# 3. Extract a Contrastive Activation Addition (CAA) steering vector
steer_vec = compute_caa_vector(pos_acts, neg_acts, layer_name="model.layers.16.mlp")

# 4. Fold the steering vector into a downstream linear bias (zero runtime overhead!)
new_bias = fold_steering_into_linear_bias(W_next, b_next, steer_vec, alpha=2.0)
```

---

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```
