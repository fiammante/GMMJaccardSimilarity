# GMMJaccardSimilarity
# GMM Similarity Metric

A normalized, symmetric, interpretable similarity metric for comparing two
**Gaussian Mixture Models (GMMs)** based on pairwise component-level
**Jaccard overlap** of probability density functions.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.24%2B%20%7C%202.0%2B-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Comparing two GMMs is a common requirement in signal processing, neuroscience,
and machine learning — for instance, assessing whether an EEG spectral profile
has shifted between conditions, or detecting covariate shift in ML pipelines.

This project provides a **component-wise approach** that is more interpretable
than KL divergence or Wasserstein distance: each matched component pair
contributes an additive, bounded Jaccard score, and the final metric is
normalized to `[0, 1]`.

| Score | Meaning |
|-------|---------|
| `1.0` | Identical distributions |
| `0.5` | Moderate overlap |
| `0.0` | Fully disjoint distributions |

---

## Algorithm

```
1. Normalize each GMM so that total AUC = 1.0  (proper probability density)
2. Sort components of GMM_A and GMM_B by AUC descending
3. Greedily match A[0] to its best Jaccard partner in B (exhaustive search)
4. Remove matched B component; repeat for A[1], A[2], ...
5. Stop when either GMM has no remaining components
6. Normalize: score = sum(Jaccard_i) / min(K_A, K_B)
```

The **Jaccard index on densities** is defined as:

```
J(f1, f2) = integral(min(f1, f2) dx) / integral(max(f1, f2) dx)
```

This is the continuous analogue of set-theoretic Jaccard: `|A ∩ B| / |A ∪ B|`.

---

## Quickstart

```python
import numpy as np
from sklearn.mixture import GaussianMixture

# Fit two GMMs
X_a = np.concatenate([np.random.normal(-2, 0.5, 300),
                       np.random.normal( 2, 1.0, 500)]).reshape(-1, 1)
X_b = np.concatenate([np.random.normal(-1.5, 0.6, 300),
                       np.random.normal( 2.5, 0.9, 500)]).reshape(-1, 1)

gmm_a = GaussianMixture(n_components=2).fit(X_a)
gmm_b = GaussianMixture(n_components=2).fit(X_b)

# Compare
score, raw_score, matches = gmm_similarity(gmm_a, gmm_b)
print(f"Normalized similarity: {score:.4f}")

# Visualize
plot_gmm_matching(gmm_a, gmm_b, matches)
```

---

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib
```

No additional dependencies. Compatible with NumPy 1.x (`np.trapz`) and
NumPy 2.x (`np.trapezoid`) — handled automatically at runtime.

---

## File Structure

```
GaussianMixtureComparison.ipynb   # Main notebook (4 cells)
README.md                         # This file
GMM_Similarity_Report.docx        # Scientific report
```

### Notebook Cells

| Cell | Contents |
|------|----------|
| 0 | GMM fitting, AUC computation (numerical + analytical), visualization |
| 1 | Pairwise Gaussian similarity: Jaccard + Overlap coefficient |
| 2 | Full GMM-to-GMM comparison with greedy matching and normalization |
| 3 | Validation test suite (12 tests, 6 groups) |

---

## Validation Results

All **12/12 tests pass** across 6 validation groups:

### Group 0 — Normalization
All GMMs normalize to total AUC = `1.00000000` regardless of component count
or weight imbalance.

### Group 1 — Identity & Near-Identity

| Test | Score | Result |
|------|-------|--------|
| Identical single Gaussian | 1.0000 | PASS |
| Identical 3-component GMM | 1.0000 | PASS |
| Near-identical (shift = 0.1 sigma) | 0.9101 | PASS |
| Near-identical 3-component (shift = 0.1) | 0.8806 | PASS |

### Group 2 — Zero Overlap

| Test | Score | Result |
|------|-------|--------|
| Gaussians separated by 20 sigma | 0.0000 | PASS |
| 3-component GMMs fully separated | 0.0000 | PASS |

### Group 3 — Partial Overlap

| Test | Score | Result |
|------|-------|--------|
| Shift = 1 sigma | 0.4379 | PASS |
| Shift = 2 sigma | 0.1999 | PASS |
| Same mean, std ratio = 2x | 0.5094 | PASS |
| Same mean, std ratio = 5x | 0.2129 | PASS |

### Group 4 — Asymmetric Component Counts

| Test | Score | Result |
|------|-------|--------|
| 3 vs 2 components (2 matching well) | 0.7959 | PASS |
| 1 vs 3 components (1 match expected) | 0.5001 | PASS |

### Group 5 — Symmetry
`score(A, B) = score(B, A) = 0.4621` — difference = `0.0000`

### Group 6 — Monotonicity

| Shift (sigma) | Score |
|---------------|-------|
| 0.0 | 0.9439 |
| 0.5 | 0.7116 |
| 1.0 | 0.4806 |
| 2.0 | 0.1914 |
| 4.0 | 0.0258 |

Strict monotonic decrease confirmed across all shift levels.

---

## Development History: Prompt Chain

This project was built incrementally through a conversational AI-assisted
development session. The full chain of prompts is reproduced below as a
record of the iterative methodology.

---

### Step 1 — Draw Gaussians from GMM
```
In Python give code to draw gaussians from the result of GMM
```
Produced: GMM fitting with scikit-learn, per-component weighted PDF rendering,
histogram overlay, legend with mu/sigma/weight per component.

---

### Step 2 — Compute AUC Under Each Gaussian
```
Add code to compute the surface under the curve of each gaussian
```
Added two AUC methods per component:
- **Numerical**: `np.trapz(pdf, x)`
- **Analytical**: `weight * (CDF(x_max) - CDF(x_min))`

---

### Step 3 — Fix Unicode SyntaxError (x1)
```
SyntaxError: invalid character '≈' (U+2248)
```
The `≈` character in an explanatory text block was accidentally included in
the notebook cell. Fixed by replacing with plain ASCII `~`.

---

### Step 4 — Fix Unicode SyntaxError (x2)
The same error recurred because the output example block (containing `≈`) was
also pasted into the cell. Resolved by isolating the runnable code only.

---

### Step 5 — Fix `np.trapz` AttributeError
```
AttributeError: module 'numpy' has no attribute 'trapz'
```
`np.trapz` was removed in NumPy 2.0. Fixed with a version-safe fallback:
```python
try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz
```

---

### Step 6 — Pairwise Gaussian Similarity (Jaccard + Overlap)
```
Create code to compare similarity of 2 gaussian by computing fraction
of intersecting AUC with AUC of the 2 gaussians
```
Produced `gaussian_similarity()` computing:
- **Jaccard**: `intersection / union`
- **Overlap coefficient**: `intersection / min(AUC1, AUC2)`

With dual-panel visualization (PDF + intersection fill, bar chart of metrics).

---

### Step 7 — Fix Unicode Arrow SyntaxError
```
SyntaxError: invalid character '→' (U+2192)
```
Algorithm flow explanation block with `→` and `∈` characters was included
in the cell. Fixed by wrapping in a triple-quoted string comment `''' ... '''`.

---

### Step 8 — Normalize by Max Possible Score
```
Normalize result using the max possible
```
Added `n_matched = min(len(comps_a), len(comps_b))` normalization, mapping
the raw Jaccard sum to `[0, 1]`. Updated verbose output and plot title.

---

### Step 9 — Full GMM-to-GMM Greedy Matching
```
Given 2 different GMM create a similarity metric by
1: sorting each GMM result by gaussian AUC
2: starting with the GMM that has the component with the highest AUC
   and finding the gaussian in the other GMM that has the highest jaccard
3: removing these 2 gaussians from the mutual comparison, and apply
   similar approach iteratively to remaining components
4: return the sum of jaccard results as metric
```
Produced the core `gmm_similarity()` greedy matching algorithm and
`plot_gmm_matching()` with one subplot per matched pair.

---

### Step 10 — Fix Unicode Arrow SyntaxError (again)
```
SyntaxError: invalid character '→' (U+2192)
Cell In[8], line 169
```
The algorithm flow comment block at the bottom of the cell again caused the
issue. Fixed by wrapping in `''' ... '''`.

---

### Step 11 — Normalization for Proper Probability Densities
```
Fix test cases by normalizing mixtures so that the total surface of all
components of a given GMM is 1.0 — GMM of probability distributions
```
Added AUC renormalization inside `extract_components()`: weights are scaled
so `sum(AUC_i) = 1.0` over the finite integration domain `[-20, 20]`,
correcting for truncation artifacts and grounding all comparisons on a
proper probability simplex.

---

### Step 12 — Comprehensive Validation Test Suite
```
Create a comprehensive test set to validate the metric for GMM comparison
```
Produced 12 programmatic tests across 6 groups (identity, zero overlap,
partial overlap, asymmetric K, symmetry, monotonicity) with expected ranges
and pass/fail reporting. Dual visualization: monotonicity curve +
horizontal pass/fail bar chart.

---

### Step 13 — Validate Notebook and Write Scientific Report
```
Validate Notebook results, and write a scientific style report on the approach
```
Full extraction and cross-check of all notebook cell outputs. Generated a
scientific Word document (`GMM_Similarity_Report.docx`) covering:
abstract, introduction, methodology (formal definitions), implementation
table, validation results tables, discussion of limitations, and conclusions.

---

### Step 14 — This README
```
Create a readme md for github, including a summary of the chain
of prompts to get to the result
```

---

## Key Design Decisions

**Why Jaccard and not KL divergence?**
KL divergence is asymmetric and undefined when one distribution has zero
mass where the other does not. Jaccard on densities is symmetric, always
defined, and bounded.

**Why greedy AUC-sorted matching and not Hungarian algorithm?**
For typical GMM sizes (K <= 10), the greedy approach and optimal assignment
produce identical results because Jaccard scores between well-separated
components are numerically zero, leaving no ambiguity. Greedy runs in
O(K_A * K_B) vs O(K^3) for Hungarian.

**Why normalize weights to AUC = 1?**
`sklearn` GMM weights already sum to 1 over the full real line, but over a
finite grid there is a small truncation deficit. Explicit renormalization
ensures all comparisons are on a true probability simplex and results are
reproducible regardless of integration domain choice.

**Why min(K_A, K_B) as the normalization denominator?**
It reflects the number of component pairs that can actually be compared.
Unmatched components in the larger GMM implicitly contribute zero, penalizing
structural mismatch without requiring an explicit penalty term.

---

## Potential Applications

- **EEG / neural signal analysis**: compare spectral density GMM profiles
  across subjects, sessions, or experimental conditions
- **Covariate shift detection**: quantify distributional drift in ML features
  between training and production data
- **Generative model evaluation**: compare sample density of a generative
  model to the target distribution
- **Population biomarker studies**: assess similarity of biomarker
  distributions across cohorts

---

## Limitations

- Currently 1-D only. Multivariate extension requires d-dimensional numerical
  integration (expensive for d > 3; consider Monte Carlo or Bhattacharyya
  coefficient approximations).
- Greedy matching is not globally optimal. For K > 20 with many
  near-equal Jaccard scores, Hungarian matching may be preferable.
- Normalization by `min(K_A, K_B)` does not penalize `|K_A - K_B|`
  directly. Use `max(K_A, K_B)` as denominator for stricter structural
  penalty.

---

## Citation

If you use this metric in your work, please cite:

```
GMM Similarity Metric — Greedy Jaccard-Based Comparison of Gaussian Mixture Models
Technical Report, Paris Brain Institute / AP-HP, February 2026
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
