# GMMJaccardSimilarity
# GMM Similarity Metric

A normalized, symmetric, interpretable similarity metric for comparing two
**Gaussian Mixture Models (GMMs)** based on pairwise component-level
**Jaccard overlap** of probability density functions.

Two integration variants are provided, sharing the same matching algorithm:

| Variant | Backend | Accuracy | Speed |
|---------|---------|----------|-------|
| **Numerical** (`gmm_similarity`) | `np.trapezoid` on 10k-point grid | O(h^2) ~ 1e-5 | 420-530 us/call |
| **Analytical** (`gmm_similarity_erf`) | `math.erf` exact CDF | Machine epsilon ~ 1e-16 | 23-50 us/call |

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.24%2B%20%7C%202.0%2B-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Comparing two GMMs is a common requirement in signal processing, neuroscience,
and machine learning -- for instance, assessing whether an EEG spectral profile
has shifted between conditions, or detecting covariate shift in ML pipelines.

Both variants use a **component-wise Jaccard approach** that is more
interpretable than KL divergence or Wasserstein distance: each matched
component pair contributes an additive, bounded Jaccard score, and the
final metric is normalized to `[0, 1]`.

| Score | Meaning |
|-------|---------|
| `1.0` | Identical distributions |
| `0.5` | Moderate overlap |
| `0.0` | Fully disjoint distributions |

---

## Matching Algorithm: AUC-Priority Greedy

Both variants use the same **AUC-priority greedy matching** strategy.

The term *greedy* is used in the strict algorithmic sense: at each step the
locally optimal choice is made (the B-component maximizing Jaccard with the
current A-component) without backtracking. *AUC-priority* means components
are processed in descending AUC order, so statistically dominant modes drive
the score.

```
1. Normalize each GMM so that total AUC = 1.0  (proper probability density)
2. Sort GMM_A and GMM_B components by AUC descending
3. Take A[0] (highest AUC); search ALL remaining B components for best Jaccard
4. Record the match, remove B[j*] from candidates
5. Repeat for A[1], A[2], ... until either GMM is exhausted
6. Normalize: score = sum(Jaccard_i) / min(K_A, K_B)  -> [0, 1]
```

This greedy solution is identical to the globally optimal (Hungarian)
assignment in the common case of well-separated GMM components. For K > 20
with many near-equal Jaccard scores, the Hungarian algorithm (O(K^3)) can be
substituted for the greedy inner loop (O(K^2)) without other changes.

---

## Variant 1 -- Numerical (Trapezoidal)

**Functions:** `jaccard()`, `extract_components()`, `gmm_similarity()`

Evaluates the Jaccard index by building a uniform 10,000-point grid,
computing PDFs via `scipy.stats.norm.pdf`, and integrating with
`np.trapezoid` (NumPy 2.x) or `np.trapz` (NumPy 1.x -- handled automatically).

```python
score, raw_score, matches = gmm_similarity(gmm_a, gmm_b)
```

**When to use:** Exploratory work, educational contexts, or cases where
integration transparency is preferred over speed.

---

## Variant 2 -- Analytical (erf-based)

**Functions:** `jaccard_erf()`, `extract_components_erf()`, `gmm_similarity_erf()`

Computes the Jaccard index exactly in three steps:

**Step A -- Find intersection points analytically.**
Setting f1(x) = f2(x) and taking logarithms yields:

```
Equal sigma:    x = (mu1+mu2)/2 + (sigma^2/(mu2-mu1)) * ln(w1/w2)

Unequal sigma:  A*x^2 + B*x + C = 0
                A = 1/(2*s2^2) - 1/(2*s1^2)
                B = mu1/s1^2 - mu2/s2^2
                C = mu2^2/(2*s2^2) - mu1^2/(2*s1^2) - ln(w2*s1/w1*s2)
```

At most 2 real intersections exist; the discriminant determines which case
applies (0, 1, or 2 roots).

**Step B -- Partition into segments.**
The intersection points divide the real line into at most 3 segments. On each
segment one Gaussian strictly dominates everywhere, identified by evaluating
both PDFs at the segment midpoint using scalar `math.exp` -- no arrays.

**Step C -- Integrate each segment via erf.**
The exact AUC of a weighted Gaussian over [a, b] is:

```
AUC(mu, sigma, w, a, b) = w * (CDF(b) - CDF(a))
CDF(x) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
```

At most **6 erf calls** per Jaccard evaluation (3 segments x 2 CDF values).
No array allocation. No numerical grid.

```python
score, raw_score, matches = gmm_similarity_erf(gmm_a, gmm_b)
```

**When to use:** Production pipelines, large-scale comparisons, or any context
where Jaccard is called inside an optimization loop.

---

## Performance Comparison

Measured on Python 3.11.7, NumPy 2.x, 2000 repetitions per case:

| Test case | t_erf (us) | t_num (us) | Speedup |
|-----------|------------|------------|---------|
| Different sigma | 49.4 | 503.5 | **10.2x** |
| Same mean, diff weight | 27.9 | 525.7 | **18.8x** |
| Well separated | 34.2 | 470.1 | **13.8x** |
| Near-identical | 36.8 | 478.8 | **13.0x** |
| Identical | 23.4 | 450.5 | **19.2x** |

**Accuracy** (analytical vs numerical, same cases):

| Test case | diff |
|-----------|------|
| Different sigma | 9.4e-08 |
| Same mean, diff weight | 5.6e-17 |
| Well separated | 1.3e-10 |
| Near-identical | 1.8e-07 |
| Identical | 0.0e+00 |

---

## Quickstart

```python
import numpy as np
from sklearn.mixture import GaussianMixture

np.random.seed(42)
X_a = np.concatenate([np.random.normal(-2, 0.5, 300),
                       np.random.normal( 2, 1.0, 500)]).reshape(-1, 1)
X_b = np.concatenate([np.random.normal(-1.5, 0.6, 300),
                       np.random.normal( 2.5, 0.9, 500)]).reshape(-1, 1)

gmm_a = GaussianMixture(n_components=2).fit(X_a)
gmm_b = GaussianMixture(n_components=2).fit(X_b)

# Numerical variant
score_num, _, matches_num = gmm_similarity(gmm_a, gmm_b)

# Analytical variant (recommended for production)
score_erf, _, matches_erf = gmm_similarity_erf(gmm_a, gmm_b)

print(f"Numerical : {score_num:.6f}")
print(f"Analytical: {score_erf:.6f}")

plot_gmm_matching(gmm_a, gmm_b, matches_erf)
```

---

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib
```

No additional dependencies. Compatible with NumPy 1.x and 2.x.

---

## File Structure

```
GaussianMixtureComparison.ipynb      # Original notebook (numerical variant)
GaussianMixtureComparisonERF.ipynb   # Extended notebook (+ analytical variant)
GMM_Similarity_Report_v2.docx        # Scientific report (both variants compared)
README.md                            # This file
```

### Notebook Cells (ERF notebook)

| Cell | Contents |
|------|----------|
| 0 | GMM fitting, AUC computation (numerical + analytical), visualization |
| 1 | Pairwise Gaussian similarity: Jaccard + Overlap coefficient |
| 2 | Full GMM comparison -- **Numerical variant** |
| 3 | Full GMM comparison -- **Analytical (erf) variant** + correctness check + benchmark |
| 4 | Validation test suite (12 tests, 6 groups) |

---

## Validation Results

All **12/12 tests pass** across 6 validation groups (numerical variant backend).
The analytical variant agrees to at least 5 significant figures on all cases
where it was explicitly cross-checked.

### Group 0 -- Normalization
All GMMs normalize to total AUC = `1.00000000`.

### Group 1 -- Identity & Near-Identity

| Test | Score | Result |
|------|-------|--------|
| Identical single Gaussian | 1.0000 | PASS |
| Identical 3-component GMM | 1.0000 | PASS |
| Near-identical (shift = 0.1 sigma) | 0.9101 | PASS |
| Near-identical 3-component (shift = 0.1) | 0.8806 | PASS |

### Group 2 -- Zero Overlap

| Test | Score | Result |
|------|-------|--------|
| Gaussians separated by 20 sigma | 0.0000 | PASS |
| 3-component GMMs fully separated | 0.0000 | PASS |

### Group 3 -- Partial Overlap

| Test | Score | Result |
|------|-------|--------|
| Shift = 1 sigma | 0.4379 | PASS |
| Shift = 2 sigma | 0.1999 | PASS |
| Same mean, std ratio = 2x | 0.5094 | PASS |
| Same mean, std ratio = 5x | 0.2129 | PASS |

### Group 4 -- Asymmetric Component Counts

| Test | Score | Result |
|------|-------|--------|
| 3 vs 2 components (2 matching well) | 0.7959 | PASS |
| 1 vs 3 components (1 match expected) | 0.5001 | PASS |

### Group 5 -- Symmetry
`score(A,B) = score(B,A) = 0.4621` -- difference = `0.0000`

### Group 6 -- Monotonicity

| Shift (sigma) | Score |
|---------------|-------|
| 0.0 | 0.9439 |
| 0.5 | 0.7116 |
| 1.0 | 0.4806 |
| 2.0 | 0.1914 |
| 4.0 | 0.0258 |

---

## Development History: Prompt Chain

| Step | Prompt summary | Output |
|------|---------------|--------|
| 1 | Draw Gaussians from GMM result | GMM fit + per-component PDF rendering |
| 2 | Compute AUC under each Gaussian | Numerical + analytical AUC |
| 3-4 | Fix Unicode SyntaxError (approx sign) x2 | ASCII-safe code |
| 5 | Fix np.trapz AttributeError | NumPy 2.0-safe fallback |
| 6 | Pairwise Gaussian similarity: Jaccard + Overlap | gaussian_similarity() |
| 7 | Fix Unicode SyntaxError (arrow) | Triple-quoted comment |
| 8 | Normalize by max possible score | / min(K_A,K_B) |
| 9 | Full GMM-to-GMM AUC-priority greedy matching | gmm_similarity() + plot |
| 10 | Fix Unicode SyntaxError (arrow) again | Triple-quoted comment |
| 11 | Normalize weights so total AUC = 1.0 | extract_components() renormalization |
| 12 | Comprehensive validation test suite | 12 tests, 6 groups, 100% PASS |
| 13 | Validate notebook + scientific report | GMM_Similarity_Report.docx v1 |
| 14 | README with prompt chain | README v1 |
| 15 | LinkedIn post | Posted at github.com/fiammante |
| 16 | Analytical erf variant (10-20x faster, machine precision) | jaccard_erf(), gmm_similarity_erf() |
| 17 | Inject erf variant into notebook as Cell 3 | GaussianMixtureComparisonERF.ipynb |
| 18 | Update report + README, clarify greedy term | Report v2 + this README |

---

## Key Design Decisions

**Why Jaccard and not KL divergence?**
KL divergence is asymmetric and undefined when one distribution has zero
mass where the other does not. Jaccard on densities is symmetric, always
defined, and bounded in [0, 1].

**Why AUC-priority greedy and not Hungarian algorithm?**
For typical GMM sizes (K <= 10), greedy and optimal assignment produce
identical results because Jaccard between well-separated components is
numerically zero, leaving no assignment ambiguity. Greedy runs in O(K_A x K_B)
vs O(K^3) for Hungarian, with no loss of solution quality in practice.

**Why the analytical erf variant?**
Trapezoidal integration requires O(N) array allocation and O(h^2) error.
The analytical variant requires at most 6 scalar erf evaluations and achieves
machine epsilon accuracy. The intersection geometry is always tractable
(linear or quadratic equation) for 1-D Gaussians.

**Why normalize weights to AUC = 1?**
sklearn GMM weights already sum to 1 over the full real line, but over a
finite integration grid there is a small truncation deficit. Explicit
renormalization ensures all comparisons are on a true probability simplex.

**Why min(K_A, K_B) as the normalization denominator?**
It reflects the number of component pairs that can actually be compared.
Unmatched components in the larger GMM contribute zero, naturally penalizing
structural mismatch without an explicit penalty term.

---

## Limitations

- **1-D only.** Multivariate extension requires d-dimensional numerical
  integration (expensive for d > 3) or approximations (Bhattacharyya, Monte Carlo).
- **Greedy is not globally optimal.** For K > 20 with many near-equal Jaccard
  scores, substitute the Hungarian algorithm for the inner search loop.
- **min(K_A, K_B) denominator** does not explicitly penalize |K_A - K_B|.
  Use max(K_A, K_B) for stricter structural penalty.
- **Analytical variant near-equal sigma.** When sigma1 is very close but not
  equal to sigma2, the quadratic coefficient A is near zero. The np.isclose
  guard handles the exact case; near-equal sigmas fall through to the quadratic
  branch correctly but with reduced numerical margin.

---

## Citation

```
GMM Similarity Metric -- AUC-Priority Greedy Jaccard Comparison of Gaussian Mixture Models
Technical Report v2, Paris Brain Institute / AP-HP, February 2026
https://github.com/fiammante/GMMJaccardSimilarity
```

---

## License

MIT License -- see LICENSE for details.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
