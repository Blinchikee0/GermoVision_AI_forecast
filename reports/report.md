# GermoVision training report

Generated 2026-09-05 16:07:50 · source `synthetic` · run time 52.9s

> **Note.** SYNTHETIC DATA. These metrics show that the pipeline is correct, not that the model is clinically good. Hypothesis H1 requires the CRyPTIC dataset.

## Data

- Isolates: **6000**, relatedness clusters: 1693
- Countries: 10, lineages: 5, variants: 68
- Period: 2015-01-23 — 2025-04-22
- Split: temporal_cluster, {'train': 3899, 'test': 1203, 'calib': 898}

## Internal test

Closed — share of isolates given a correct answer in 1-2 days instead of
~60. Missed — share of resistant isolates confidently called susceptible:
the only genuinely dangerous outcome. Both are computed over all isolates,
not only the answered ones.

The Lab column marks drugs where the missed-resistance limit is not met:
genomic prediction alone is insufficient and laboratory confirmation is
required.

| Drug | N | Resistant | Closed | Missed | Lab | Sensitivity | Specificity | PR-AUC | Catalogue | Answered |
|---|---|---|---|---|---|---|---|---|---|---|
| Rifampicin | 1161 | 319 | **80.8%** | 21.8% | required | 0.774 [0.729–0.815] | 0.842 [0.816–0.866] | 0.750 [0.710–0.791] | 0.679 [0.624–0.730] | 98.1% |
| Rifabutin | 988 | 217 | **81.0%** | 20.7% | required | 0.756 [0.696–0.806] | 0.903 [0.883–0.924] | 0.633 [0.577–0.698] | 0.613 [0.549–0.668] | 93.0% |
| Isoniazid | 1133 | 349 | **82.5%** | 16.4% | required | 0.822 [0.781–0.862] | 0.876 [0.853–0.899] | 0.780 [0.742–0.819] | 0.663 [0.618–0.711] | 95.9% |
| Ethambutol | 1096 | 197 | **91.5%** | 21.8% | required | 0.756 [0.695–0.812] | 0.990 [0.983–0.997] | 0.785 [0.733–0.839] | 0.673 [0.614–0.736] | 96.6% |
| Levofloxacin | 1108 | 157 | **85.0%** | 44.6% | required | 0.548 [0.471–0.618] | 0.906 [0.887–0.922] | 0.435 [0.371–0.507] | 0.541 [0.459–0.623] | 99.4% |
| Moxifloxacin | 1040 | 132 | **84.7%** | 23.2% | required | 0.735 [0.667–0.811] | 0.913 [0.892–0.932] | 0.446 [0.388–0.526] | 0.642 [0.566–0.715] | 95.2% |
| Bedaquiline | 860 | 56 | **93.3%** | 55.4% | required | 0.446 [0.312–0.581] | 0.966 [0.952–0.978] | 0.398 [0.279–0.519] | 0.446 [0.312–0.571] | 100.0% |
| Linezolid | 920 | 55 | **95.8%** | 67.3% | required | 0.327 [0.218–0.455] | 0.998 [0.994–1.000] | 0.366 [0.256–0.486] | 0.327 [0.200–0.436] | 100.0% |
| Clofazimine | 857 | 46 | **93.7%** | 51.8% | required | 0.391 [0.249–0.522] | 0.990 [0.984–0.996] | 0.305 [0.193–0.454] | 0.333 [0.222–0.463] | 97.8% |
| Delamanid | 670 | 39 | **89.9%** | 55.0% | required | 0.436 [0.282–0.590] | 1.000 [1.000–1.000] | 0.084 [0.052–0.140] | 0.425 [0.275–0.575] | 92.9% |
| Amikacin | 1005 | 83 | **89.9%** | 36.5% | required | 0.578 [0.470–0.675] | 0.966 [0.956–0.977] | 0.395 [0.301–0.506] | 0.500 [0.396–0.604] | 96.3% |
| Kanamycin | 955 | 111 | **88.0%** | 29.0% | required | 0.676 [0.590–0.757] | 0.932 [0.915–0.948] | 0.563 [0.469–0.652] | 0.589 [0.504–0.674] | 97.5% |
| Ethionamide | 1009 | 171 | **88.8%** | 16.8% | required | 0.825 [0.766–0.877] | 0.914 [0.895–0.930] | 0.723 [0.651–0.789] | 0.687 [0.615–0.754] | 98.8% |

## External validation (trained without KZ)

The only estimate that reflects a real deployment.

| Drug | N | Sensitivity | Specificity | H1 target | Verdict |
|---|---|---|---|---|---|
| Rifampicin | 466 | 0.871 [0.825–0.912] | 0.667 [0.598–0.717] | ≥0.90 / ≥0.95 | not met |
| Rifabutin | 406 | 0.868 [0.815–0.921] | 0.824 [0.780–0.871] | — | — |
| Isoniazid | 487 | 0.874 [0.833–0.913] | 0.734 [0.676–0.793] | ≥0.90 / ≥0.95 | not met |
| Ethambutol | 441 | 0.837 [0.772–0.894] | 0.987 [0.975–0.997] | — | — |
| Levofloxacin | 440 | 0.694 [0.604–0.770] | 0.790 [0.743–0.836] | — | — |
| Moxifloxacin | 419 | 0.835 [0.763–0.907] | 0.783 [0.741–0.825] | — | — |
| Bedaquiline | 342 | 0.391 [0.174–0.565] | 0.972 [0.954–0.987] | — | — |
| Linezolid | 384 | 0.481 [0.259–0.667] | 0.992 [0.980–1.000] | — | — |
| Clofazimine | 350 | 0.500 [0.273–0.727] | 0.994 [0.985–1.000] | — | — |
| Delamanid | 316 | 0.263 [0.078–0.474] | 1.000 [1.000–1.000] | — | — |
| Amikacin | 429 | 0.677 [0.569–0.800] | 0.934 [0.909–0.959] | — | — |
| Kanamycin | 418 | 0.762 [0.662–0.850] | 0.867 [0.828–0.902] | — | — |
| Ethionamide | 432 | 0.871 [0.809–0.921] | 0.857 [0.816–0.894] | — | — |

## Ablations (rifampicin)

Sensitivity and specificity are computed on the decisions actually issued;
PR-AUC on the calibrated probability.

| Configuration | Sens | Spec | PR-AUC | Abstained |
|---|---|---|---|---|
| WHO catalogue only (baseline) | 0.679 | 0.851 | 0.523 | 0.0% |
| mutations, no catalogue features | 0.880 | 0.687 | 0.768 | 10.8% |
| + catalogue features | 0.865 | 0.733 | 0.781 | 17.6% |
| + per-gene burden | 0.865 | 0.733 | 0.777 | 17.6% |
| + variants outside target genes | 0.634 | 0.953 | 0.767 | 9.7% |
| full model (+ rule tier) | 0.774 | 0.842 | 0.750 | 1.9% |
| ... plus lineage context (hurts) | 0.777 | 0.842 | 0.758 | 2.0% |

## Answer rate versus accuracy (rifampicin)

Abstaining sends the sample for phenotypic testing: the system trades
speed for certainty. Which point on this curve to pick is the
organisation's decision, not the developer's.

| alpha | Answered | Accuracy | Sens | Spec |
|---|---|---|---|---|
| 0.02 | 33.6% | 0.666 | 0.974 | 0.244 |
| 0.05 | 91.4% | 0.835 | 0.815 | 0.842 |
| 0.10 | 98.1% | 0.823 | 0.774 | 0.842 |
| 0.15 | 100.0% | 0.817 | 0.748 | 0.844 |
| 0.20 | 98.1% | 0.823 | 0.774 | 0.842 |
| 0.30 | 91.0% | 0.836 | 0.821 | 0.842 |

## GV-Growth: recovering growth coefficients

Estimated shrinkage tau = 0.0323

| Lineage | True beta | Estimated beta | Spread across regions |
|---|---|---|---|
| L4_Euro_American | +0.0000 | +0.0000 | 0.0000 |
| L2_Beijing | +0.0200 | +0.0073 | 0.0143 |
| L2_Beijing_MDR | +0.1100 | +0.0888 | 0.0271 |
| L3_CAS | -0.0300 | -0.0569 | 0.0158 |
