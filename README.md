# Preclinical Neural Divergence Associated with Genetic Risk for Alzheimer's Disease

*An Exploratory Resting-State EEG and Neuroimaging Study in Cognitively Healthy Adults*

[![Paper Preprint](https://img.shields.io/badge/Zenodo-Paper%20Preprint-636B2F?logo=zenodo)](https://zenodo.org/records/19244475)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![MNE](https://img.shields.io/badge/MNE--Python-EEG-00897B)

Independent undergraduate research by **Shlok Khare**, University of California, Davis.

---

## Overview

Some adults carry genes that raise their lifetime risk of Alzheimer's disease years before any memory problem appears. This project asks whether resting brain activity differs between cognitively healthy carriers of two genetic risk variants thought to act through distinct biological pathways, and whether a second imaging modality adds useful information:

- **APOE ε4**, which impairs amyloid clearance via lipid metabolism
- **PICALM rs3851179**, which disrupts clathrin-mediated endocytosis and the cellular handling of amyloid precursor protein

The study is **exploratory**. It uses resting-state EEG and fMRI from the same session in a presymptomatic cohort (mean age ≈ 55 years) and compares the two risk groups directly. There are no healthy controls, so the results describe differences *between* risk groups, not slowing or change relative to healthy aging.

Three research questions guide the work:

1. **RQ1.** Can resting-state EEG separate APOE ε4 carriers from PICALM carriers, and how consistently do individual recordings point to the correct group?
2. **RQ2.** Which EEG features and scalp regions carry the separation, and do different analyses point to the same regions?
3. **RQ3.** Does fMRI Default Mode Network (DMN) connectivity relate to the EEG measures, and does it improve classification when combined with EEG?

A short companion analysis uses the OASIS-2 longitudinal cohort to ask a related clinical question in people with mild cognitive impairment (MCI).

---

## Abstract

*For the full methodology, results, and discussion, see the [paper preprint](https://zenodo.org/records/19244475).*

Using the public PEARL-Neuro dataset (OpenNeuro ds004796; 48 APOE ε4 carriers and 30 PICALM carriers), a compact neural network (EEGNet) evaluated with leave-one-subject-out (LOSO) cross-validation classified 76 of 77 participants correctly at the subject level (accuracy 98.7%, AUC 0.989). Because checkpoints were selected using the held-out participant's own epochs, **this figure is optimistic and should be read as an upper bound**. The consistency of epoch-level predictions within a person varied: every epoch was correct for 57.8% of APOE e3/e4 carriers and 80.0% of PICALM carriers, a gap that did not reach conventional significance (Fisher exact test, p = .051), and age did not explain it. Descriptive topographic maps, in-sample SHAP importances, and spatial filters from one fold pointed tentatively to temporal and frontal channels, but a variance filter removed raw band-power features because of their measurement units, so the study cannot say whether signal strength also differs. DMN connectivity from fMRI did not improve classification. In the companion OASIS-2 analysis (N = 137), the Mini-Mental State Examination (MMSE) classified MCI better than structural MRI features (AUC 0.814 vs 0.678). The findings support further work on pathway-specific markers and argue for repeated recordings over single-session screening.

---

## Key Results

| Question | Finding |
|---|---|
| **RQ1:** EEG-only classification | EEGNet (LOSO, N = 77): subject-level AUC 0.989, accuracy 98.7% (76/77); epoch-level AUC 0.978. An optimistic upper bound (see Limitations). |
| **RQ1:** Linear baseline | Logistic regression on 521 summary features scored below chance (AUC 0.2465; 38.5% accuracy) and a 20-feature spatially targeted LR scored AUC 0.403. Treated as a failed baseline, not interpreted. |
| **RQ1:** Epoch consistency | All epochs correct in 50 of 77 folds. Fully consistent participants: 57.8% of APOE e3/e4 (26/45) vs 80.0% of PICALM (24/30), p = .051. Age and epoch count did not explain the variation. Not evidence for two biological subgroups. |
| **RQ2:** Where the signal is | Topographic maps, SHAP values, and EEGNet spatial filters all point tentatively to bilateral temporal and frontal channels. These analyses use the same data, so agreement is encouraging but not independent confirmation. |
| **RQ3:** Does fMRI help? | No. Late fusion (AUC 0.968) and intermediate fusion (AUC 0.897) both classified fewer participants correctly than EEGNet alone (N = 68). DMN connectivity alone performed at chance (AUC 0.463) and showed no significant linear relationship with EEG measures. |
| **OASIS-2 companion** | MMSE alone: AUC 0.814. Imaging features alone: 0.678. MMSE + imaging: 0.835 to 0.848. ResNet18 on axial slices: 0.676. |

A note on **DMN connectivity and SHAP**: DMN mean connectivity had the single largest SHAP value, but that ranking comes from an LR specification that scored below chance under LOSO, and DMN alone classified at chance. It is not treated as evidence that DMN connectivity separates the groups.

---

## Datasets

**PEARL-Neuro** — [OpenNeuro ds004796](https://openneuro.org/datasets/ds004796) (Dzianok & Kublik, 2024). Resting-state EEG (127 channels) and fMRI from the Nencki Institute. Openly available via DataLad. 78 of 80 participants were available after quality control; EEGNet ran 77 folds (one participant had no epoch file) and the fusion analyses covered 68 participants. Labels derive from APOE haplotype and PICALM rs3851179 genotype columns; no pre-made group column exists in the dataset. The classification is binary: APOE risk (label 0) vs PICALM risk (label 1, non-"G/G" rs3851179 and no APOE ε4 allele).

**OASIS-2** — [OASIS Brains](https://www.oasis-brains.org/). Longitudinal structural MRI with CDR-based cognitive labels. 150 baseline sessions were labelled Control (CDR 0.0, n = 85), MCI (CDR 0.5, n = 52), or Alzheimer's (CDR ≥ 1.0, n = 13). The 13 Alzheimer's participants were excluded from the binary MCI vs Control comparison (N = 137). Openly available from Washington University.

> Note: PEARL-Neuro structural MRI requires a Data Use Agreement countersigned by an institutional legal representative (Nencki Institute). This pathway is unavailable to independent researchers without faculty affiliation (confirmed in correspondence with the Nencki Institute, April 2026), so linking EEG to brain structure remains future work.

---

## Limitations

Please read these before relying on any number above.

- **Optimistic evaluation.** EEGNet early stopping, learning-rate scheduling, and checkpoint selection used the held-out participant's own epochs, so no fold had an untouched test set. Reported EEGNet and fusion performance are upper bounds.
- **No shuffled-label control yet.** The headline AUC has not been checked against a shuffled-label control. In the 68-participant fusion subset, 35 participants (51%) had an aggregated probability within 0.1 of the 0.5 boundary and 14 were within 0.02, so subject-level rankings rest on very small margins.
- **Small samples.** N = 77 (EEGNet), N = 68 (fusion), 30 PICALM carriers, 52 MCI participants. Cross-modal analyses were declared exploratory in advance and are only powered to detect large correlations (about r = 0.34 or larger).
- **Variance filter removed band power.** Absolute band-power and Hjorth activity features were dropped by a 10⁻⁴ variance filter because they are stored as raw power in V² (about 10⁻¹¹ to 10⁻¹²). The absence of band-power features from the linear analysis is a unit artifact and does not mean there are no group differences in signal strength.
- **In-sample interpretability.** SHAP used an LR fit on all 78 participants, and spatial filters came from a single fold (Fold 2) and were not compared across folds. Topographic maps were not statistically tested; channel-averaged theta and alpha power did not differ between groups.
- **Mild leakage in the LR baseline.** The variance filter and median imputation were fitted on all participants before the LOSO loop (neither uses labels).
- **Design.** Cross-sectional, no healthy controls, eyes-open and eyes-closed conditions not analysed separately, and scalp EEG cannot localise sources.
- **OASIS-2 comparison is partly circular.** CDR scores draw on clinical assessment of cognition, so the MMSE result shows how little imaging adds beyond a cognitive measure rather than proving MMSE is a better diagnostic. Grad-CAM maps were coarse and support no claim about specific regions or about the EEG findings.

---

## Repository Structure

```
├── data/
│   ├── metadata/            # Manifests, feature matrices, processing summaries
│   ├── processed/           # EEG epochs, features, fMRI features, MRI tensors
│   └── splits/              # LOSO and k-fold split definitions
├── extract/
│   ├── xtpneuro.py          # PEARL-Neuro preprocessing pipeline (EEG + fMRI)
│   └── xtoasis2.py          # OASIS-2 preprocessing pipeline (MRI)
├── notebook/
│   ├── preprocess.ipynb     # Feature extraction and label derivation
│   ├── model.ipynb          # EEGNet, ResNet18, LR baselines, EDA
│   ├── analysis.ipynb       # SHAP, spatial filters, significance tests, Grad-CAM
│   └── fusion.ipynb         # Late + intermediate fusion, fold-consistency subgroup analysis
├── results/
│   ├── checkpoints/         # EEGNet (77 folds) + ResNet18 v2 (5 folds) checkpoints
│   ├── figures/             # All publication figures
│   └── *.csv                # Fold-level results, SHAP scores, fusion metrics
├── report.md                # Full experimental log and results documentation
├── roadmap.md               # Research questions, paper structure, task plan
└── README.md
```

---

## Dependencies

Core: `mne`, `mne-bids`, `torch`, `scikit-learn`, `nilearn`, `nibabel`, `shap`, `numpy`, `pandas`, `scipy`

FSL (`bet`, `flirt`) is required for OASIS-2 skull stripping and MRI registration. DataLad is required to fetch PEARL-Neuro raw files.

---

## Future Work

1. Hold out a validation split drawn from training participants only, add a shuffled-label control, and nest feature selection inside each training fold, to confirm the main result.
2. Rescale features before variance filtering so band power can be assessed.
3. Follow carriers longitudinally, especially those with misclassified epochs, to test whether the EEG signature emerges with age or varies with noise or state.
4. Link EEG measures to the seven PEARL-Neuro psychometric tests (not analysed here) to test for cognitive correlates.
5. Replicate in a larger cohort to test subgroup claims at adequate power.

---

## Citation

If you use this code or build on this work, please cite:

```
Khare, S. (2026). Preclinical Neural Divergence Associated With Genetic
Risk for Alzheimer's Disease: An Exploratory Resting-State EEG and
Neuroimaging Study in Cognitively Healthy Adults. University of California,
Davis. https://github.com/shlokye16/psychM2D2
```

---

## License

Dataset usage is subject to the terms of OpenNeuro (PEARL-Neuro) and OASIS (OASIS-2) respectively.
