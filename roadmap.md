# Research Roadmap
### Preclinical Neural Divergence Associated With Genetic Risk for Alzheimer's Disease — PEARL-Neuro + OASIS-2

*An Exploratory Resting-State EEG and Neuroimaging Study in Cognitively Healthy Adults*

---

## Research Theme

> "Exploratory characterisation of pre-symptomatic differences in resting-state EEG between cognitively healthy APOE ε4 and PICALM rs3851179 risk carriers, and whether fMRI DMN connectivity adds information beyond EEG."

## Novelty Statement

> "To my knowledge, this is the first within-risk-group comparison of resting-state EEG between APOE ε4 and PICALM carriers in a clinically normal, age-matched cohort, using same-session EEG and fMRI from the public PEARL-Neuro dataset. There is no existing EEG benchmark for this task, so the contribution is analytical clarity about two mechanistically distinct risk pathways, not a higher accuracy on an established task."

The study is **exploratory**. It has no healthy controls, is cross-sectional, and its headline EEGNet result is an optimistic upper bound.

## Contributions (as supported by the data)

1. **Two-pathway comparison** — direct EEG comparison of APOE ε4 and PICALM risk carriers in a presymptomatic cohort (mean age ≈ 55), removing the disease-severity confound of AD-vs-control designs.
2. **Same-session EEG + fMRI** — both modalities from the same individuals, enabling within-subject cross-modal analysis.
3. **Tentative spatial pattern (RQ2)** — topographic maps, in-sample SHAP, and one-fold EEGNet spatial filters point to bilateral temporal and frontal channels. Agreement is encouraging but not independent confirmation.
4. **Honest null on fusion (RQ3)** — DMN connectivity did not improve classification, alone or fused, including in the participants with misclassified epochs. Practical lesson: with N ≈ 68, keep fusion simple (late fusion beat intermediate fusion).
5. **Epoch-consistency analysis (RQ1)** — consistency of epoch-level predictions varied (57.8% of APOE e3/e4 vs 80.0% of PICALM carriers fully consistent, p = .051). Whether this reflects distinctness of signature, recording noise, or checkpoint selection cannot be settled with cross-sectional data.
6. **Companion OASIS-2 analysis** — MMSE alone (AUC 0.814) outperformed structural imaging features (0.678); imaging added only 0.021 to 0.034 AUC. Partly circular because CDR draws on cognition.

---

## Research Questions

**RQ1 — EEG-Only Classification**
Can resting-state EEG separate APOE ε4 carriers from PICALM carriers in cognitively healthy adults, and how consistently do individual recordings point to the correct group?

**RQ2 — Which Features and Regions**
Which EEG features and scalp regions carry the separation, and do different analyses point to the same regions?

**RQ3 — Neuroimaging Contribution**
Does DMN connectivity from fMRI relate to the EEG measures, and does it improve classification when combined with EEG?

### RQ Coverage in Paper

| RQ | Introduction | Methods | Results | Discussion |
|---|---|---|---|---|
| RQ1 | State it | EEGNet, LOSO, epoch vs subject-level metrics | Subject AUC 0.989 (upper bound), LR baseline failed, epoch consistency + Fisher test | Three open explanations (distinctness, noise/state, selection); caution on single-session screening |
| RQ2 | State it | Feature engineering, variance filter, SHAP, spatial filters | Topomaps, SHAP (in-sample), Fold 2 filters | Temporal/frontal hypothesis; band power untestable due to variance filter |
| RQ3 | State it | Late + intermediate fusion, correlation analyses | No fusion benefit, DMN-only at chance, null correlations (exploratory, underpowered) | Ceiling effect; keep fusion simple; OASIS-2 companion (MMSE vs imaging) |

---

## Task Plan

### Phase 1 — Data Acquisition

- Download PEARL-Neuro (OpenNeuro ds004796) via DataLad
- Download OASIS-2 (oasis-brains.org, PART1 + PART2)
- Verify manifest and subject counts
- Delete raw data after preprocessing (free storage)

### Phase 2 — Preprocessing

**PEARL-Neuro EEG**
- Load BrainVision files (127 channels) via mne-bids; eyes-open and eyes-closed not analysed separately
- Band-pass 0.5–45 Hz, 50 Hz notch, downsample to 500 Hz
- Average re-reference, 15-component FastICA for eye artifacts
- 4 s epochs (0.5 s overlap), reject epochs > 150 µV (mean 157.5 epochs per subject, range 7–224)
- QC: 78 of 80 participants retained; one has no epoch file, so EEGNet runs 77 folds

**PEARL-Neuro fMRI**
- Concatenate AP + PA phase-encoding runs
- Harvard-Oxford 48-ROI atlas via nilearn; standardise, detrend, 0.01–0.1 Hz band-pass
- Correlation matrix → 4 DMN features (within-DMN mean, SD, minimum, global mean)

**PEARL-Neuro structural MRI** [skipped — DUA requires institutional legal countersignature; unavailable to independent researchers, confirmed with Nencki Institute, April 2026]

**PEARL-Neuro labels**
- Derived from genotype columns (no pre-made group column)
- APOE risk (label 0): APOE haplotype contains ε4, including any participant who also carries a PICALM risk genotype (48 in feature analysis, 47 in EEGNet evaluation)
- PICALM risk (label 1, n = 30): rs3851179 ≠ "G/G" and no APOE ε4
- Age matched (55.65 vs 54.87 years); no healthy controls

**OASIS-2**
- 150 baseline sessions; FSL BET skull-stripping and FLIRT registration to MNI152
- Intensity normalisation, resize to 128³
- Labels from CDR: 0.0 Control (n = 85), 0.5 MCI (n = 52), ≥ 1.0 AD (n = 13, excluded from binary comparison, leaving N = 137)

**Splits**
- PEARL-Neuro: LOSO (77 folds EEGNet, 78 LR)
- OASIS-2: 5-fold stratified

### Phase 3 — EDA and Baselines

- EEG topographic maps per band (descriptive only, no channel-level test run)
- LR baseline on 521 features after variance filter (LOSO) — scored below chance; treated as a failed baseline
- Spatially targeted 20-feature LR — AUC 0.403
- OASIS-2: MMSE ablation across four configurations (Table SI)

### Phase 4 — Model Training

**PEARL-Neuro**
- EEGNet: F1 = 8, D = 2, F2 = 16, dropout 0.5, 127 channels, 2000-sample epochs; up to 100 epochs, lr 10⁻³, batch 32, weight decay 10⁻³, early stopping patience 12
- **Known flaw:** early stopping, LR scheduling, and checkpoint selection use macro F1 on the held-out participant's own epochs, so performance is an upper bound

**OASIS-2**
- ResNet18 (ImageNet-pretrained) on 2D axial slices, 40 epochs, subject-level predictions averaged across slices
- Sensitivity-favouring threshold t = 0.30 (MCI recall 63.5%, below the > 85% needed for screening)
- Grad-CAM on last conv block — supplementary only, coarse, no regional claims

### Phase 5 — Interpretability and Statistics

- SHAP (LinearExplainer on LR fit on all 78 participants, background 50) — in-sample
- 16 EEGNet spatial filters from Fold 2 only; not compared across folds
- Paired t-test and Wilcoxon on per-participant correctness for model comparisons
- Fisher exact test for epoch-consistency proportions; Mann-Whitney U for age, epoch count, and DMN connectivity between fold types
- Cross-modal correlations (exploratory, N = 68; only r ≈ 0.34 or larger detectable at 80% power)

### Phase 6 — Fusion

- Late fusion: LR (C = 0.1, balanced) on EEGNet probability + 4 DMN scalars (5–6 dims)
- Intermediate fusion: same LR on 992-d EEGNet embedding + 4 DMN scalars (~996 dims, severely underdetermined at ~67 training subjects)
- LOSO on N = 68 participants (ten lacked data for the fusion analysis)
- Subgroup analysis: partition by epoch-level fold F1 (perfect = 1.0; imperfect = 0.26 to 0.50) — descriptive only; not evidence for biological subgroups
- Result: neither fusion beat EEGNet alone; late fusion in the 24 imperfect-fold participants lowered AUC to 0.947 and flipped four correct calls to incorrect

### Phase 7 — Outstanding Work (to confirm the main result)

- [ ] Validation split drawn from training participants only, so checkpoint selection never sees the held-out participant
- [ ] Shuffled-label control for the headline AUC
- [ ] Nest feature selection inside each training fold; rescale features before variance filtering so band power can be tested
- [ ] Compare spatial filters across folds
- [ ] Link EEG measures to the seven PEARL-Neuro psychometric tests (not yet analysed)
- [ ] Longitudinal follow-up of carriers with misclassified epochs
- [ ] Replicate in a larger cohort

### Phase 8 — Paper Draft

See Paper Structure below.

---

## Paper Structure

```
Abstract
├── Background (preclinical genetic risk, APOE ε4 vs PICALM)
├── Methods summary (PEARL-Neuro EEG + fMRI, EEGNet LOSO, OASIS-2 companion)
├── Key results (76/77 subject-level, AUC 0.989 as upper bound; consistency 57.8% vs 80.0%,
│   p = .051; DMN did not help; MMSE 0.814 vs imaging 0.678)
└── Implication: pathway-specific markers worth pursuing; repeated recordings over single-session screening

Introduction
├── Motivation (prevalence, modest treatment effect, need for presymptomatic markers)
├── Genetic risk (APOE ε4, PICALM) and why a within-risk-group comparison helps
├── EEG and fMRI (possible complementarity vs redundancy)
└── Research questions RQ1–RQ3

Literature Review
├── EEG biomarkers in AD and MCI (Jeong 2004, Czigler 2008, Babiloni 2016, Cassani 2018, Ieracitano 2019)
└── Genetic mechanisms (Corder 1993, Harold 2009); PEARL-Neuro cohort

Methods
├── Datasets: PEARL-Neuro (78 → 77 EEGNet, 68 fusion), OASIS-2 (N = 137)
├── Feature engineering (incl. variance-filter caveat)
├── Model architectures: EEGNet, LR, late/intermediate fusion, ResNet18 (Supp. S1)
└── Evaluation: LOSO / 5-fold; statistical tests; SHAP; spatial filters

Results
├── RQ1: LR baseline (failed), EEGNet, epoch consistency
├── RQ2: topographic maps, SHAP, spatial filters (agreement ≠ independent confirmation)
├── RQ3: fusion (no benefit), imperfect-fold subgroup, DMN–EEG correlations (underpowered)
└── OASIS-2 companion

Discussion
├── Three open explanations for epoch-consistency variation
├── Temporal and frontal channels as a hypothesis only
├── Adding fMRI did not help; keep fusion simple
├── MMSE outperformed imaging (partly circular)
└── Implications: repeated measurement, pathway-specific screening, link to cognition

Limitations
├── Optimistic EEGNet evaluation (checkpoint selection on held-out participant)
├── No shuffled-label control; subject-level margins very small (51% within 0.1 of boundary)
├── Small samples; cross-sectional; no healthy controls
├── Variance filter removed band power; in-sample SHAP; single-fold spatial filters
├── Mild LR preprocessing leakage; PEARL-Neuro structural MRI inaccessible (DUA)
└── OASIS-2: n = 52 MCI, partly circular MMSE comparison

Conclusion
Future Work
Supplementary Note S1 (OASIS-2: ResNet18, Grad-CAM, ablation)
```

---

## Required Reading

| Paper | Relevance |
|---|---|
| Dzianok & Kublik, 2024 | PEARL-Neuro dataset paper (Scientific Data 11, 264) |
| Marcus et al., 2010 | OASIS-2 dataset paper |
| Jeong, 2004 | EEG dynamics in AD — grounds theta and complexity features |
| Czigler et al., 2008 | Early-stage AD EEG: alpha, theta, complexity |
| Babiloni et al., 2016 | EEG coherence in MCI/AD — grounds coherence features |
| Cassani et al., 2018 | Systematic review — benchmark map |
| Ieracitano et al., 2019 | CNN on EEG spectrograms, ~88% AD vs HC |
| Lawhern et al., 2018 | EEGNet architecture |
| Wen et al., 2020 | CNNs for AD classification — ResNet18 reference |
| Petersen, 2016 | MCI definition — CDR 0.5 proxy justification |
| Corder et al., 1993 | Original APOE ε4 AD risk paper |
| Harold et al., 2009 | Original PICALM GWAS |
| Selvaraju et al., 2017 | Grad-CAM |
| Lundberg & Lee, 2017 | SHAP |
| GBD 2019 Dementia Collaborators, 2022 | Prevalence forecast |
| van Dyck et al., 2023; Sims et al., 2023 | Lecanemab / donanemab — modest benefit in early symptomatic disease |

### Benchmark Numbers

- Cassani et al. 2018: ~85–90% accuracy, binary AD vs HC (different task)
- Ieracitano et al. 2019: ~88% accuracy, binary AD vs HC (different task)
- No prior benchmark for APOE vs PICALM EEG comparison; the useful comparison is analytical clarity, not prior accuracy
