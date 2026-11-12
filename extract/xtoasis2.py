"""
OASIS-2 Structural MRI Processing Pipeline
===========================================
Rewrite notes (this version fixes real bugs in the previous draft and moves
brain extraction / registration / ROI labeling onto antspyx + antspynet
instead of FSL bet/flirt + nilearn's Harvard-Oxford atlas):

  BUG FIXED — Harvard-Oxford off-by-one: the old ROI extraction built
    `label_to_val = {lbl: idx + 1 for idx, lbl in enumerate(atlas_obj.labels)}`,
    which assumes atlas_obj.labels does NOT include 'Background'. nilearn's
    Harvard-Oxford atlas.labels DOES include 'Background' at index 0, so
    every single ROI volume/intensity in the old script was pulled from the
    WRONG label value (off by one) for every subject, silently. This version
    sidesteps the whole nilearn/Harvard-Oxford path and gets hippocampus,
    amygdala, and the DMN-adjacent cortical ROIs from antspynet's
    Desikan-Killiany-Tourville (DKT) deep-learning labeling instead, using
    the exact label numbers documented in ANTsPyNet's own source (see
    DKT_ROI_LABELS below) — no atlas-index arithmetic left to get wrong.

  BUG FIXED — crash-on-fresh-run __main__ block: the previous version's
    `if __name__ == "__main__":` block had real build_manifest()/run_pipeline()
    calls commented out, but left an unconditional
    `pd.read_csv("data/metadata/oasis2_feature_matrix.csv")` at the bottom
    that would throw FileNotFoundError on any run before that file exists.
    Replaced with fully-commented usage examples, nothing executes on import
    or a bare `python xtoasis2.py`.

  CLEANUP — removed the unused `NiftiLabelsMasker` import (dead code, ROI
    extraction was always done by hand, never through that masker), and
    removed stale PART1/PART2 docstring references left over from an even
    older version of this dataset's layout (this download is flat, no split).

  ARCHITECTURE CHANGE — FSL bet/flirt (+ the nilearn resample_to_img
    fallback for when FSL wasn't installed) is replaced by
    antspynet.preprocess_brain_image(), which does brain extraction, N4 bias
    correction, and template registration in one call. No more "is FSL on
    PATH" branching — antspyx/antspynet is now the only path, so behavior is
    identical across machines instead of silently degrading when FSL is
    missing.

  MRI format:   NIfTI, gzipped (mpr-N.nii.gz) — NOT Analyze 7.5 .hdr/.img
  Acquisitions: 3–4 mpr-N repeated scans per session → averaged before any processing
  Sessions:     555 folders total (OAS2_XXXX_MRY folder naming), flat under one dir
  Labels:       Derived from CDR column — NEVER from Group column or folder name
  Layout:       No per-session RAW/ subfolder — mpr-N.nii.gz files sit directly
                inside each OAS2_XXXX_MRY folder
  Extras:       Some sessions carry an OLD/ subfolder (old FSL t4 transform files,
                not scans) and occasional stray files (.log, .ima.gz, odd-numbered
                .nii.gz from re-runs) sitting alongside the real mpr-N.nii.gz data.
                These are ignored automatically since we only glob top-level
                "mpr-*.nii.gz" — that pattern never touches OLD/ or stray files.
  Manifest:     A manifest.json ships at the root of the extracted folder (session/
                scanner metadata). Not required by this pipeline since the
                demographics .xlsx already carries labels + volumetrics, but it's
                left in place in case you want to cross-check session counts.

Setup (run once before this script):
    Download the OASIS-2 archive from: https://sites.wustl.edu/oasisbrains/
    Extract to:
        data/raw/oasis2/
            OAS2_0001_MR1/mpr-1.nii.gz ...
            OAS2_0001_MR2/mpr-1.nii.gz ...
            ...
            manifest.json
    Place demographics file at:
        data/metadata/oasis_longitudinal_demographics*.xlsx
    (the filename may carry a hash suffix, e.g.
     "oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx" — the script
     globs for it rather than hardcoding the exact name)

Install dependencies:
    pip install nibabel numpy scipy pandas torch openpyxl
    pip install antspyx antspynet tensorflow
    # antspynet's brain extraction / preprocessing / DKT labeling are deep
    # learning models — first call per model downloads pretrained weights
    # (needs internet once; cached under ~/.antspynet/ after that).
    # No FSL, no nilearn, no external CLI tools needed anymore.

Label mapping from CDR:
    CDR 0.0  →  "Control"   (label 0)
    CDR 0.5  →  "MCI"       (label 1) — validated MCI proxy in OASIS literature
    CDR 1.0+ →  "AD"        (label 2) — CDR 1.0 and 2.0 merged

Converter subjects (Group == "Converted") are flagged separately — these
subjects started nondemented and converted during the study. They are your
most analytically valuable subgroup for early-detection framing.

Subject numbering: OAS2_0001 through OAS2_0150 (some sessions may be missing).
MRI ID format:     OAS2_XXXX_MRY (maps exactly to folder names on disk).
"""

import logging
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import ants
import antspynet
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR      = Path("../data/oasis/oasis2")
MRI_FEAT_DIR = Path("../processed/oasis2/mri_features")
TENSOR_DIR   = Path("../processed/oasis2/mri_tensors")
METADATA_DIR = Path("../data/oasis/oasis2/")
SCRATCH_DIR  = Path("../processed/oasis2/mri_scratch")   # intermediate antspyx outputs
LOG_DIR      = Path("logs")

# Demographics filename ships with a hash suffix that varies per download
# (e.g. "oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"), so we glob
# for it lazily via find_demographics_xlsx() rather than hardcoding the name.
DEMOGRAPHICS_XLSX_GLOB = "oasis_longitudinal_demographics*.xlsx"
MANIFEST_CSV           = METADATA_DIR / "oasis2_manifest.csv"
PROCESSING_LOG         = METADATA_DIR / "oasis2_processing_summary.csv"

# Optional: the raw manifest.json that ships at the root of the extracted
# oasis2/ folder (session/scanner metadata). Not required by this pipeline.
RAW_MANIFEST_JSON = RAW_DIR / "manifest.json"

# CNN input shape — (D, H, W) after resize
MRI_TARGET_SHAPE = (128, 128, 128)

# Intensity normalization — clip z-scores beyond this after standardizing
INTENSITY_CLIP_SIGMA = 3.0

# Minimum file size to treat an mpr-N.nii.gz as real data (not a stub).
# These are gzip-compressed NIfTI files now (not raw Analyze .img), so a
# real T1 volume typically still lands in the multi-MB range even compressed.
MIN_BYTES = 500_000

# antspynet.preprocess_brain_image() config for the CNN-tensor track (brain
# extraction + N4 + registration in one call). "Affine" mirrors the old
# FLIRT -dof 12 intent — swap to "SyN" if you want nonlinear registration
# instead, at the cost of significantly more compute per session.
BRAIN_EXTRACTION_MODALITY = "t1"
REGISTRATION_TEMPLATE     = "croppedMni152"   # antspynet's bundled/downloadable template
TEMPLATE_TRANSFORM_TYPE   = "Affine"

# DKT (Desikan-Killiany-Tourville) label numbers, straight from ANTsPyNet's
# own source (antspynet/utilities/desikan_killiany_tourville_labeling.py).
# Subcortical (inner) labels: 17/53 = left/right hippocampus, 18/54 =
# left/right amygdala. Cortical (outer) labels picked as the nearest DKT
# equivalent to the old Harvard-Oxford DMN-adjacent ROI list:
#   - HO "Angular Gyrus" has no separate DKT label; DKT's protocol folds it
#     into "inferior parietal" (1008/2008) — this is a deliberate substitution,
#     not a 1:1 match, flagged here rather than silently assumed.
#   - HO "Frontal Medial Cortex" has no direct DKT equivalent either; nearest
#     anatomical analogue used here is "medial orbitofrontal" (1014/2014) —
#     same caveat, worth a second look if this ROI specifically matters.
DKT_ROI_LABELS = {
    "Left_Hippocampus":  17,  "Right_Hippocampus":  53,
    "Left_Amygdala":     18,  "Right_Amygdala":     54,
    "Left_PosteriorCingulate": 1023, "Right_PosteriorCingulate": 2023,
    "Left_Precuneus":          1025, "Right_Precuneus":          2025,
    "Left_MiddleTemporal":     1015, "Right_MiddleTemporal":     2015,
    "Left_Parahippocampal":    1016, "Right_Parahippocampal":    2016,
    "Left_InferiorParietal":   1008, "Right_InferiorParietal":   2008,  # HO "Angular Gyrus" substitute
    "Left_MedialOrbitofrontal": 1014, "Right_MedialOrbitofrontal": 2014,  # HO "Frontal Medial Cortex" substitute
}

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

for d in [MRI_FEAT_DIR, TENSOR_DIR, METADATA_DIR, SCRATCH_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "oasis2_pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_real_file(path: Path, min_bytes: int = MIN_BYTES) -> bool:
    """
    Check that a file both exists and has real content (not a stub or zero-byte
    placeholder). mpr-N.nii.gz files are gzip-compressed NIfTI volumes, so
    anything below min_bytes is clearly incomplete or corrupt.
    """
    try:
        return path.exists() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def find_demographics_xlsx() -> Path:
    """
    Locate the demographics spreadsheet in METADATA_DIR. The filename ships
    with a hash suffix that varies per download, so we glob instead of
    hardcoding it. Raises FileNotFoundError with setup instructions if
    nothing matches, and warns (but proceeds with the first match) if more
    than one candidate is found.
    """
    matches = sorted(METADATA_DIR.glob(DEMOGRAPHICS_XLSX_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{DEMOGRAPHICS_XLSX_GLOB}' found in {METADATA_DIR}.\n"
            "Download oasis_longitudinal_demographics.xlsx from:\n"
            "  https://sites.wustl.edu/oasisbrains/"
        )
    if len(matches) > 1:
        log.warning(
            f"Multiple demographics files found in {METADATA_DIR}: "
            f"{[m.name for m in matches]} — using {matches[0].name}"
        )
    return matches[0]


def cdr_to_label(cdr) -> str:
    """
    Map CDR score to 3-class label.
    Always use CDR column — Group column is NOT equivalent.
    CDR 0.5 == 'Very mild dementia' in OASIS terminology == 'MCI' in ours.
    CDR 1.0 and 2.0 are merged into 'AD'.
    """
    if cdr == 0.0:
        return "Control"
    if cdr == 0.5:
        return "MCI"
    return "AD"


def find_session_raw_dir(mri_id: str) -> Optional[Path]:
    """
    Locate the session folder for one MRI ID. The mpr-N.nii.gz files sit
    directly inside data/raw/oasis2/OAS2_XXXX_MRY/ — no per-session RAW/
    subfolder. Returns the Path if found, None otherwise.
    """
    candidate = RAW_DIR / mri_id
    return candidate if candidate.exists() else None


def load_and_average_acquisitions(raw_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all mpr-N.nii.gz acquisitions sitting directly inside a session
    folder and average them together. This is the standard OASIS protocol —
    multiple acquisitions are collected per session to reduce motion
    artifact, and they must be averaged before any subsequent processing.

    Explicitly globs top-level "mpr-*.nii.gz" only (non-recursive), which
    naturally skips the OLD/ subfolder (old t4 transform files, not scans)
    and any stray .log / .ima.gz / odd-numbered .nii.gz files that some
    sessions carry alongside the real data — none of those match the
    "mpr-*.nii.gz" pattern.

    Returns:
        averaged  — float32 numpy array of shape (X, Y, Z)
        affine    — 4x4 affine matrix from the first acquisition
    """
    nii_files = sorted(raw_dir.glob("mpr-*.nii.gz"))
    if not nii_files:
        raise FileNotFoundError(f"No mpr-*.nii.gz files in {raw_dir}")

    # Verify each file has real content (not a truncated/corrupt download)
    valid = [f for f in nii_files if is_real_file(f)]
    skipped = set(nii_files) - set(valid)
    for f in skipped:
        log.warning(f"Skipping {f.name} — file missing or too small")

    if not valid:
        raise FileNotFoundError(f"No valid mpr-*.nii.gz files in {raw_dir}")

    volumes = []
    affine  = None
    for nii_path in valid:
        img = nib.load(str(nii_path))
        if affine is None:
            affine = img.affine
        arr = img.get_fdata(dtype=np.float32)
        # OASIS-2 mpr-N.nii.gz volumes sometimes load as 4D (X, Y, Z, 1) —
        # squeeze to 3D before averaging. np.squeeze removes ALL size-1 dims.
        arr = np.squeeze(arr)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected volume shape after squeeze: {arr.shape}")
        volumes.append(arr)

    averaged = np.mean(volumes, axis=0)
    log.info(f"  Averaged {len(volumes)} acquisition(s) → shape {averaged.shape}")
    return averaged, affine


def antspynet_preprocess(t1_ants) -> dict:
    """
    Brain extraction + N4 bias correction + registration to
    REGISTRATION_TEMPLATE, all in one antspynet call. Replaces the old FSL
    bet + FSL flirt (+ nilearn resample_to_img fallback) path entirely.

    Returns antspynet's own preprocessing dict — the two keys used
    downstream are 'preprocessed_image' (registered, bias-corrected, NOT yet
    masked) and 'brain_mask' (in that same registered space). Multiply them
    together to get the skull-stripped, registered volume — same pattern
    ANTsPyNet's own DKT labeling code uses internally.
    """
    return antspynet.preprocess_brain_image(
        t1_ants,
        truncate_intensity=(0.01, 0.99),
        brain_extraction_modality=BRAIN_EXTRACTION_MODALITY,
        template=REGISTRATION_TEMPLATE,
        template_transform_type=TEMPLATE_TRANSFORM_TYPE,
        do_bias_correction=True,
        do_denoising=True,
        verbose=False,
    )


def ants_to_nibabel_via_disk(ants_img, tmp_path: Path) -> nib.Nifti1Image:
    """
    Bridge an ANTsImage back to a nibabel image via a real file on disk
    rather than antspyx's in-memory to_nibabel() converter — that converter
    has had open, version-dependent bugs around qform/orientation on round
    trips (ANTsX/ANTsPy issues #52, #64). Writing through ANTs' own NIfTI
    writer and reading it back with nibabel is the same trick to_nibabel()
    uses internally anyway, just made explicit and version-safe here.
    """
    ants.image_write(ants_img, str(tmp_path))
    return nib.load(str(tmp_path))


def normalize_intensity(arr: np.ndarray) -> np.ndarray:
    """
    Z-score normalize the volume and clip to ±INTENSITY_CLIP_SIGMA.
    Applied after skull stripping and registration so only brain voxels
    influence the mean/std calculation.

    Brain voxels are estimated as values above zero (post-skull-strip).
    This avoids background zeros from inflating the mean.
    """
    brain_mask = arr > 0
    if brain_mask.sum() < 1000:
        log.warning("Very few brain voxels — normalization may be unreliable")

    mu    = arr[brain_mask].mean() if brain_mask.any() else arr.mean()
    sigma = arr[brain_mask].std()  if brain_mask.any() else arr.std()

    normalized = (arr - mu) / (sigma + 1e-8)
    normalized = np.clip(normalized, -INTENSITY_CLIP_SIGMA, INTENSITY_CLIP_SIGMA)
    return normalized.astype(np.float32)


def resize_volume(arr: np.ndarray, target: tuple = MRI_TARGET_SHAPE) -> np.ndarray:
    """
    Resize a 3D volume to target shape using trilinear zoom (order=1).
    Order 1 is a good balance — faster than cubic (order=3) with acceptable
    quality for CNN input at 128^3.
    """
    factors = tuple(t / s for t, s in zip(target, arr.shape))
    return zoom(arr, factors, order=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — BUILD MANIFEST (run once before processing)
# ─────────────────────────────────────────────────────────────────────────────

def build_manifest() -> pd.DataFrame:
    """
    Join the demographics Excel file with the actual filesystem to produce
    a manifest CSV that links every MRI ID to its folder path, label, and
    pre-computed features (nWBV, eTIV, MMSE).

    Run this once before processing. It will tell you:
    - Which sessions are present on disk vs missing entirely (download incomplete)
    - How many mpr-N acquisitions each session has
    - Label distribution (Control / MCI / AD)
    - Converter subjects (Group == 'Converted')

    Saves to data/metadata/oasis2_manifest.csv and returns the DataFrame.
    """
    demographics_xlsx = find_demographics_xlsx()
    df = pd.read_excel(str(demographics_xlsx))

    # Normalize column names — the xlsx sometimes has trailing spaces
    df.columns = df.columns.str.strip()

    # ── Label derivation — CDR column only ───────────────────────────────────
    df["label"]          = df["CDR"].apply(cdr_to_label)
    df["label_int"]      = df["label"].map({"Control": 0, "MCI": 1, "AD": 2})
    df["is_converter"]   = df["Group"].str.strip() == "Converted"

    # ── Handle known missing values ───────────────────────────────────────────
    # 19 sessions missing SES, 2 sessions missing MMSE — document don't drop
    df["ses_imputed"]    = df["SES"].isna()
    df["SES"]            = df["SES"].fillna(df["SES"].median())
    df["mmse_missing"]   = df["MMSE"].isna()

    # ── Verify filesystem presence ────────────────────────────────────────────
    records  = []
    missing  = []

    for _, row in df.iterrows():
        mri_id  = str(row["MRI ID"]).strip()
        raw_dir = find_session_raw_dir(mri_id)

        if raw_dir is None:
            missing.append(mri_id)
            continue

        nii_files = list(raw_dir.glob("mpr-*.nii.gz"))
        nii_ok    = [f for f in nii_files if is_real_file(f)]

        records.append({
            "mri_id":        mri_id,
            "subject_id":    str(row["Subject ID"]).strip(),
            "visit":         row["Visit"],
            "cdr":           row["CDR"],
            "label":         row["label"],
            "label_int":     row["label_int"],
            "is_converter":  row["is_converter"],
            "group":         str(row["Group"]).strip(),
            "age":           row["Age"],
            "sex":           str(row["M/F"]).strip(),
            "mmse":          row["MMSE"],
            "mmse_missing":  row["mmse_missing"],
            "educ":          row.get("EDUC", np.nan),
            "ses":           row["SES"],
            "ses_imputed":   row["ses_imputed"],
            "nWBV":          row["nWBV"],
            "eTIV":          row["eTIV"],
            "ASF":           row["ASF"],
            "n_acquisitions": len(nii_ok),
            "raw_path":      str(raw_dir),
        })

    manifest = pd.DataFrame(records)
    manifest.to_csv(str(MANIFEST_CSV), index=False)

    log.info(f"Sessions found:     {len(manifest)}")
    log.info(f"Sessions missing:   {len(missing)}")
    log.info(f"Label distribution:\n{manifest['label'].value_counts().to_string()}")
    log.info(f"Converters:         {manifest['is_converter'].sum()} sessions")
    log.info(f"nWBV sanity check (CDR → mean nWBV — expect 0.740 / 0.721 / 0.702):")
    log.info(manifest.groupby("cdr")["nWBV"].mean().round(4).to_string())

    if missing:
        log.warning(f"Missing sessions (first 10): {missing[:10]}")
        log.warning(f"Check the oasis2 archive extracted fully to {RAW_DIR}.")

    log.info(f"Manifest saved → {MANIFEST_CSV}")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD + PREPROCESS MRI (one session)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_mri_session(mri_id: str) -> Optional[nib.Nifti1Image]:
    """
    Full preprocessing pipeline for one OASIS-2 session:
      1. Locate session folder
      2. Load + average all mpr-N acquisitions
      3. antspynet.preprocess_brain_image(): brain extraction + N4 + registration
         to REGISTRATION_TEMPLATE, all via antspyx (no FSL, no nilearn)
      4. Intensity normalize (z-score, clipped to ±3 sigma)

    Returns a preprocessed Nifti1Image in template space, or None on failure.
    """
    raw_dir = find_session_raw_dir(mri_id)
    if raw_dir is None:
        log.error(f"[{mri_id}] Session directory not found under {RAW_DIR}")
        return None

    scratch = SCRATCH_DIR / mri_id
    scratch.mkdir(parents=True, exist_ok=True)

    # ── Load + average acquisitions ───────────────────────────────────────────
    try:
        averaged, affine = load_and_average_acquisitions(raw_dir)
    except Exception as e:
        log.error(f"[{mri_id}] Acquisition averaging failed: {e}")
        return None

    avg_nii_path = scratch / "averaged.nii.gz"
    nib.save(nib.Nifti1Image(averaged, affine), str(avg_nii_path))

    # ── Brain extraction + N4 + registration (antspynet, one call) ───────────
    try:
        t1_ants = ants.image_read(str(avg_nii_path))
        preproc = antspynet_preprocess(t1_ants)
        brain_mni = preproc["preprocessed_image"] * preproc["brain_mask"]
        log.info(f"[{mri_id}] antspynet preprocessing done "
                 f"(brain extraction + N4 + {TEMPLATE_TRANSFORM_TYPE} registration to {REGISTRATION_TEMPLATE})")
    except Exception as e:
        log.error(f"[{mri_id}] antspynet preprocessing failed: {e}")
        return None

    mni_nii_path = scratch / "brain_mni.nii.gz"
    registered_img = ants_to_nibabel_via_disk(brain_mni, mni_nii_path)

    # ── Intensity normalization ───────────────────────────────────────────────
    arr        = registered_img.get_fdata(dtype=np.float32)
    normalized = normalize_intensity(arr)
    final_img  = nib.Nifti1Image(normalized, registered_img.affine)

    log.info(f"[{mri_id}] Preprocessed: shape {normalized.shape}, "
             f"range [{normalized.min():.2f}, {normalized.max():.2f}]")
    return final_img


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FEATURE EXTRACTION (one session)
# ─────────────────────────────────────────────────────────────────────────────

def extract_mri_features(mri_id: str, manifest_row: pd.Series) -> dict:
    """
    Extract handcrafted MRI features for one session.

    Two categories:
    (A) Pre-computed from demographics CSV — available immediately, no image needed:
        nWBV, eTIV, ASF, MMSE, Age, EDUC, SES

    (B) DKT-labeled ROI volumes — antspynet.desikan_killiany_tourville_labeling()
        run directly on the averaged native-space T1 (do_preprocessing=True,
        so it does its own internal brain extraction + registration and warps
        the label image back to the ORIGINAL native T1 space). Volumes and
        mean intensities are measured against that same native-space T1, not
        the template-registered CNN-preprocessing output from Step 1 — these
        are two independent antspynet calls on purpose (see antspynet_preprocess
        docstring for why they aren't shared).

    Returns a flat dict ready to be saved as one row in a features CSV.
    """
    features = {
        "mri_id":       mri_id,
        "subject_id":   manifest_row["subject_id"],
        "visit":        manifest_row["visit"],
        "label":        manifest_row["label"],
        "label_int":    manifest_row["label_int"],
        "is_converter": manifest_row["is_converter"],
        # ── (A) Pre-computed CSV features — use directly ──────────────────────
        "nWBV":         manifest_row["nWBV"],
        "eTIV":         manifest_row["eTIV"],
        "ASF":          manifest_row["ASF"],
        "mmse":         manifest_row["mmse"],
        "age":          manifest_row["age"],
        "educ":         manifest_row.get("educ", np.nan),
        "ses":          manifest_row["ses"],
    }

    e_tiv = float(manifest_row["eTIV"]) if manifest_row["eTIV"] > 0 else 1.0

    try:
        raw_dir = find_session_raw_dir(mri_id)
        avg_nii_path = SCRATCH_DIR / mri_id / "averaged.nii.gz"
        if not avg_nii_path.exists():
            raise FileNotFoundError(
                f"{avg_nii_path} not found — run preprocess_mri_session() first, "
                f"it writes this as a side effect."
            )

        t1_native = ants.image_read(str(avg_nii_path))
        dkt_label_image = antspynet.desikan_killiany_tourville_labeling(
            t1_native, do_preprocessing=True, verbose=False
        )
        dkt_array = dkt_label_image.numpy()
        t1_array  = t1_native.numpy()
        vox_vol_mm3 = float(np.prod(t1_native.spacing))

        for roi_name, label_val in DKT_ROI_LABELS.items():
            roi_mask = dkt_array == label_val
            vol_mm3  = float(roi_mask.sum()) * vox_vol_mm3
            vol_norm = vol_mm3 / e_tiv
            mean_int = float(t1_array[roi_mask].mean()) if roi_mask.any() else np.nan

            features[f"vol_{roi_name}"]            = vol_mm3
            features[f"vol_{roi_name}_norm"]        = vol_norm
            features[f"mean_intensity_{roi_name}"]  = mean_int

        # ── Derived hippocampal features ───────────────────────────────────
        vol_l = features.get("vol_Left_Hippocampus",  np.nan)
        vol_r = features.get("vol_Right_Hippocampus", np.nan)

        if not (np.isnan(vol_l) or np.isnan(vol_r)):
            features["hippo_vol_bilateral"]      = vol_l + vol_r
            features["hippo_vol_bilateral_norm"] = (
                features["vol_Left_Hippocampus_norm"] +
                features["vol_Right_Hippocampus_norm"]
            )
            denom = vol_l + vol_r
            features["hippo_asymmetry_index"] = (
                (vol_l - vol_r) / (denom + 1e-8) if denom > 0 else np.nan
            )
        else:
            features["hippo_vol_bilateral"]      = np.nan
            features["hippo_vol_bilateral_norm"] = np.nan
            features["hippo_asymmetry_index"]    = np.nan

        log.info(f"[{mri_id}] DKT ROI extraction done — "
                 f"hippo bilateral = {features.get('hippo_vol_bilateral', np.nan):.1f} mm³")

    except Exception as e:
        log.error(f"[{mri_id}] DKT ROI feature extraction failed: {e}")
        for roi_name in DKT_ROI_LABELS:
            features[f"vol_{roi_name}"]            = np.nan
            features[f"vol_{roi_name}_norm"]       = np.nan
            features[f"mean_intensity_{roi_name}"] = np.nan
        features["hippo_vol_bilateral"]      = np.nan
        features["hippo_vol_bilateral_norm"] = np.nan
        features["hippo_asymmetry_index"]    = np.nan

    return features


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SAVE CNN TENSOR (one session)
# ─────────────────────────────────────────────────────────────────────────────

def save_cnn_tensor(mri_id: str, preprocessed_img: nib.Nifti1Image) -> bool:
    """
    Resize the preprocessed volume to MRI_TARGET_SHAPE and save as a
    PyTorch tensor (.pt) for CNN training.

    Output shape: [1, D, H, W] — single-channel 3D volume.
    Saved to data/processed/mri_tensors/{mri_id}/mri_tensor.pt

    Returns True on success, False on failure.
    """
    try:
        arr     = preprocessed_img.get_fdata(dtype=np.float32)
        resized = resize_volume(arr, MRI_TARGET_SHAPE)
        tensor  = torch.tensor(resized).float().unsqueeze(0)  # [1, D, H, W]

        out_dir = TENSOR_DIR / mri_id
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, str(out_dir / "mri_tensor.pt"))

        log.info(f"[{mri_id}] CNN tensor saved: {tuple(tensor.shape)}")
        return True
    except Exception as e:
        log.error(f"[{mri_id}] Tensor save failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CLEAN UP SCRATCH (optional, per session)
# ─────────────────────────────────────────────────────────────────────────────

def clean_scratch(mri_id: str):
    """
    Remove intermediate antspyx files for this session from SCRATCH_DIR.
    The raw mpr-N.nii.gz files under data/raw/oasis2/ are NOT touched —
    OASIS-2 is a direct download so we cannot re-fetch on demand. Only
    remove scratch, keep raw.

    NOTE: extract_mri_features() reads SCRATCH_DIR/{mri_id}/averaged.nii.gz,
    so don't clean scratch before features have been extracted for a session
    (process_session() below already sequences this correctly).
    """
    scratch = SCRATCH_DIR / mri_id
    if scratch.exists():
        shutil.rmtree(str(scratch))
        log.info(f"[{mri_id}] Scratch cleaned")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — one session, all steps
# ─────────────────────────────────────────────────────────────────────────────

def process_session(mri_id: str, manifest_row: pd.Series,
                    clean_after: bool = True) -> dict:
    """
    Full pipeline for one OASIS-2 MRI session.

    Steps:
      1. Preprocess MRI (average acquisitions → antspynet brain extraction +
         N4 + registration → normalize)
      2. Extract ROI features (antspynet DKT labeling + pre-computed CSV features)
      3. Save CNN tensor (128^3 float32)
      4. Optionally clean intermediate scratch files

    A failure in feature extraction does NOT abort tensor saving and vice versa.
    A session is still useful if at least one output is produced.

    Args:
        mri_id:       OASIS-2 MRI ID string, e.g. "OAS2_0001_MR1"
        manifest_row: corresponding row from the manifest DataFrame
        clean_after:  set False during debugging to inspect intermediate files
    """
    status = {
        "mri_id":      mri_id,
        "subject_id":  manifest_row["subject_id"],
        "visit":       manifest_row["visit"],
        "label":       manifest_row["label"],
        "is_converter": manifest_row["is_converter"],
        "timestamp":   datetime.now().isoformat(),
        "preprocess":  "skip",
        "features":    "skip",
        "tensor":      "skip",
        "overall":     "unknown",
        "notes":       "",
    }

    log.info(f"\n{'='*60}\nProcessing {mri_id}  [{manifest_row['label']}]"
             f"  Visit {manifest_row['visit']}\n{'='*60}")

    feat_path   = MRI_FEAT_DIR / f"{mri_id}_features.csv"
    tensor_path = TENSOR_DIR   / mri_id / "mri_tensor.pt"

    if feat_path.exists() and tensor_path.exists():
        log.info(f"[{mri_id}] Already processed — skipping")
        status["preprocess"] = status["features"] = status["tensor"] = "skip"
        status["overall"]    = "already_done"
        _append_session_status(status)
        return status

    # Step 1 — preprocess
    preprocessed = preprocess_mri_session(mri_id)
    if preprocessed is None:
        status["preprocess"] = "failed"
        status["overall"]    = "failed_preprocess"
        status["notes"]      = "MRI preprocessing failed."
        _append_session_status(status)
        return status
    status["preprocess"] = "ok"

    # Step 2 — features (runs even if tensor fails)
    if not feat_path.exists():
        try:
            features = extract_mri_features(mri_id, manifest_row)
            pd.DataFrame([features]).to_csv(str(feat_path), index=False)
            log.info(f"[{mri_id}] Features saved → {feat_path.name}")
            status["features"] = "ok"
        except Exception as e:
            log.error(f"[{mri_id}] Feature extraction crashed: {e}")
            status["features"]  = "failed"
            status["notes"]    += f"Features failed: {e}. "
    else:
        status["features"] = "skip"

    # Step 3 — CNN tensor (independent of features)
    if not tensor_path.exists():
        ok = save_cnn_tensor(mri_id, preprocessed)
        status["tensor"] = "ok" if ok else "failed"
        if not ok:
            status["notes"] += "Tensor failed. "
    else:
        status["tensor"] = "skip"

    # Step 4 — clean scratch
    if clean_after:
        clean_scratch(mri_id)

    feat_ok   = status["features"] in ("ok", "skip")
    tensor_ok = status["tensor"]   in ("ok", "skip")
    status["overall"] = (
        "success" if (feat_ok and tensor_ok)
        else "partial" if (feat_ok or tensor_ok)
        else "failed"
    )

    log.info(f"[{mri_id}] Done — {status['overall']}")
    _append_session_status(status)
    return status


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(mri_ids: list, clean_after: bool = True) -> pd.DataFrame:
    """
    Process a list of MRI IDs sequentially. Continues past individual failures.

    Usage:
        manifest = build_manifest()
        run_pipeline(manifest["mri_id"].tolist()[:5], clean_after=False)   # test
        run_pipeline(manifest["mri_id"].tolist(), clean_after=True)         # full run

    For batched runs across sessions:
        batches = [all_ids[i:i+20] for i in range(0, len(all_ids), 20)]
        run_pipeline(batches[BATCH_TO_RUN], clean_after=True)
    """
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"{MANIFEST_CSV} not found. Run build_manifest() first."
        )

    manifest = pd.read_csv(str(MANIFEST_CSV))
    manifest_lookup = {row["mri_id"]: row for _, row in manifest.iterrows()}

    statuses = []
    for mid in mri_ids:
        if mid not in manifest_lookup:
            log.error(f"[{mid}] Not in manifest — skipping (run build_manifest() again?)")
            statuses.append({
                "mri_id": mid, "timestamp": datetime.now().isoformat(),
                "overall": "not_in_manifest", "notes": "MRI ID missing from manifest CSV."
            })
            continue
        try:
            statuses.append(process_session(mid, manifest_lookup[mid], clean_after=clean_after))
        except Exception as e:
            log.error(f"[{mid}] Crashed: {e}\n{traceback.format_exc()}")
            statuses.append({
                "mri_id": mid, "timestamp": datetime.now().isoformat(),
                "overall": "crashed", "notes": str(e)
            })

    summary = pd.DataFrame(statuses)
    summary.to_csv(str(PROCESSING_LOG), index=False)
    log.info("\nOASIS-2 Pipeline complete.")
    if "overall" in summary.columns:
        log.info(summary["overall"].value_counts().to_string())
    return summary


def _append_session_status(status: dict):
    """Upsert a session's status row into the rolling processing log CSV."""
    path = PROCESSING_LOG
    row  = pd.DataFrame([status])
    if path.exists():
        df  = pd.read_csv(str(path))
        df  = df[df["mri_id"] != status["mri_id"]]
        row = pd.concat([df, row], ignore_index=True)
    row.to_csv(str(path), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def assemble_feature_matrix(
    baseline_only: bool = True,
    flag_converters: bool = True,
) -> pd.DataFrame:
    """
    Merge all per-session feature CSVs into one flat matrix.
    Call this after all sessions are processed.

    Args:
        baseline_only:    If True, keep only Visit 1 per subject to avoid
                          data leakage in cross-sectional experiments.
                          Set to False for longitudinal experiments — but then
                          use subject-level train/test splits (never session-level).
        flag_converters:  If True, adds an 'is_converter' boolean column.
                          Converter sessions (subjects who started nondemented
                          and converted during the study) are kept in the matrix
                          but flagged for separate subgroup analysis.

    Returns a DataFrame saved to data/metadata/oasis2_feature_matrix.csv.
    """
    feat_files = sorted(MRI_FEAT_DIR.glob("OAS2_*_features.csv"))
    if not feat_files:
        raise FileNotFoundError(f"No feature CSVs found in {MRI_FEAT_DIR}")

    dfs   = [pd.read_csv(f) for f in feat_files]
    merged = pd.concat(dfs, ignore_index=True)

    if baseline_only:
        merged = merged[merged["visit"] == 1].copy()
        log.info(f"Baseline only (Visit 1): {len(merged)} sessions")

    log.info(f"Label distribution:\n{merged['label'].value_counts().to_string()}")
    if flag_converters:
        log.info(f"Converter sessions: {merged['is_converter'].sum()}")

    # nWBV sanity check — should decrease monotonically with CDR
    log.info("nWBV by label (sanity check — expect Control > MCI > AD):")
    log.info(merged.groupby("label")["nWBV"].mean().round(4).to_string())

    out = METADATA_DIR / "oasis2_feature_matrix.csv"
    merged.to_csv(str(out), index=False)
    log.info(f"Feature matrix: {merged.shape} → {out}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Nothing runs automatically on `python xtoasis2.py` — uncomment the
    # steps you actually want, in order, and run them one at a time.
    # Building the manifest first lets you eyeball session counts and label
    # distribution before spending antspynet compute on anything.

    # Step 0 — build manifest (run once, inspect output before processing)
    # manifest = build_manifest()
    # print(manifest[["mri_id", "label", "visit", "is_converter", "nWBV"]].head(10).to_string())

    # Single session test — keep scratch for inspection
    # manifest = pd.read_csv(str(MANIFEST_CSV))
    # row = manifest[manifest["mri_id"] == "OAS2_0001_MR1"].iloc[0]
    # process_session("OAS2_0001_MR1", row, clean_after=False)

    # Full run — all sessions
    manifest = pd.read_csv(str(MANIFEST_CSV))
    # run_pipeline(manifest["mri_id"].tolist(), clean_after=True)

    # Batched run — recommended given antspynet's per-session compute cost
    all_ids    = manifest["mri_id"].tolist()
    BATCH_SIZE = 50
    batches    = [all_ids[i:i + BATCH_SIZE] for i in range(0, len(all_ids), BATCH_SIZE)]
    BATCH_TO_RUN = 0  # change this each session: 0, 1, 2 ... up to len(batches)-1
    summary = run_pipeline(batches[BATCH_TO_RUN], clean_after=True)
    print(summary[["mri_id", "label", "overall"]].to_string())

    # Assemble feature matrix after all sessions processed
    # df = assemble_feature_matrix(baseline_only=True, flag_converters=True)
    # print(df.groupby("label")[["nWBV", "hippo_vol_bilateral_norm", "hippo_asymmetry_index"]].mean().round(4))
    pass
