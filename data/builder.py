"""
dataset/builder.py

Builds a multimodal ROI QC dataset.

Each sample contains:

patches.npy
    shape: (N, 2, H, W)

    channel 0:
        normalized mean fluorescence image patch

    channel 1:
        binary ROI mask patch


signals.npy
    shape: (N, 2, T)

    channel 0:
        fluorescence trace from qc_results/fluo.npy

    channel 1:
        neuropil trace from qc_results/neuropil.npy


metrics.npy
    shape: (N, M)

    morphology and fluorescence-derived metrics


labels.npy
    shape: (N,)

    1 = good
    0 = bad


roi.npy
    shape: (N,)

    zero-based ROI index


session.npy
    shape: (N,)

    session name


metric_names.npy
    shape: (M,)

    names corresponding to columns in metrics.npy


Important ROI alignment:

    ROI index i

        fluo[i]
        neuropil[i]
        stat[i]
        mask value == i + 1
        ROI_label.h5 label ID == i

The QC results contain the post-filtered ROI subset, so all of the
signal/stat/mask arrays should have the same number of ROIs.

Important preprocessing decision:

    The builder handles:
        - ROI alignment
        - fixed-length resampling

    dataset.py handles:
        - signal normalization
        - metric normalization
        - augmentation

This prevents normalization from being performed twice.
"""

import os

import numpy as np
import h5py
from tqdm import tqdm

from .crop import make_two_channel_patch


# =========================================================================
# CONFIGURATION
# =========================================================================

TRACE_LENGTH = 2048


# =========================================================================
# METRIC DEFINITIONS
# =========================================================================

METRIC_NAMES = np.array([
    # Morphology / Suite2p stat metrics
    "footprint",
    "compact",
    "solidity",
    "npix",
    "npix_soma",
    "radius",
    "aspect_ratio",
    "npix_norm_no_crop",
    "npix_norm",

    # Existing fluorescence statistics
    "stat_skew",
    "stat_std",

    # Fluorescence trace statistics
    "trace_mean",
    "trace_std",
    "trace_skew",
    "trace_p95",
    "trace_p50",
    "trace_p05",

    # Simple SNR-like statistics
    "snr_p95_p50",
    "snr_p95_p05",

    # Activity statistics
    "trace_max",
    "trace_range",

    # Neuropil relationship
    "neuropil_mean",
    "neuropil_std",
    "fluo_neuropil_corr",

], dtype="<U64")


# =========================================================================
# LABEL LOADING
# =========================================================================

def load_labels(session_path, n_rois, qc_stat=None):
    """Read labels only; prefer HDF5, otherwise map original Suite2p NPY labels."""
    from pathlib import Path
    from .alignment import map_qc_to_original

    session = Path(session_path)
    plane = session / "suite2p" / "plane0"
    result = np.full(n_rois, -1, dtype=np.int8)

    def mapping():
        current = qc_stat
        if current is None:
            current = np.load(session / "qc_results" / "stat.npy", allow_pickle=True)
        if len(current) != n_rois:
            raise ValueError("QC stat and fluorescence counts disagree.")
        original = np.load(plane / "stat.npy", allow_pickle=True)
        return original, map_qc_to_original(current, original)

    h5_path = session / "ROI_label.h5"
    if h5_path.exists():
        with h5py.File(h5_path, "r") as handle:
            raw_space = handle.attrs.get("index_space", "qc")
            space = raw_space.decode() if isinstance(raw_space, bytes) else str(raw_space)
            good = np.asarray(handle["good_roi"][:]) if "good_roi" in handle else np.array([], dtype=int)
            bad = np.asarray(handle["bad_roi"][:]) if "bad_roi" in handle else np.array([], dtype=int)
        if space not in {"qc", "qc_results", "", "suite2p_original"}:
            raise ValueError(f"Unsupported HDF5 label index space: {space}")
        if space == "suite2p_original":
            original, indices = mapping()
            limit = len(original)
        else:
            indices, limit = None, n_rois
        for values in [good, bad]:
            if (values.ndim != 1 or not np.issubdtype(values.dtype, np.number)
                    or not np.isfinite(values).all() or not np.all(values == values.astype(int))
                    or np.any(values < 0) or np.any(values >= limit)
                    or len(np.unique(values)) != len(values)):
                raise ValueError(f"{h5_path}: invalid or duplicate label indices.")
        if len(np.intersect1d(good, bad)):
            raise ValueError(f"{h5_path}: conflicting good/bad labels.")
        labels = np.full(limit, -1, dtype=np.int8)
        labels[good.astype(int)] = 1
        labels[bad.astype(int)] = 0
        result = labels if indices is None else labels[indices]
        print(f"Labels: {h5_path} ({space})")
        return result

    npy_path = plane / "roi_manual_labels.npy"
    if not npy_path.exists():
        print(f"WARNING: No supported manual labels in {session}; no training samples.")
        return result
    labels = np.load(npy_path, allow_pickle=False)
    original, indices = mapping()
    if labels.ndim != 1 or len(labels) != len(original) or not np.issubdtype(labels.dtype, np.number):
        raise ValueError(f"{npy_path}: expected numeric labels for original Suite2p rows.")
    if not np.all(np.isin(labels, [0, 1, 2]) | np.isnan(labels)):
        raise ValueError(f"{npy_path}: expected 0, 1, 2, or NaN.")
    mapped = labels[indices]
    result[mapped == 0] = 0
    result[mapped == 1] = 1
    print(f"Labels: {npy_path} (original Suite2p -> QC)")
    return result



# =========================================================================
# SESSION LOADING
# =========================================================================

def load_session(session_path):
    """
    Load all data required for one QC session.
    """

    # ---------------------------------------------------------------------
    # Mean image
    # ---------------------------------------------------------------------

    ops_path = os.path.join(
        session_path,
        "suite2p",
        "plane0",
        "ops.npy",
    )

    ops = np.load(
        ops_path,
        allow_pickle=True,
    ).item()

    mean_img = ops["meanImg"].astype(
        np.float32
    )


    # ---------------------------------------------------------------------
    # Spatial mask
    # ---------------------------------------------------------------------

    masks_path = os.path.join(
        session_path,
        "qc_results",
        "masks.npy",
    )

    masks = np.load(
        masks_path,
        mmap_mode="r",
    )


    # ---------------------------------------------------------------------
    # Fluorescence traces
    # ---------------------------------------------------------------------

    fluo_path = os.path.join(
        session_path,
        "qc_results",
        "fluo.npy",
    )

    fluo = np.load(
        fluo_path,
        mmap_mode="r",
    )


    # ---------------------------------------------------------------------
    # Neuropil traces
    # ---------------------------------------------------------------------

    neuropil_path = os.path.join(
        session_path,
        "qc_results",
        "neuropil.npy",
    )

    neuropil = np.load(
        neuropil_path,
        mmap_mode="r",
    )


    # ---------------------------------------------------------------------
    # ROI statistics
    # ---------------------------------------------------------------------

    stat_path = os.path.join(
        session_path,
        "qc_results",
        "stat.npy",
    )

    stat = np.load(
        stat_path,
        allow_pickle=True,
    )


    # ---------------------------------------------------------------------
    # Validate ROI alignment
    # ---------------------------------------------------------------------

    n_rois = fluo.shape[0]

    if neuropil.shape[0] != n_rois:

        raise ValueError(
            f"Neuropil ROI count does not match fluo.\n"
            f"fluo:     {fluo.shape[0]}\n"
            f"neuropil: {neuropil.shape[0]}"
        )

    if len(stat) != n_rois:

        raise ValueError(
            f"stat ROI count does not match fluo.\n"
            f"fluo: {fluo.shape[0]}\n"
            f"stat: {len(stat)}"
        )


    # ---------------------------------------------------------------------
    # Check mask IDs
    # ---------------------------------------------------------------------

    mask_ids = np.unique(masks)

    mask_ids = mask_ids[
        mask_ids > 0
    ].astype(int)

    expected_ids = np.arange(
        1,
        n_rois + 1,
    )

    if not np.array_equal(
        mask_ids,
        expected_ids,
    ):

        raise ValueError(
            "Mask ROI IDs do not match expected "
            f"1..{n_rois}."
        )


    # ---------------------------------------------------------------------
    # Labels
    # ---------------------------------------------------------------------

    labels = load_labels(
        session_path,
        n_rois,
        qc_stat=stat,
    )


    return {
        "mean_img": mean_img,
        "masks": masks,
        "fluo": fluo,
        "neuropil": neuropil,
        "stat": stat,
        "labels": labels,
        "n_rois": n_rois,
    }


# =========================================================================
# TRACE PROCESSING
# =========================================================================

def downsample_trace(
    trace,
    target_length,
):
    """
    Resample a trace to a fixed length.

    Normalization is intentionally NOT performed here.
    """

    trace = np.asarray(
        trace,
        dtype=np.float32,
    )

    n = len(trace)

    if n == target_length:

        return trace.astype(
            np.float32
        )

    old_x = np.linspace(
        0,
        1,
        n,
        dtype=np.float32,
    )

    new_x = np.linspace(
        0,
        1,
        target_length,
        dtype=np.float32,
    )

    result = np.interp(
        new_x,
        old_x,
        trace,
    )

    return result.astype(
        np.float32
    )


def process_trace(
    trace,
    target_length=TRACE_LENGTH,
):
    """
    Convert a raw trace into a fixed-length model input.

    Normalization is handled later by dataset.py.
    """

    return downsample_trace(
        trace,
        target_length,
    )


# =========================================================================
# METRIC COMPUTATION
# =========================================================================

def safe_corr(x, y):
    """
    Safely calculate correlation.
    """

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    if (
        len(x) != len(y)
        or len(x) < 2
    ):
        return 0.0

    sx = np.std(x)
    sy = np.std(y)

    if sx < 1e-8 or sy < 1e-8:
        return 0.0

    corr = np.corrcoef(
        x,
        y,
    )[0, 1]

    if not np.isfinite(corr):
        return 0.0

    return float(corr)


def compute_metrics(
    stat,
    fluo,
    neuropil,
):
    """
    Compute scalar features for the metrics MLP.
    """

    fluo = np.asarray(
        fluo,
        dtype=np.float32,
    )

    neuropil = np.asarray(
        neuropil,
        dtype=np.float32,
    )


    # ---------------------------------------------------------------------
    # Basic fluorescence statistics
    # ---------------------------------------------------------------------

    trace_mean = float(
        np.mean(fluo)
    )

    trace_std = float(
        np.std(fluo)
    )

    centered = fluo - trace_mean

    if trace_std > 1e-8:

        trace_skew = float(
            np.mean(centered ** 3)
            / (trace_std ** 3)
        )

    else:

        trace_skew = 0.0


    p95 = float(
        np.percentile(
            fluo,
            95,
        )
    )

    p50 = float(
        np.percentile(
            fluo,
            50,
        )
    )

    p05 = float(
        np.percentile(
            fluo,
            5,
        )
    )


    # ---------------------------------------------------------------------
    # SNR-like measurements
    # ---------------------------------------------------------------------

    denom_95_50 = max(
        abs(p50),
        1e-6,
    )

    snr_p95_p50 = (
        p95 - p50
    ) / denom_95_50


    denom_95_05 = max(
        abs(p05),
        1e-6,
    )

    snr_p95_p05 = (
        p95 - p50
    ) / denom_95_05


    # ---------------------------------------------------------------------
    # Activity
    # ---------------------------------------------------------------------

    trace_max = float(
        np.max(fluo)
    )

    trace_range = float(
        np.max(fluo)
        - np.min(fluo)
    )


    # ---------------------------------------------------------------------
    # Neuropil relationship
    # ---------------------------------------------------------------------

    neuropil_mean = float(
        np.mean(neuropil)
    )

    neuropil_std = float(
        np.std(neuropil)
    )

    fluo_neuropil_corr = safe_corr(
        fluo,
        neuropil,
    )


    # ---------------------------------------------------------------------
    # Morphology metrics
    # ---------------------------------------------------------------------

    metrics = np.array([

        float(stat.get(
            "footprint",
            0.0,
        )),

        float(stat.get(
            "compact",
            0.0,
        )),

        float(stat.get(
            "solidity",
            0.0,
        )),

        float(stat.get(
            "npix",
            0.0,
        )),

        float(stat.get(
            "npix_soma",
            0.0,
        )),

        float(stat.get(
            "radius",
            0.0,
        )),

        float(stat.get(
            "aspect_ratio",
            0.0,
        )),

        float(stat.get(
            "npix_norm_no_crop",
            0.0,
        )),

        float(stat.get(
            "npix_norm",
            0.0,
        )),

        float(stat.get(
            "skew",
            0.0,
        )),

        float(stat.get(
            "std",
            0.0,
        )),

        trace_mean,
        trace_std,
        trace_skew,
        p95,
        p50,
        p05,
        snr_p95_p50,
        snr_p95_p05,
        trace_max,
        trace_range,
        neuropil_mean,
        neuropil_std,
        fluo_neuropil_corr,

    ], dtype=np.float32)


    metrics = np.nan_to_num(
        metrics,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return metrics


# =========================================================================
# DATASET COUNTING
# =========================================================================

def count_labeled_rois(session_paths):

    total = 0

    for session_path in session_paths:

        session = load_session(
            session_path
        )

        labels = session["labels"]

        n_labeled = int(
            np.sum(labels >= 0)
        )

        print(
            f"{os.path.basename(session_path)}: "
            f"{n_labeled} labeled / "
            f"{session['n_rois']} total"
        )

        total += n_labeled

    return total


# =========================================================================
# DATASET BUILDER
# =========================================================================

def build_dataset(
    session_paths,
    output_dir,
    size=64,
    trace_length=TRACE_LENGTH,
):
    """
    Build the complete multimodal ROI QC dataset.
    """

    print("Counting labeled ROIs...")

    n_samples = count_labeled_rois(
        session_paths
    )

    print(
        f"\nFound {n_samples} labeled ROIs."
    )


    # ---------------------------------------------------------------------
    # Preallocate
    # ---------------------------------------------------------------------

    patches = np.empty(
        (
            n_samples,
            2,
            size,
            size,
        ),
        dtype=np.float32,
    )


    # IMPORTANT:
    #
    # signals[:, 0, :] = fluorescence
    # signals[:, 1, :] = neuropil

    signals = np.empty(
        (
            n_samples,
            2,
            trace_length,
        ),
        dtype=np.float32,
    )


    metrics = np.empty(
        (
            n_samples,
            len(METRIC_NAMES),
        ),
        dtype=np.float32,
    )


    labels = np.empty(
        n_samples,
        dtype=np.int8,
    )


    rois = np.empty(
        n_samples,
        dtype=np.int32,
    )


    sessions = np.empty(
        n_samples,
        dtype="<U64",
    )


    idx = 0


    # ---------------------------------------------------------------------
    # Process sessions
    # ---------------------------------------------------------------------

    for session_path in session_paths:

        session_name = os.path.basename(
            session_path
        )

        print(
            f"\nProcessing: {session_name}"
        )


        session = load_session(
            session_path
        )

        mean_img = session["mean_img"]
        masks = session["masks"]
        fluo = session["fluo"]
        neuropil = session["neuropil"]
        stat = session["stat"]
        session_labels = session["labels"]
        n_rois = session["n_rois"]


        # -------------------------------------------------------------
        # Iterate through zero-based ROI indices
        # -------------------------------------------------------------

        for roi_idx in tqdm(
            range(n_rois),
            desc="ROIs",
        ):

            label = int(
                session_labels[roi_idx]
            )

            if label < 0:
                continue


            # ---------------------------------------------------------
            # Spatial patch
            # ---------------------------------------------------------

            mask_id = roi_idx + 1

            patch = make_two_channel_patch(
                mean_img,
                masks,
                mask_id,
                patch_size=size,
            )


            # ---------------------------------------------------------
            # Fluorescence signal
            # ---------------------------------------------------------

            fluo_trace = process_trace(
                fluo[roi_idx],
                target_length=trace_length,
            )


            # ---------------------------------------------------------
            # Neuropil signal
            # ---------------------------------------------------------

            neuropil_trace = process_trace(
                neuropil[roi_idx],
                target_length=trace_length,
            )


            # ---------------------------------------------------------
            # Scalar metrics
            # ---------------------------------------------------------

            roi_metrics = compute_metrics(
                stat=stat[roi_idx],
                fluo=fluo[roi_idx],
                neuropil=neuropil[roi_idx],
            )


            # ---------------------------------------------------------
            # Save sample
            # ---------------------------------------------------------

            patches[idx] = patch

            signals[idx, 0] = fluo_trace
            signals[idx, 1] = neuropil_trace

            metrics[idx] = roi_metrics

            labels[idx] = label

            rois[idx] = roi_idx

            sessions[idx] = session_name

            idx += 1


    # ---------------------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------------------

    if idx != n_samples:

        raise RuntimeError(
            f"Expected {n_samples} samples "
            f"but wrote {idx}."
        )


    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True,
    )


    np.save(
        os.path.join(
            output_dir,
            "patches.npy",
        ),
        patches,
    )


    np.save(
        os.path.join(
            output_dir,
            "signals.npy",
        ),
        signals,
    )


    np.save(
        os.path.join(
            output_dir,
            "metrics.npy",
        ),
        metrics,
    )


    np.save(
        os.path.join(
            output_dir,
            "labels.npy",
        ),
        labels,
    )


    np.save(
        os.path.join(
            output_dir,
            "roi.npy",
        ),
        rois,
    )


    np.save(
        os.path.join(
            output_dir,
            "session.npy",
        ),
        sessions,
    )


    np.save(
        os.path.join(
            output_dir,
            "metric_names.npy",
        ),
        METRIC_NAMES,
    )


    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------

    n_good = int(
        np.sum(labels == 1)
    )

    n_bad = int(
        np.sum(labels == 0)
    )


    print("\nDONE")

    print(
        "Patches:",
        patches.shape,
    )

    print(
        "Signals:",
        signals.shape,
    )

    print(
        "Metrics:",
        metrics.shape,
    )

    print(
        "Labels:",
        labels.shape,
    )

    print(
        "Good:",
        n_good,
    )

    print(
        "Bad:",
        n_bad,
    )


# =========================================================================
# CLI
# =========================================================================

