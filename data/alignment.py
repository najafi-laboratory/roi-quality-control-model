"""Exact, unambiguous QC-to-original Suite2p ROI mapping."""
import numpy as np

def pixel_key(roi):
    y, x = np.asarray(roi["ypix"]), np.asarray(roi["xpix"])
    if (y.ndim != 1 or x.shape != y.shape or y.size == 0
            or not np.isfinite(y).all() or not np.isfinite(x).all()):
        raise ValueError("Invalid ROI pixel coordinates.")
    return tuple(y), tuple(x)

def map_qc_to_original(qc_stat, original_stat):
    lookup = {}
    for index, roi in enumerate(original_stat):
        lookup.setdefault(pixel_key(roi), []).append(index)
    mapping = []
    for index, roi in enumerate(qc_stat):
        matches = lookup.get(pixel_key(roi), [])
        if len(matches) != 1:
            raise ValueError(f"QC ROI {index} has {len(matches)} exact original matches.")
        mapping.append(matches[0])
    mapping = np.asarray(mapping, dtype=np.int64)
    if len(np.unique(mapping)) != len(mapping):
        raise ValueError("Multiple QC ROIs map to the same original ROI.")
    return mapping
