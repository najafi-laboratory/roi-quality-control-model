"""Read QC features without reading or changing any label file."""
from pathlib import Path
import numpy as np
from .data.crop import make_two_channel_patch
from .data.builder import compute_metrics, downsample_trace
from .data.dataset import ROIQCDataset
from .data.alignment import map_qc_to_original

class Session:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        qc = self.path / "qc_results"
        plane = self.path / "suite2p" / "plane0"
        self.stat = np.load(qc / "stat.npy", allow_pickle=True)
        original = np.load(plane / "stat.npy", allow_pickle=True)
        self.original_roi_count = len(original)
        self.original_indices = map_qc_to_original(self.stat, original)
        self.fluo = np.load(qc / "fluo.npy", mmap_mode="r")
        self.neuropil = np.load(qc / "neuropil.npy", mmap_mode="r")
        self.masks = np.load(qc / "masks.npy", mmap_mode="r")
        self.mean_img = np.load(plane / "ops.npy", allow_pickle=True).item()["meanImg"]
        n = len(self.stat)
        if (not n or self.fluo.ndim != 2 or self.fluo.shape[0] != n
                or self.neuropil.shape != self.fluo.shape or self.fluo.shape[1] < 2):
            raise ValueError("QC fluorescence, neuropil and stat rows do not align.")
        if self.masks.shape != self.mean_img.shape or self.masks.ndim != 2:
            raise ValueError("QC masks and mean image shapes do not align.")
        ids = np.unique(self.masks)
        if not np.array_equal(ids[ids > 0], np.arange(1, n + 1)):
            raise ValueError("QC masks must use IDs 1..N, aligned with QC rows.")

    def features(self, index, state, profile):
        f = np.asarray(self.fluo[index], dtype=np.float32)
        n = np.asarray(self.neuropil[index], dtype=np.float32)
        if not np.isfinite(f).all() or not np.isfinite(n).all():
            raise ValueError(f"QC ROI {index}: nonfinite fluorescence/neuropil.")
        signal = np.stack([downsample_trace(f, profile["trace_length"]),
                           downsample_trace(n, profile["trace_length"])])
        signal = ROIQCDataset._normalize_signal(signal)
        metrics = compute_metrics(self.stat[index], f, n)
        metrics = (metrics - state["metric_mean"]) / state["metric_std"]
        image = None
        if profile["config"]["use_image"]:
            image = make_two_channel_patch(self.mean_img, self.masks, index + 1,
                                           patch_size=profile["patch_size"]) / 255.0
            if not np.isfinite(image).all():
                raise ValueError(f"QC ROI {index}: nonfinite image patch.")
        return image, signal.astype(np.float32), metrics.astype(np.float32)
