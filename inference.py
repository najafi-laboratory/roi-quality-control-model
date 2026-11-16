"""Export model scores with original Suite2p ROI IDs; never modify labels."""
import argparse
import datetime
import json
import os
from pathlib import Path
import tempfile
import h5py
import numpy as np
import torch
from .checkpoints import load_model, profiles, sha256
from .sessions import Session

def predict(session, model_name, checkpoint=None, output=None, batch_size=32,
            device="auto", good_threshold=None, bad_threshold=None, overwrite=False):
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    model, state, profile, checkpoint_path, device = load_model(model_name, checkpoint, device)
    good = profile["good_threshold"] if good_threshold is None else good_threshold
    bad = profile["bad_threshold"] if bad_threshold is None else bad_threshold
    if not 0 <= bad < good <= 1:
        raise ValueError("Require 0 <= bad threshold < good threshold <= 1.")
    session = Session(session)
    output = Path(output).expanduser().resolve() if output else session.path / "roi_model_scores.h5"
    if output.name not in {"roi_model_scores.h5", "roi_predictions.h5", "roi_qc_predictions.h5"}:
        raise ValueError("Use a reviewer score filename, not a label filename.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to replace this score file.")
    if output.exists() and any(output.samefile(session.path / name)
                               for name in ["ROI_label.h5", "roi_manual_labels.npy"]
                               if (session.path / name).exists()):
        raise ValueError("Refusing to replace a label file.")
    all_probabilities = []
    with torch.inference_mode():
        for start in range(0, len(session.stat), batch_size):
            rows = [session.features(i, state, profile)
                    for i in range(start, min(start + batch_size, len(session.stat)))]
            args = {
                "signal": torch.from_numpy(np.stack([row[1] for row in rows])).to(device),
                "metrics": torch.from_numpy(np.stack([row[2] for row in rows])).to(device),
            }
            if profile["config"]["use_image"]:
                args["image"] = torch.from_numpy(np.stack([row[0] for row in rows])).to(device)
            probabilities = torch.sigmoid(model(**args).squeeze(1)).cpu().numpy()
            all_probabilities.append(probabilities)
    probabilities = np.concatenate(all_probabilities)
    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned nonfinite probabilities.")
    states = np.where(probabilities >= good, "good",
                      np.where(probabilities <= bad, "bad", "gray"))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".roi-scores-", suffix=".h5", dir=output.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as handle:
            handle["suite2p_roi"] = session.original_indices
            handle["qc_roi"] = np.arange(len(probabilities), dtype=np.int64)
            handle["probability"] = probabilities.astype(np.float32)
            handle.create_dataset("state", data=states.astype(object),
                                  dtype=h5py.string_dtype("utf-8"))
            handle.attrs.update({
                "index_space": "suite2p_original", "model_name": model_name,
                "model_source": "portable_roi_qc", "model_path": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "target_structure": profile["target_structure"],
                "good_threshold": good, "bad_threshold": bad,
                "preprocessing": profile["preprocessing"],
                "session_path": str(session.path),
                "original_roi_count": session.original_roi_count,
                "scored_roi_count": len(probabilities),
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            handle.flush()
        # No-clobber publication unless overwriting was explicitly requested.
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    summary = {"output": str(output), "model": model_name, "scored": len(probabilities),
               "original_rois": session.original_roi_count,
               "good": int(np.sum(states == "good")), "bad": int(np.sum(states == "bad")),
               "gray": int(np.sum(states == "gray"))}
    print(json.dumps(summary, indent=2))
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--model", dest="model_name", required=True, choices=list(profiles()))
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--good-threshold", type=float)
    parser.add_argument("--bad-threshold", type=float)
    parser.add_argument("--overwrite", action="store_true")
    predict(**vars(parser.parse_args()))

if __name__ == "__main__":
    main()
