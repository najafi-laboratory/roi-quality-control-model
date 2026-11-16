"""Load profile-selected weights and the exact saved normalization."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .cnn import ROICNN

PACKAGE_ROOT = Path(__file__).resolve().parent

def profiles():
    return json.loads((PACKAGE_ROOT / "profiles.json").read_text())

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_model(name, checkpoint=None, device="auto", freeze_backbone=True):
    profile = profiles()[name]
    path = Path(checkpoint).expanduser().resolve() if checkpoint else PACKAGE_ROOT / profile["weights"]
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state", "config", "num_metrics", "metric_mean", "metric_std"}
    if not required.issubset(state):
        raise ValueError(f"Checkpoint lacks {sorted(required - set(state))}.")
    if state["config"] != profile["config"] or state["num_metrics"] != profile["num_metrics"]:
        raise ValueError("Checkpoint architecture does not match the selected model profile.")
    for key in ["metric_mean", "metric_std"]:
        array = np.asarray(state[key], dtype=np.float32)
        if array.shape != (profile["num_metrics"],) or not np.isfinite(array).all():
            raise ValueError(f"Invalid {key} in checkpoint.")
        state[key] = array
    if np.any(state["metric_std"] <= 0):
        raise ValueError("Checkpoint metric standard deviations must be positive.")
    if state.get("preprocessing", profile["preprocessing"]) != profile["preprocessing"]:
        raise ValueError("Checkpoint preprocessing differs from this profile.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ROICNN(num_metrics=state["num_metrics"], pretrained=False,
                   freeze_backbone=freeze_backbone, **state["config"]).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    return model, state, profile, path, torch.device(device)
