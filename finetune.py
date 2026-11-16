"""Fine-tune a selected existing ROI-QC checkpoint on labeled sessions."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from .checkpoints import PACKAGE_ROOT, load_model, profiles, sha256
from .data.builder import build_dataset, METRIC_NAMES
from .data.dataset import ROIQCDataset, split_by_session

def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)

def metrics(y, p, threshold):
    prediction = p >= threshold
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }

def pass_epoch(model, loader, device, threshold, config, optimizer=None, freeze_backbone=True):
    training = optimizer is not None
    model.train(training)
    if training and config["use_image"] and freeze_backbone:
        # Keep frozen ResNet normalization buffers fixed while still learning
        # its adapted first convolution and trainable projection.
        model.image_encoder.backbone.eval()
    total_loss, ys, ps = 0.0, [], []
    with torch.set_grad_enabled(training):
        for image, signal, scalar, labels in loader:
            labels = labels.to(device)
            inputs = {}
            if config["use_image"]:
                inputs["image"] = image.to(device)
            if config["use_signal"]:
                inputs["signal"] = signal.to(device)
            if config["use_metrics"]:
                inputs["metrics"] = scalar.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(**inputs).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training/validation loss.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            ys.append(labels.detach().cpu().numpy())
            ps.append(torch.sigmoid(logits.detach()).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    result = metrics(y, p, threshold)
    result["loss"] = total_loss / len(y)
    return result, y, p

def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid hyperparameters.")
    if not 0 < args.val_ratio < 1:
        raise ValueError("Validation ratio must lie strictly between zero and one.")
    if args.sessions_file:
        manifest = Path(args.sessions_file).expanduser().resolve()
        paths = json.loads(manifest.read_text())
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError("Sessions JSON must contain a list of path strings.")
        sessions = [str((manifest.parent / p).resolve()) if not Path(p).is_absolute() else str(Path(p).resolve()) for p in paths]
    else:
        sessions = [str(Path(p).expanduser().resolve()) for p in args.sessions]
    if len(sessions) < 2 or len(sessions) != len(set(sessions)):
        raise ValueError("Use at least two distinct labeled session directories.")
    if len({Path(p).name for p in sessions}) != len(sessions):
        raise ValueError("Session basenames must be unique for session-aware splitting.")
    for path in sessions:
        if not Path(path).is_dir():
            raise FileNotFoundError(path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, state, profile, initial_path, device = load_model(
        args.model, args.checkpoint, args.device,
        freeze_backbone=not args.unfreeze_backbone,
    )
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output).expanduser().resolve() if args.output else PACKAGE_ROOT / "runs" / (args.model + "_finetune_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    data_dir = output / "dataset"
    run_config = {
        "model": args.model, "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": sha256(initial_path),
        "data_sessions": sessions,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "patience": args.patience,
        "val_ratio": args.val_ratio, "seed": args.seed,
        "metric_threshold": profile["good_threshold"],
        "normalization": "retained from initial trained checkpoint",
        "freeze_backbone": not args.unfreeze_backbone,
        "preprocessing": profile["preprocessing"],
        "torch_version": str(torch.__version__), "device": str(device),
        "training_metrics": "online metrics from augmented training batches",
    }
    save_json(output / "run_config.json", run_config)
    print("Fine-tuning existing weights:", initial_path, flush=True)
    print("Output:", output, flush=True)
    build_dataset(sessions, str(data_dir), size=profile["patch_size"], trace_length=profile["trace_length"])
    session_ids = np.load(data_dir / "session.npy")
    if len(np.unique(session_ids)) < 2:
        raise ValueError("At least two sessions must contribute labeled examples.")
    train_idx, val_idx = split_by_session(session_ids, val_ratio=args.val_ratio, seed=args.seed)
    labels = np.load(data_dir / "labels.npy")
    for name, indices in [("train", train_idx), ("validation", val_idx)]:
        if not len(indices) or len(np.unique(labels[indices])) != 2:
            raise ValueError(f"The {name} split needs both good and bad labels; choose more sessions or another split seed.")
    ds_args = dict(
        patches_path=str(data_dir / "patches.npy"), signals_path=str(data_dir / "signals.npy"),
        metrics_path=str(data_dir / "metrics.npy"), labels_path=str(data_dir / "labels.npy"),
        roi_path=str(data_dir / "roi.npy"), session_path=str(data_dir / "session.npy"),
        metric_mean=state["metric_mean"], metric_std=state["metric_std"],
    )
    train_ds = ROIQCDataset(**ds_args, indices=train_idx, augment=True, signal_augment=True)
    val_ds = ROIQCDataset(**ds_args, indices=val_idx, augment=False, signal_augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    split = {
        "train_sessions": sorted(np.unique(session_ids[train_idx]).tolist()),
        "val_sessions": sorted(np.unique(session_ids[val_idx]).tolist()),
        "train_indices": train_idx.tolist(), "val_indices": val_idx.tolist(),
    }
    save_json(output / "split.json", split)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    best_auc, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        training, _, _ = pass_epoch(model, train_loader, device, profile["good_threshold"],
                                   state["config"], optimizer, not args.unfreeze_backbone)
        validation, y, p = pass_epoch(model, val_loader, device, profile["good_threshold"], state["config"])
        row = {"epoch": epoch}
        for prefix, values in [("train", training), ("val", validation)]:
            for key, value in values.items():
                row[prefix + "_" + key] = value
        history.append(row)
        save_json(output / "history.json", history)
        temp = output / "history.csv.tmp"
        with temp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(output / "history.csv")
        improved = validation["roc_auc"] > best_auc or (
            validation["roc_auc"] == best_auc and validation["loss"] < best_loss)
        if improved:
            best_auc, best_loss, best_epoch, stale = validation["roc_auc"], validation["loss"], epoch, 0
        else:
            stale += 1
        saved = {
            "model_state": model.state_dict(),
            "config": state["config"], "num_metrics": state["num_metrics"],
            "metric_mean": state["metric_mean"], "metric_std": state["metric_std"],
            "metric_names": METRIC_NAMES.tolist(),
            "experiment": state.get("experiment", args.model),
            "best_auc": best_auc, "best_epoch": best_epoch, "val_loss": best_loss,
            "epoch": epoch, "run_config": run_config,
            "preprocessing": profile["preprocessing"],
        }
        if improved:
            temporary = output / "best_model.pt.tmp"
            torch.save(saved, temporary)
            temporary.replace(output / "best_model.pt")
            np.savez(output / "best_validation_predictions.npz", labels=y, probabilities=p,
                     session=session_ids[val_idx], dataset_indices=val_idx)
            save_json(output / "summary.json", {"best_epoch": best_epoch, "best_auc": best_auc,
                                               "best_val_loss": best_loss, "best_metrics": row})
        temporary = output / "last.pt.tmp"
        torch.save({**saved, "optimizer_state": optimizer.state_dict(), "patience_counter": stale}, temporary)
        temporary.replace(output / "last.pt")
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            print("Early stopping.", flush=True)
            break
    print(f"Best validation checkpoint: {output / 'best_model.pt'}", flush=True)
    print("Validation was used for selection; use separate test sessions for final assessment.")
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(profiles()))
    parser.add_argument("--checkpoint", help="Optional compatible trained checkpoint to initialize from.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sessions", nargs="+")
    group.add_argument("--sessions-file", help="JSON list of labeled session paths.")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Fine-tuning default is lower than the original training rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--unfreeze-backbone", action="store_true",
                        help="Train the full image backbone; default retains adapted-convolution/projection training.")
    run(parser.parse_args())

if __name__ == "__main__":
    main()
