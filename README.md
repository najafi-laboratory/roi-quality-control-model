# ROI Quality Control Models

This repository provides preprocessing, score-only inference, and fine-tuning code for Purkinje dendrite and PPC/VIS ROI quality-control models. Pretrained weights and imaging data are not included. Inference and fine-tuning require a compatible checkpoint.

## Supported model profiles

| Profile | Expected checkpoint location | Inputs | Default score thresholds |
|---|---|---|---|
| purkinje | weights/purkinje_all.pt | image + fluorescence/neuropil signals + 24 metrics | good >= 0.85; bad <= 0.15 |
| ppc_vis | weights/ppc_vis_signal_metrics.pt | fluorescence/neuropil signals + 24 metrics | good >= 0.85; bad <= 0.15 |

Pretrained checkpoints are not included in this repository. The paths in profiles.json specify where compatible checkpoints should be placed:
- weights/purkinje_all.pt
- weights/ppc_vis_signal_metrics.pt

Alternatively, supply a compatible checkpoint using --checkpoint /path/to/model.pt. Checkpoints must contain the expected model weights, modality configuration, and metric normalization statistics. Arbitrary PyTorch checkpoints are not supported.

Both models predict P(good ROI), not cell identity. Select the appropriate profile explicitly. There is no automatic classification into Purkinje versus PPC/VIS.

## Installation and portability

Use Python 3.10 or later in a suitable environment:

    python -m pip install -r /path/to/modeling_final/requirements.txt

PyTorch CPU inference works; a compatible CUDA-enabled PyTorch installation can use a GPU. No model weights are downloaded at runtime. You must supply a compatible checkpoint before running inference or fine-tuning.

requirements.txt is a clean dependency list rather than a machine-specific Conda export.

The code is portable across machines. Session locations are supplied as command-line arguments. The standalone run.py entry point works regardless of the repository folder name.

## Inference commands

From any directory:

    python /path/to/roi-qc-models/run.py infer \
      --model ppc_vis \
      --session /path/to/session

For Purkinje dendrites:

    python /path/to/roi-qc-models/run.py infer \
      --model purkinje \
      --session /path/to/session


Options:

    --checkpoint /path/to/compatible/best_model.pt
    --output /path/to/output/roi_model_scores.h5
    --batch-size 32
    --device auto|cpu|cuda
    --good-threshold 0.85
    --bad-threshold 0.15
    --overwrite

The default output is <session>/roi_model_scores.h5. Existing score files are not overwritten unless --overwrite is supplied. A custom output directory is useful if the input session is read-only. The file must be placed at the reviewer session root before building the interface.

Inference never opens manual label files to get labels, never updates ROI_label.h5 or roi_manual_labels.npy, and never trains the model. It loads the selected existing model weights and checkpoint normalization, then evaluates all available QC ROIs.

## Required input layout

This initial package supports the same processed-session layout as the trained models:

    session/
      suite2p/
        plane0/
          ops.npy                  # must contain meanImg
          stat.npy                 # original Suite2p ROI rows
      qc_results/
        stat.npy                   # QC ROI rows
        fluo.npy                   # N x T
        neuropil.npy               # N x T
        masks.npy                  # 2-D ROI labels 1..N; zero is background

Each QC row must represent the same ROI across stat, fluo and neuropil. Mask label i+1 belongs to QC row i. Images/masks must share coordinates.

The original suite2p/plane0/stat.npy is used to map QC rows back to original ROI indices by exact ordered xpix/ypix arrays. Missing, ambiguous or repeated matches fail explicitly. Counts alone are never used to infer identity.

The package does not run Suite2p, generate qc_results, process raw movies, or support every external layout accepted by the reviewer. Those adapters can be added separately. The checkpoint profile should only be used on compatible inputs.

## Output consumed by roi-reviewer-demo

The HDF5 score file contains:

- suite2p_roi: original zero-based Suite2p ROI index for each row.
- qc_roi: corresponding zero-based QC row, for audit/debugging.
- probability: P(good ROI), float32.
- state: UTF-8 good, bad or gray.
- attributes: index_space=suite2p_original, model_name, model_source,
  model_path, checkpoint_sha256, target_structure, good_threshold,
  bad_threshold, preprocessing, source session, counts and creation time.

Only QC ROIs are scored. Original ROIs not present in qc_results have no score; they are not silently labeled bad.

For the current roi-reviewer-demo:

1. Put roi_model_scores.h5 at the session root.
2. Build the reviewer using input_layout: suite2p, or auto when it selects the original Suite2p data.
3. The model probability appears under Model Derived Metrics.
4. Leave initialize_labels_from_model_scores: false to open manual labels independently.
   Set it to true only if you intentionally want the reviewer to initialize its in-memory labels from scores.

The demo reads score filenames in this priority order:
roi_model_scores.h5, roi_qc_predictions.h5, roi_predictions.h5.
Using the first filename avoids an older score file taking priority.

IMPORTANT: use original Suite2p layout for these original-index score files. The current demo score loader does not translate original indices for a filtered qc_results reviewer layout. It can display the precomputed scores without importing this model package.

The demo contains some static explanatory text describing the older dendrite model. This does not identify which model generated a file; use the exported model metadata instead.

## Preserved preprocessing

The original compatible checkpoints require the following preprocessing:
- 64 x 64 image patches: z-score mean-image channel, binary-mask channel,
  followed by division of BOTH channels by 255.
- 2048-point resampled fluorescence and neuropil traces.
- Independent robust signal normalization using median and 1.4826 * MAD,
  with standard-deviation and constant-trace fallbacks.
- The original 24 QC metrics in the original column order.
- Metric normalization using metric_mean/metric_std stored in the checkpoint.

The image division by 255 is retained intentionally for compatibility, even though
the preceding patch normalization already produces standardized values. Do not remove it for existing checkpoints without retraining and validating a new preprocessing version.

PPC/VIS inference skips image-patch construction, but still uses the QC/original ROI identity information. Purkinje inference uses every branch.

## Fine-tuning commands

Fine-tuning loads the selected EXISTING checkpoint before the first optimization step. It does not start a new randomly initialized ROI classifier.

Supply at least two labeled sessions:

    python /path/to/modeling_final/run.py finetune \
      --model ppc_vis \
      --sessions /path/to/session_A /path/to/session_B /path/to/session_C

Or supply a JSON file containing a list of session-path strings:

    python /path/to/modeling_final/run.py finetune \
      --model purkinje \
      --sessions-file /path/to/sessions.json \
      --epochs 25 --batch-size 32 --lr 0.0001

Relative paths in the JSON file resolve relative to that JSON file.

Use --checkpoint to initialize from a compatible previously fine-tuned checkpoint.
The profile must still match the checkpoint architecture and feature count.

### Supported labels

Inference needs no labels. Fine-tuning reads, but never changes:

1. <session>/ROI_label.h5, when present, takes priority.
   good_roi and bad_roi are zero-based QC indices by default.
   An explicit index_space=suite2p_original attribute is also supported.
   Conflicting, duplicate, noninteger and out-of-range indices are errors.
2. Otherwise <session>/suite2p/plane0/roi_manual_labels.npy is read.
   It must have one row per ORIGINAL Suite2p ROI:
   0=bad, 1=good, 2=unsure, NaN=unlabeled.
   Labels are mapped exactly to QC rows.

Unsure/unlabeled ROIs are excluded from training. Use finalized independent human
labels rather than treating model-generated labels as ground truth.

### Training behavior and settings

- Default: 25 epochs, batch size 32, learning rate 1e-4, AdamW weight decay 1e-4.
  The fine-tuning learning rate is lower than the original from-scratch run.
- Session-aware 20% validation split, seed 42.
- At least two sessions must contribute labels, and both classes must exist in both splits.
- Checkpoint metric normalization is retained; no normalization is fitted on validation/test data.
- Training augmentation matches the original dataset implementation; validation has no augmentation.
- Image backbone: the original freezing convention trains the adapted first convolution and
  projection while freezing other ResNet parameters. Frozen-backbone running statistics remain fixed.
  --unfreeze-backbone makes the full image backbone trainable.
- For signal+metrics there is no image backbone.
- Best epoch: highest validation ROC-AUC; lower validation loss breaks ties.
- Early stopping patience 5, configurable.
- Architecture/profile and all preprocessing stay fixed.

Other options: --weight-decay, --patience, --val-ratio, --seed, --device, --output.
New output directories are required; existing training runs are never overwritten.

### Fine-tuning outputs

By default, outputs are saved under runs/<profile>_finetune_<timestamp>/ inside the repository folder.

- dataset/: generated training arrays; no input files are modified.
- run_config.json and split.json: source model, settings and exact split.
- history.json/history.csv: training and validation loss, accuracy, precision,
  recall, F1, ROC-AUC and confusion matrices saved after every epoch.
- summary.json: metrics at the selected best epoch.
- best_model.pt: complete reusable best checkpoint.
- last.pt: final epoch checkpoint plus optimizer/early-stopping state.
- best_validation_predictions.npz: selected validation predictions.

New checkpoints retain normalization, modality flags and feature schema, so inference
can use them directly with --checkpoint. Base weights under weights/ are never modified.
This package does not automatically resume an interrupted optimizer; --checkpoint starts
a new fine-tuning run from the supplied learned weights.

Training metrics are measured on augmented batches during training.
Validation is used for selection and is not an independent final test.

## Files and provenance

- cnn.py: copied network definitions from the source model/cnn.py; unused legacy
  ROIModel wrapper removed. Layer definitions are preserved.
- data/crop.py: copied patch extraction, without its standalone self-test.
- data/builder.py: copied feature/dataset builder, with package imports, no hard-coded
  session list, and strict read-only label loading.
- data/dataset.py: copied original dataset/normalization/augmentation implementation.
- data/alignment.py: exact shared ROI mapping.
- checkpoints.py: profile-aware checkpoint loading.
- sessions.py: label-free session features.
- inference.py: score-only batched inference and original-index HDF5 export.
- finetune.py: checkpoint-initialized single-model trainer.
- run.py: standalone command dispatch.
- profiles.json: model registry and fixed preprocessing.
- weights/: user-provided compatible checkpoints; excluded from version control.

The parent project's code is not imported. There is no ablation grid, hyperparameter search,
calcium-transient validation, reviewer notebook, raw data, or legacy prediction output in this
package. Those are separate workflows.


## Integration with roi-reviewer-demo

This repository can generate roi_model_scores.h5 independently. Place the exported file at the reviewer session root and follow the configuration instructions above. The reviewer does not need this package or its checkpoints to display exported scores.

Optionally, copy this repository's source files into roi-reviewer-demo/model/.
Then python model/run.py infer ... works from the reviewer repository root.
Compatible checkpoints must be supplied separately.