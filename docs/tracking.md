# Molmo2 Tracking: Training & Testing Details

This document covers all aspects of Molmo2's video tracking pipeline — datasets, data formats, training setup, evaluation benchmarks, and metrics.

## Table of Contents

- [Overview](#overview)
- [Task Types](#task-types)
- [Output Format](#output-format)
- [Datasets](#datasets)
  - [Academic Tracking Datasets (Training & Evaluation)](#academic-tracking-datasets-training--evaluation)
  - [Molmo2 Custom Tracking Datasets](#molmo2-custom-tracking-datasets)
- [Data Pipeline](#data-pipeline)
  - [Frame–Point Alignment](#framepoint-alignment)
  - [FPS Subsampling](#fps-subsampling)
- [Training](#training)
  - [Training Commands](#training-commands)
  - [Key Configuration Parameters](#key-configuration-parameters)
- [Evaluation](#evaluation)
  - [Evaluation Benchmarks](#evaluation-benchmarks)
  - [Running Evaluation](#running-evaluation)
  - [Object Tracking Metrics (F1, HOTA)](#object-tracking-metrics-f1-hota)
  - [Point Tracking Metrics (TAP-Vid)](#point-tracking-metrics-tap-vid)
- [Prediction Parsing](#prediction-parsing)
  - [Supported Prediction Formats](#supported-prediction-formats)
- [Frame–Point Consistency Guarantee](#framepoint-consistency-guarantee)
- [Code Map](#code-map)

---

## Overview

Molmo2 supports video point tracking tasks (e.g., MeViS, DAVIS, LaSOT, TAP-Vid). The model receives a video and a natural language query (and optionally initial query points), then predicts the spatial location and occlusion state of the target objects at every sampled frame. The tracking capability is trained as part of the SFT (Supervised Fine-Tuning) stage, using a multi-task mixture that includes tracking, grounding, pointing, and video QA data.

---

## Task Types

Three tracking task types are supported. They differ in what the model is asked to predict:

| Task | Description | Datasets |
|------|-------------|---------|
| `track` | Predict per-frame [x, y] coordinates (and optional occlusion flag) for one or more objects across all video frames. | MeViS, Ref-YT-VOS, Ref-DAVIS17, BURST, LV-VIS, ViCAS, ReVOS, YT-VIS, MoCA, Molmo2VideoTrack |
| `ground` | Predict start and end point locations for each object (not every frame). | MeViS, BURST, LV-VIS, ViCAS, ReVOS, MoCA |
| `single_point_track` | Track a single object given an initial query point. | MeViS, BURST, LV-VIS, ViCAS, LaSOT, WebUAV, TrackingNet, ... |

---

## Output Format

The model outputs structured text. The primary format used for the `track` task is `video_point_track_per_frame` (with optional occlusion):

```
time 0.50
{0: [52.4, 40.7], 1: [52.4, 41.5]}
time 1.00
{0: [52.4, 40.7, yes], 1: [52.4, 41.5], 2: [52.8, 40.5, yes]}
```

- Each key is an **object ID** (integer).
- Each value is `[x, y]` for a **visible** object, or `[x, y, yes]` for an **occluded** object.
- Coordinates are in the range **[0, 100]** (normalized), scaled to pixel space during evaluation.
- Each `time X.XX` block corresponds to one extracted video frame.

Additional supported formats:

**`video_point_ground_start_end`** — start and end positions only:
```
0: ([34.0, 63.0, 0.50], [35.0, 64.0, 1.00])
1: ([72.0, 49.0, 0.50], [73.0, 50.0, 1.00])
```

**`single_point_track_per_frame`** — single-object, coordinates and timestamp per entry:
```
[34.0, 63.0, 0.50], [35.0, 64.0, 1.00], [36.0, 65.0, 1.50]
```

---

## Datasets

### Academic Tracking Datasets (Training & Evaluation)

All academic datasets are stored under `VIDEO_TRACK_DATA_HOME` and their annotations are hosted on HuggingFace (`allenai/molmo2-*`). Annotations are auto-downloaded; videos typically need to be downloaded separately (see download instructions in each class).

| Dataset Class | HF Source | Tasks | Default FPS | Splits |
|---------------|-----------|-------|-------------|--------|
| `Mevis` | `allenai/molmo2-mevis` | track, ground, single_point_track | 6 | train, valid, valid_u |
| `RefYoutubeVOS` | `allenai/molmo2-ref-yt-vos` | track | 6 | train, valid, test |
| `RefDavis17` | `allenai/molmo2-ref-davis17` | track | 6 | train, valid, test |
| `ReasonVOS` | `allenai/molmo2-reasonvos` | track | 6 | train, test |
| `Burst` | `allenai/molmo2-burst` | track, ground, single_point_track | 6 | train, valid |
| `LVVIS` | `allenai/molmo2-lv-vis` | track, ground, single_point_track | 4 | train, valid |
| `ViCAS` | `allenai/molmo2-vicas` | track, ground, single_point_track | 6 | train, valid |
| `YTVIS` | (yt-vis) | track | 6 | train, valid |
| `MoCA` | `allenai/molmo2-moca` | track, ground | 6 | train, test |
| `SingleObjectTrack` | `allenai/molmo2-single-object-track` | single_point_track | varies | train/test per config |

`SingleObjectTrack` includes: LaSOT, WebUAV, TrackingNet, and more (one HF config per sub-dataset).

**Download command (bulk):**
```bash
python3 scripts/download_datasets.py video_tracking --n-procs 8
```

Some datasets require manual download due to licensing (e.g., Ref-YT-VOS, YT-VIS, LaSOT). See `olmo/data/academic_video_track_datasets.py` for instructions.

### Molmo2 Custom Tracking Datasets

Molmo2 also includes a large proprietary tracking dataset (`Molmo2VideoTrack`) compiled from 16+ sources:

| Source Class | Description |
|--------------|-------------|
| `MOSESource` | MOSE dataset |
| `MOSEv2Source` | MOSEv2 dataset |
| `SAVSource` | Segment Anything Video (SAV) |
| `VIPSegSource` | VIPSeg dataset |
| `AnimalTrackSource` | AnimalTrack |
| `APTv2Source` | APTv2 dataset |
| `BFTSource` | BFT dataset |
| `SoccerNetSource` | SoccerNet |
| `SportsMOTSource` | SportsMOT |
| `TeamTrackSource` | TeamTrack |
| `MOT20Source` | MOT20 |
| `PersonPath22Source` | PersonPath22 |
| `DanceTrackSource` | DanceTrack |
| `BDD100KSource` | BDD100K tracking |
| `UAVDTSource` | UAVDT (drone) |
| `SeaDronesSeeSource` | SeaDronesSee (maritime) |

| Dataset | HF Source | FPS | Splits |
|---------|-----------|-----|--------|
| `Molmo2VideoTrack` | `allenai/Molmo2-VideoTrack` | varies | train |
| `Molmo2VideoTrackEval` | `allenai/Molmo2-VideoTrackEval` | varies | test |
| `Molmo2VideoTrackInstruction` | `allenai/molmo2-track-instruction` | varies | train |

---

## Data Pipeline

### Frame–Point Alignment

A critical design requirement is that the number of annotation point-frames must always equal the number of video frames the model actually sees. This is enforced by `_filter_frames_to_video()` in `olmo/preprocessing/data_formatter.py`.

After a video is decoded, the extractor produces a `VideoFrames` object with per-frame timestamps. The formatter then filters annotation frames to keep only those whose timestamps are within `eps = 0.01` seconds of an actual extracted frame:

```python
diff_matrix = np.abs(frame_times[:, None] - video_timestamps)  # [n_annot_frames, n_video_frames]
min_diffs = np.min(diff_matrix, axis=1)
filtered_frames = [f for i, f in enumerate(frames_data) if min_diffs[i] < eps]
```

This guarantees a **one-to-one correspondence** between annotation point-frames and extracted video frames.

### FPS Subsampling

Datasets specify a `sampling_fps` field (e.g., 1 or 2 FPS). When set, `_sample_at_fps()` generates a regular timestamp grid at the target rate and calls `_filter_frames_to_video()` to retain only annotation frames aligned to those grid timestamps. The video decoder extracts frames at the matching rate:

- The model sees exactly `T = duration × sampling_fps` frames.
- The annotation sequence contains exactly `T` point-frames (one per extracted frame).

**Constraint**: `sampling_fps` must evenly divide the video's native FPS. Examples violating this constraint are filtered out at dataset load time.

The maximum video FPS used is capped at `MAX_VIDEO_FPS = 10`.

---

## Training

Tracking data is used in the **SFT stage** (not pre-training). It is mixed with other task data (QA, pointing, video QA, etc.) via `launch_scripts/sft.py`.

### Training Commands

**Debug run:**
```bash
torchrun --nproc-per-node=1 launch_scripts/sft.py /path/to/pretrained/model debug \
  --debug --save_folder=dbg --save_overwrite
```

**Standard SFT (8 GPUs, Qwen3-4B):**
```bash
WANDB_API_KEY=key torchrun --nproc-per-node=8 launch_scripts/sft.py \
  /path/to/pretrained/model molmo2 \
  --wandb.name=run_name --wandb.entity=entity --wandb.project=project \
  --save_folder=/path/to/save/folder
```

**Long-Context SFT (384 frames, B200s):**
```bash
torchrun --nproc-per-node=8 launch_scripts/sft.py /path/to/sft/checkpoint molmo2 \
  --max_duration=2000 --device_batch_size=1 --data.num_workers=4 \
  --seq_len=36864 \
  --model.mm_preprocessor.video.max_frames=384 \
  --model.llm.max_sequence_length=36864
```

**With Context Parallelism (CP degree 2 on 8 GPUs):**
```bash
torchrun --nproc-per-node=8 launch_scripts/sft.py /path/to/checkpoint molmo2 \
  --cp_degree=2 \
  --parallelism.context_parallel_config.degree=2 \
  --parallelism.context_parallel_config.attention_type=ulysses \
  --save_folder=/path/to/save/folder
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `global_batch_size` | 128 | Total batch size across all devices |
| `device_batch_size` | 1 (tracking) | Per-GPU batch size (tracking is memory-intensive) |
| `seq_len` | 2536 | Max sequence length (increase for long-context SFT) |
| `model.mm_preprocessor.video.max_frames` | 16–64 | Max video frames per example |
| `learning_rate` | varies | Initial learning rate |
| `weight_decay` | 0.1 | L2 regularization |
| `gradient_clipping` | 1.0 | Max gradient norm |
| `eval_interval` | 1000 | Steps between in-loop evaluations |
| `ft_llm` | True | Fine-tune LLM parameters |
| `ft_vision` | True | Fine-tune vision parameters |

**Distributed training options:**
- `data_parallel_replicate_degree`: DDP/HSDP weight replication degree
- `data_parallel_shard_degree`: FSDP/HSDP sharding degree (-1 = auto)
- `parallelism.context_parallel_config.degree`: Context parallel degree
- `parallelism.context_parallel_config.attention_type`: `ulysses` (default) or `ring`

---

## Evaluation

### Evaluation Benchmarks

The `tracking` task group includes five benchmarks, all evaluated at 1 FPS:

| Benchmark (task name) | Dataset | Split | Description |
|-----------------------|---------|-------|-------------|
| `mevis_track_eval_1fps:test` | MeViS | test | Motion-guided expression video segmentation |
| `ref_yt_vos_track_eval_1fps:test` | Ref-YT-VOS | test | Referring YouTube-VOS |
| `ref_davis17_track_eval_1fps:test` | Ref-DAVIS17 | test | Referring DAVIS 2017 |
| `reasonvos_track_eval_1fps:test` | ReasonVOS | test | Reasoning-guided VOS |
| `molmo2_video_track_eval_1fps:test` | Molmo2VideoTrackEval | test | Molmo2 custom track eval |

### Running Evaluation

**Single task:**
```bash
torchrun --nproc-per-node 8 launch_scripts/eval.py Molmo2-4B \
  --task=mevis_track_eval_1fps:test --save_to_checkpoint_dir
```

**All tracking benchmarks:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_TIMEOUT_MINUTES=20 \
  torchrun --nproc-per-node 8 launch_scripts/eval_molmo2.py Molmo2-4B \
  --tasks=tracking --save_to_checkpoint_dir --num_workers=4
```

**Notes:**
- Device batch size for tracking tasks is **1** (memory intensive).
- `NCCL_TIMEOUT_MINUTES=20` is recommended to prevent timeout on long tracking videos.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces OOM risk.
- Results (metrics + predictions) are cached in the checkpoint directory. Re-runs skip recomputation automatically.

### Object Tracking Metrics (F1, HOTA)

Used by the `VideoObjectTrackingEval` evaluator (`olmo/eval/evaluators.py`). Predictions are compared against segmentation masks using Hungarian matching.

| Metric | Description |
|--------|-------------|
| **Precision** | Fraction of predicted points falling inside the GT mask |
| **Recall** | Fraction of GT objects with a matching prediction inside their mask |
| **F1** | Harmonic mean of Precision and Recall |
| **HOTA** | Higher-Order Tracking Accuracy (balances detection and association) |
| **DetA** | Detection Accuracy component of HOTA |
| **AssA** | Association Accuracy component of HOTA |

Point matching uses the **Hungarian algorithm** (optimal bipartite matching) based on pairwise Euclidean distances. A prediction is "correct" if the matched predicted point falls within the corresponding GT segmentation mask.

Per-category metrics are also computed and logged to W&B.

### Point Tracking Metrics (TAP-Vid)

Used by `compute_tapvid_metrics()` in `olmo/eval/point_tracking_utils.py`. These are the standard TAP-Vid (Tracking Any Point in Video) metrics.

| Metric | Description |
|--------|-------------|
| **Occlusion Accuracy (OA)** | Accuracy at predicting whether a point is occluded |
| **pts_within_1** | Fraction of visible GT points with prediction within 1 pixel |
| **pts_within_2** | Fraction of visible GT points with prediction within 2 pixels |
| **pts_within_4** | Fraction of visible GT points with prediction within 4 pixels |
| **pts_within_8** | Fraction of visible GT points with prediction within 8 pixels |
| **pts_within_16** | Fraction of visible GT points with prediction within 16 pixels |
| **jaccard_1..16** | Jaccard metric (TP / (TP + FP + FN)) at each pixel threshold |
| **average_pts_within_thresh (< δ_avg)** | Mean of pts_within_1..16 |
| **average_jaccard (AJ)** | Mean of jaccard_1..16 |

Query modes:
- `first`: Only evaluate frames **after** the query frame (the model is given the initial location).
- `strided`: Evaluate all frames **except** the query frame.

---

## Prediction Parsing

After the model generates text, predictions are parsed into structured arrays for metric computation.

### Supported Prediction Formats

**`video_point_track_per_frame`** (primary tracking format, no occlusion):

Handled by `extract_video_point_track_per_frame()` in `olmo/eval/object_tracking_utils.py`.

```
time 0.50
{0: [34.0, 63.0], 1: [72.0, 49.0]}
time 1.00
{0: [35.0, 64.0]}
```

**`video_point_track_all_frames_with_occlusion`** (TAP-Vid, with occlusion flag):

Handled by `extract_video_point_track_per_frame_with_occlusion()` in `olmo/eval/point_tracking_utils.py`.

```
time 0.50
{0: [52.4, 40.7], 1: [52.4, 41.5]}
time 1.00
{0: [52.4, 40.7, yes], 1: [52.4, 41.5], 2: [52.8, 40.5, yes]}
```

- `yes` / `true` / `1` → occluded; absence or `no` → visible.

**`video_point_ground_start_end`** (grounding, start/end only):

Handled by `extract_video_points_start_end()` in `olmo/eval/object_tracking_utils.py`.

```
0: ([34.0, 63.0, 0.50], [35.0, 64.0, 1.00])
1: ([72.0, 49.0, 0.50], [73.0, 50.0, 1.00])
```

**`single_point_track_per_frame`** (single object, compact format):

Handled by `extract_single_point_track_per_frame()` in `olmo/eval/object_tracking_utils.py`.

```
[34.0, 63.0, 0.50], [35.0, 64.0, 1.00], [36.0, 65.0, 1.50]
```

**Coordinate convention:** All coordinates are normalized to [0, 100]. They are converted to pixel coordinates using `get_absolute_coordinates([x, y], width, height)` during parsing.

**Format auto-detection:** If no format is specified, `ObjectTrackingParser.detect_format()` uses heuristic regex matching to identify the format automatically.

---

## Frame–Point Consistency Guarantee

Both during training and testing, the number of annotation point-frames always equals the number of video frames the model sees. The table below summarizes the guarantee:

| Stage | Frames seen by model | Point annotations |
|-------|----------------------|-------------------|
| **Training** | `T` extracted frames (at `sampling_fps`) | `T` filtered annotation frames |
| **Testing** | `T` extracted frames (at `sampling_fps`) | `T` filtered GT frames; `T` prediction slots |

Both `T` values are determined by the **same** video-decoding + timestamp-filtering pipeline, so the counts are always consistent. See the [Tracking Pipeline section in the README](../README.md#tracking-pipeline) for implementation details.

---

## Code Map

| File | Purpose |
|------|---------|
| `olmo/data/academic_video_track_datasets.py` | Academic tracking dataset loaders (MeViS, Ref-YT-VOS, DAVIS, BURST, etc.) |
| `olmo/data/molmo2_video_track_datasets.py` | Molmo2 custom tracking datasets and per-source video downloaders |
| `olmo/preprocessing/data_formatter.py` | `format_video_object_track_points()`, `_filter_frames_to_video()`, `_sample_at_fps()` |
| `olmo/eval/object_tracking_utils.py` | Prediction parsing (multi-object), Hungarian matching, Precision/Recall/F1/HOTA computation |
| `olmo/eval/point_tracking_utils.py` | TAP-Vid metric computation (`compute_tapvid_metrics()`), occlusion-aware parsing |
| `olmo/eval/evaluators.py` | `VideoObjectTrackingEval` evaluator class |
| `olmo/eval/model_evaluator.py` | Evaluation orchestration (`ModelEvaluator`) |
| `launch_scripts/sft.py` | SFT training entry point (includes tracking data) |
| `launch_scripts/eval_molmo2.py` | Multi-task evaluation entry point (includes `TRACKING` group) |
| `scripts/download_datasets.py` | Dataset download utility (group: `video_tracking`) |
