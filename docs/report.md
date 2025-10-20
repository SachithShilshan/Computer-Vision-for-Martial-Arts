# Computer Vision for Fighter Pose Analysis and Action Recognition

**Author:** Sachith Shilshan • **Course:** DS5216

> A compact pipeline to (1) detect fighters, (2) estimate body keypoints, (3) output per-frame **kick/punch/block** probabilities, and (4) overlay a short-term **predicted (dashed) pose**.
> The emphasis of this report is **Performance** (accuracy/precision/recall/mAP & loss curves), followed by discussion, limitations, and improvements.

---

## Contents

* [1. Introduction](#1-introduction)
* [2. Dataset Preparation](#2-dataset-preparation)
* [3. Methods](#3-methods)

  * [3.1 Player Detection (YOLOv8)](#31-player-detection-yolov8)
  * [3.2 Pose (Keypoint) Detection (YOLOv8-pose)](#32-pose-keypoint-detection-yolov8pose)
  * [3.3 Action Likelihoods & Short-term Pose Prediction](#33-action-likelihoods--short-term-pose-prediction)
* [4. Performance (Main)](#4-performance-main)

  * [4.1 Player Detection Results](#41-player-detection-results)
  * [4.2 Pose (Keypoint) Detection Results](#42-pose-keypoint-detection-results)
  * [4.3 Training Curves](#43-training-curves)
  * [4.4 Speed](#44-speed)
* [5. Discussion](#5-discussion)
* [6. Limitations](#6-limitations)
* [7. Improvements & Future Work](#7-improvements--future-work)
* [8. How to Reproduce](#8-how-to-reproduce)

---

## 1. Introduction

This project analyzes martial-arts footage to track two fighters, draw pose skeletons, show **kick/punch/block** probabilities, and render a **dashed prediction** of the next pose. The system is designed to be **fast, explainable, and reproducible** in Google Colab with Google Drive storage.

---

## 2. Dataset Preparation

* Frames auto-extracted from the raw videos (≈ **20 frames/video**).
* **Player detection dataset** (YOLO format):

  * Classes: `person` only; split **80/20** train/val.
* **Pose dataset** (YOLO-pose format):

  * Pseudo-labels generated with **YOLOv8x-pose** on extracted frames.
  * `kpt_shape: [17, 3]` → 17 COCO keypoints with (x, y, visibility).
  * Split **80/20** train/val.
* No manual labeling was required.

Folder layout (under `MyDrive/DS5216`):

```
data/
  person_dataset/         # detector
    images/{train,val}/
    labels/{train,val}/
    person.yaml
  pose_dataset/           # keypoints
    images/{train,val}/
    labels/{train,val}/
    person_pose.yaml
outputs/
exp_det_train/, exp_det_eval2/
exp_pose_train/, exp_pose_eval/
```

---

## 3. Methods

### 3.1 Player Detection (YOLOv8)

* **Baseline:** `yolov8n.pt` (pretrained).
* **Fine-tuned:** `yolov8n` trained **20 epochs @ 640** on the `person_dataset`.
* Metrics: **mAP@50**, **mAP@50-95**, **precision**, **recall** on the same val split.

### 3.2 Pose (Keypoint) Detection (YOLOv8-pose)

* **Annotator (pretrained):** `yolov8x-pose.pt` — used only to generate pseudo keypoint labels.
* **Fine-tuned models:**

  * `yolov8n-pose.pt` (earlier run)
  * `yolov8s-pose.pt` (**latest, recommended**)
* Metrics: **OKS mAP@50-95** (primary), **OKS mAP@50**, plus precision/recall when available.

### 3.3 Action Likelihoods & Short-term Pose Prediction

* **Action probabilities** (per frame): features from keypoints → softmax → **kick/punch/block**

  * **Punch:** wrist near opponent head + hand velocity
  * **Kick:** ankle above hip line + leg velocity
  * **Block/Guard:** both hands near own head + low body velocity
  * Temporal smoothing: **EMA** to reduce flicker.
* **Pose prediction:** two-frame **constant-velocity** extrapolation of keypoints with

  * torso anchoring (hips/shoulders),
  * head blending,
  * bone-length constraint.
    Predicted pose is drawn as a **dashed skeleton**.

---

## 4. Performance (Main)

### 4.1 Player Detection Results

**Validation:** 60 images, 243 persons.

| Weights                        |  mAP50-95 |     mAP50 | Precision |    Recall |
| ------------------------------ | --------: | --------: | --------: | --------: |
| `yolov8n.pt` (pretrained)      | **0.744** |     0.927 |     0.920 |     0.807 |
| `best.pt` (fine-tuned yolov8n) | **0.848** | **0.977** |     0.924 | **0.975** |

**Key takeaways**

* Fine-tuning increased **mAP50-95 by +10.4 points** (0.744 → 0.848).
* **Recall** jumped from **0.807 → 0.975**, with precision remaining ~0.92 → many fewer misses without extra false positives.

---

### 4.2 Pose (Keypoint) Detection Results

#### (A) Latest: YOLOv8s-pose (recommended)

**Validation:** 50 images.

| Model                    | mAP50-95 (OKS) | mAP50 (OKS) | Precision |    Recall |
| ------------------------ | -------------: | ----------: | --------: | --------: |
| `y8s_pose_pretrained`    |          0.204 |       0.227 |     0.254 |     0.500 |
| **`y8s_pose_finetuned`** |      **0.522** |   **0.613** | **0.557** | **0.660** |

**Gain:** **+31.8 pts** OKS mAP50-95 after fine-tuning on domain pseudo-labels.

#### (B) Earlier: YOLOv8n-pose (for reference)

| Model                                 | mAP50-95 (OKS) | mAP50 (OKS) | Precision |    Recall |
| ------------------------------------- | -------------: | ----------: | --------: | --------: |
| `yolov8x-pose` (pretrained annotator) |          0.306 |       0.310 |     0.308 |     0.683 |
| **`yolov8n-pose` (fine-tuned)**       |      **0.512** |   **0.618** | **0.479** | **0.720** |

**Takeaway:** both `n-pose` and `s-pose` improve strongly; **`s-pose` edges out `n-pose`** on this dataset.

---

### 4.3 Training Curves

> Place the images below in your repo’s `assets/` folder and keep these links.

**Detector (fine-tuned `yolov8n`)**

* ![train/box\_loss](assets/det_train_box_loss.png)
* ![train/cls\_loss](assets/det_train_cls_loss.png)
* ![train/dfl\_loss](assets/det_train_dfl_loss.png)
* ![metrics/precision(B)](assets/det_val_precision.png)
* ![metrics/recall(B)](assets/det_val_recall.png)
* ![metrics/mAP50(B)](assets/det_val_map50.png)
* ![metrics/mAP50-95(B)](assets/det_val_map5095.png)

**Pose (fine-tuned `yolov8s-pose`)**

* ![train/box\_loss](assets/pose_train_box_loss.png)
* ![train/kobj\_loss](assets/pose_train_kobj_loss.png)
* ![metrics/mAP50(B)](assets/pose_val_map50.png)
* ![metrics/mAP50-95(B)](assets/pose_val_map5095.png)

One-liner reads:

* Losses **decrease** steadily → healthy optimization.
* **Val mAP/recall** increase epoch-by-epoch and stabilize near the end.
* Detector converges to **mAP50≈0.98** and **mAP50-95≈0.85**; pose reaches **OKS mAP50≈0.61**, **OKS mAP50-95≈0.50**.

---

### 4.4 Speed

* Detector (`y8n` fine-tuned) @ 640 on T4: **30.5 ± 29.9 ms/image**.
* Pose (`y8s-pose` fine-tuned) @ 640 on T4: **~9–15 ms/image** in validation logs.
* Suitable for near real-time with modest batching.

---

## 5. Discussion

* **Why fine-tune:** Pretrained models struggle with fast motion, occlusion, ring ropes, and gloves. **Short domain fine-tuning** yields substantial gains, especially **recall** for detection and **OKS mAP** for keypoints.
* **Detector vs Pose:** The detector is production-ready (high precision/recall). Pose quality is the main driver for action probabilities; the fine-tuned `yolov8s-pose` best localizes hands/feet → better kick/punch cues.
* **Explainability:** Action scores come from transparent pose features (distances, heights, velocities) + EMA smoothing.
* **Prediction horizon:** Constant-velocity pose forecasts look clean for ≲2–3 frames; longer horizons need temporal learning.

---

## 6. Limitations

* **Pseudo-labels ≠ perfect GT:** annotator errors propagate to the student model.
* **Single-class detector:** only `person`; no explicit glove/foot instances.
* **Heuristic actions:** not end-to-end trained; edge cases (feints, partial views) can be confused.
* **Small val sets:** 50–60 images → higher variance; consider k-fold or more frames.

---

## 7. Improvements & Future Work

**Temporal modeling (replace heuristics)**

* **Transformers over pose sequences (6–24 frames):** joint **action classification + anticipation** (predict labels at *t* and *t+Δ*).
* **GRU/LSTM** pose forecaster with smoothness + bone-length constraints to replace constant-velocity.

**Data & training**

* Add `flip_idx` to `person_pose.yaml` for better left/right invariance.
* Sports-aware augmentations: **motion blur**, random occluders (rope/ref), light flares.
* Increase input size (e.g., **768/960**) for finer wrist/ankle detail (trade speed).
* **Self-training loop:** re-label extra frames with your fine-tuned pose model and keep high-confidence joints.

**System**

* StrongSORT + re-ID embeddings to reduce ID switches.
* TensorRT export for sub-10 ms detector inference on supported GPUs.

---

## 8. How to Reproduce

**Colab prerequisites**

```python
BASE = "/content/drive/MyDrive/DS5216"
# Mount Drive, create folders:
# data/person_dataset, data/pose_dataset, outputs, exp_* as above
!pip -q install ultralytics==8.3.34 opencv-python-headless==4.10.0.84
```

**Steps**

1. **Extract frames** (~20 per video) into `pose_dataset/images/all` and `person_dataset/images/all`.
2. **Auto-label persons** (detector) with YOLO and write YOLO txt labels.
3. **Auto-label keypoints** (pose) with `yolov8x-pose` → YOLO-pose txt (`kpt_shape: [17,3]`).
4. **Split 80/20** into `images/{train,val}` and `labels/{train,val}`; write `person.yaml` & `person_pose.yaml`.
5. **Fine-tune**

   * Detector: `yolov8n.pt` → 20 epochs @ 640.
   * Pose: `yolov8s-pose.pt` → 20 epochs @ 640.
6. **Validate** both on the val split; collect **mAP / OKS mAP / precision / recall**.
7. **Run the pipeline** (detect → pose → probabilities → dashed pose prediction).
8. **Export figures** and place them in `assets/` to match the links in this README.

---

## Appendix: Quick Performance Summary

**Player Detection @ 640**

* `yolov8n.pt`: mAP50-95 **0.744**, Precision **0.920**, Recall **0.807**
* `best.pt` (fine-tuned): mAP50-95 **0.848**, Precision **0.924**, Recall **0.975**, **~30.5 ms/img**

**Pose (Keypoints) – OKS**

* `y8s_pose_pretrained`: **0.204** OKS mAP50-95
* **`y8s_pose_finetuned`: 0.522** OKS mAP50-95
* (`y8n-pose` fine-tuned earlier: **0.512** OKS mAP50-95)

---

### Assets to add (rename your PNGs to these)

```
assets/
  det_train_box_loss.png
  det_train_cls_loss.png
  det_train_dfl_loss.png
  det_val_precision.png
  det_val_recall.png
  det_val_map50.png
  det_val_map5095.png
  pose_train_box_loss.png
  pose_train_kobj_loss.png
  pose_val_map50.png
  pose_val_map5095.png
```

---

**Contact**: For any reproducibility questions or to extend with Transformers/GRU forecasting, open an issue in the repo. 💬
