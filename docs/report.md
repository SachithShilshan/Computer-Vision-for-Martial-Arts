# Fighters Pose Analysis and Action Recognition

**Author:** Sachith Shilshan • **Course:** DS5216

> 
(1) detect fighters, (2) estimate body keypoints, (3) output per-frame **kick/punch/block** probabilities, and (4) overlay a short-term **predicted (dashed) pose**.
> The emphasis of this report is **Performance** (accuracy/precision/recall/mAP & loss curves), followed by discussion and improvements.

---

## 1. Introduction


This project analyzes martial-arts footage to track fighters, draw pose skeletons, show **kick/punch/block** probabilities, and render a **dashed prediction** of the next pose.

---

## 2. Dataset Preparation

This project uses a ** dataset created from 6 martial arts video clips** featuring **karate, boxing, kickboxing, UFC, ..** with fighters. The videos include a mix of movement dynamics.

Player detection dataset and Pose dataset Split 80/20 for train/val.

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

<img width="640" height="480" alt="image" src= "https://github.com/user-attachments/assets/180aef20-29bb-4ffb-b093-56cc81c28f9b" />

<img width="640" height="480" alt="image" src= "https://github.com/user-attachments/assets/a6a74815-451c-412a-a961-a4742749bed0" />

<img width="640" height="480" alt="image" src= "https://github.com/user-attachments/assets/d36c48b6-f354-45c2-9fe4-85797563b979" />

<img width="640" height="480" alt="image" src= "https://github.com/user-attachments/assets/68a09e04-7799-4f80-b6ce-a5591ddcd558" />

### 3.2 Pose (Keypoint) Detection (YOLOv8-pose)

* **pretrained:** `yolov8s-pose.pt` 
* **Fine-tuned models:** `yolov8s-pose.pt` 

* Metrics: **OKS mAP@50-95** (primary), **OKS mAP@50**, plus precision/recall when available.

<img width="640" height="480" alt="image" src= "https://github.com/user-attachments/assets/12bf9e68-06d1-4a3f-b75d-6b2eb28e4832" />

<img width="640" height="480" alt="image" src= "https://github.com/user-attachments/assets/144776d9-72d8-460d-9abd-d245ba3e94a2" />
 
<img width="640" height="480" alt="image" src= "https://github.com/user-attachments/assets/01a0d937-1ce1-44ae-b040-b53441394c0a" />

<img width="640" height="480" alt="image" src= "https://github.com/user-attachments/assets/f5a8a76e-1cb1-42ce-bcf9-7675641496d8" />

<img width="640" height="480" alt="image" src= "https://github.com/user-attachments/assets/738b466a-20e5-4d12-96b5-6ec2c8224b0a" />



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

![tri_video_4_overlay_ss_03_f000040](https://github.com/user-attachments/assets/a4c804da-284c-4641-9c22-a5ef82125b83" />

---

## 4. Performance (Main)

### 4.1 Player Detection Results



| Weights                        |  mAP50-95 |     mAP50 | Precision |    Recall |
| ------------------------------ | --------: | --------: | --------: | --------: |
| `yolov8n.pt` (pretrained)      | **0.744** |     0.927 |     0.920 |     0.807 |
| `best.pt` (fine-tuned yolov8n) | **0.848** | **0.977** |     0.924 | **0.975** |

Output table

<img width="709" height="153" alt="image" src="https://github.com/user-attachments/assets/6954086d-9f13-474d-9807-20432f31f9f8" />


**Key takeaways**

* Fine-tuning increased **mAP50-95 by +10.4 points** (0.744 → 0.848).
* **Recall** jumped from **0.807 → 0.975**, with precision remaining ~0.92 → many fewer misses without extra false positives.

---

### 4.2 Pose (Keypoint) Detection Results

#### (A) Latest: YOLOv8s-pose


| Model                    | mAP50-95 (OKS) | mAP50 (OKS) | Precision |    Recall |
| ------------------------ | -------------: | ----------: | --------: | --------: |
| `y8s_pose_pretrained`    |          0.204 |       0.227 |     0.254 |     0.500 |
| **`y8s_pose_finetuned`** |      **0.522** |   **0.613** | **0.557** | **0.660** |

Output table

<img width="850" height="148" alt="image" src="https://github.com/user-attachments/assets/c2265bb3-ea6c-43b1-9f22-013e190e8850" />


**Gain:** **+31.8 pts** OKS mAP50-95 after fine-tuning.

---

### 4.3 Training Curves


**Detector (fine-tuned `yolov8n`)**

* <img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/1a6e0ff4-8c98-4ba5-a73e-c868c5606897" />

* <img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/ca77c9d6-9a06-4ea7-bf36-ffceb90c5fac" />

* <img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/e8cbecdf-df13-46f4-adeb-f25f6e046bad" />

* <img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/028af8a7-fa5a-4a14-a94c-9e89adfbc85f" />

* <img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/4076b27d-6425-47e9-8cbc-b97b3196d012" />

* <img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/b5dd7e81-43fb-4ea0-99e0-6b665f9ff62e" />

* <img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/dda6ba78-d2b8-41fd-a5bf-3e890c5210a0" />


**Pose (fine-tuned `yolov8s-pose`)**

* <img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/5c829e86-f710-42b4-99a4-c6933cb50bb0" />

* <img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/408525a8-754d-48ef-9ef8-689d7d0c494e" />

* <img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/02f4ff9b-3f1c-48f5-ad5c-3bf05a886e60" />

* <img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/8ebe7f89-4bb0-4548-ad1b-0c2e15382c58" />



* Losses **decrease** steadily → healthy optimization.
* **Val mAP/recall** increase epoch-by-epoch and stabilize near the end.
* Detector converges to **mAP50≈0.98** and **mAP50-95≈0.85**; pose reaches **OKS mAP50≈0.61**, **OKS mAP50-95≈0.50**.

---


## 5. Discussion

* **Why fine-tune:** Pretrained models struggle with fast motion, occlusion, ring ropes, and gloves. **Short domain fine-tuning** yields substantial gains, especially **recall** for detection and **OKS mAP** for keypoints.
* **Detector vs Pose:** The detector is production-ready (high precision/recall). Pose quality is the main driver for action probabilities; the fine-tuned `yolov8s-pose` best localizes hands/feet → better kick/punch cues.
* **Explainability:** Action scores come from transparent pose features (distances, heights, velocities) + EMA smoothing.
* **Prediction horizon:** Constant-velocity pose forecasts look clean for ≲2–3 frames; longer horizons need temporal learning.

---

## 7. Improvements & Future Work

**Temporal modeling (replace heuristics)**

* **Transformers over pose sequences (6–24 frames):** joint **action classification + anticipation** (predict labels at *t* and *t+Δ*).
* **GRU/LSTM** pose forecaster with smoothness + bone-length constraints to replace constant-velocity.

**Data & training**

* Sports-aware augmentations: **motion blur**, random occluders (rope/ref), light flares.
* Increase input size (e.g., **768/960**) for finer wrist/ankle detail (trade speed).
* **Self-training loop:** re-label extra frames with your fine-tuned pose model and keep high-confidence joints.

