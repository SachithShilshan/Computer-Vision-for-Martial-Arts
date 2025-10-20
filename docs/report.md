# Report: Player Detection and Tracking

**Date:** 2025-10-20

## 1. Models and Setup
- Detector: YOLO-like framework (per notebook).
- (Optional) Keypoints: OpenPose-style (bonus).
- Frameworks: PyTorch/TensorFlow as used in the notebook.

## 2. Dataset
Short 5–10s clips from public sports videos. Add exact links in README under *Dataset*.

## 3. Performance Comparison

**Precision:** N/A  
**Recall:** N/A  
**mAP:** N/A  
**Accuracy:** N/A  
**F1:** N/A  
**Loss values seen:** N/A  
**Validation loss values seen:** N/A  

Loss curves: not detected in outputs. If available in the notebook, export the plot images to screenshots/.

Include sample detections and tracked frames (see `screenshots/`).

## 4. Discussion
### Strengths
- Robust detection on clear frames and common poses.
- Efficient inference enabling near real-time tracking (depending on hardware).

### Limitations
- Occlusions and crowded scenes reduce ID stability.
- Fast camera motion causes missed detections.
- Domain shift across different sports/jerseys can degrade performance.
- Small/low-res players at long distances challenge detectors.

### Possible Improvements
- Train/fine-tune with sport-specific annotated data.
- Use strong data augmentation (motion blur, scale jitter).
- Multi-object tracker with re-ID (e.g., DeepSORT/ByteTrack) to stabilize IDs.
- Temporal smoothing with optical flow or transformer-based trackers.
- Keypoint-assisted association for overlaps/occlusions.
- Post-processing: non-maximum suppression tuning and confidence thresholds.

## 5. Reproducibility
- Code provided in `code/` (with and without outputs).  
- Metrics auto-extracted from notebook where available (`docs/metrics_summary.json`).

---
*Prepared from the provided Colab notebook with outputs.*
