# ---- cell ----
# Check GPU
!nvidia-smi -L || echo "No GPU detected (enable GPU: Runtime → Change runtime type)"

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Base path in Drive
BASE = "/content/drive/MyDrive/DS5216"
print("Using base:", BASE)


# ---- cell ----
!pip -q install ultralytics==8.3.34 supervision==0.22.0 opencv-python-headless==4.10.0.84 yt-dlp==2024.10.07
!apt -y install ffmpeg > /dev/null


# ---- cell ----
from pathlib import Path
import os, glob, json, cv2, numpy as np, pandas as pd
from ultralytics import YOLO
from datetime import datetime
from IPython.display import Video, display

# Folders
DATA_DIR = Path(BASE) / "data"
VID_DIR  = DATA_DIR / "videos"
OUT_DIR  = Path(BASE) / "outputs"
SS_DIR   = Path(BASE) / "screenshots"
for p in [DATA_DIR, VID_DIR, OUT_DIR, SS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("Folders:")
print("-", VID_DIR)
print("-", OUT_DIR)
print("-", SS_DIR)


# ---- cell ----
# List videos found
videos = sorted(glob.glob(str(VID_DIR / "*.mp4")) +
                glob.glob(str(VID_DIR / "*.mkv")) +
                glob.glob(str(VID_DIR / "*.mov")) +
                glob.glob(str(VID_DIR / "*.webm")))
print(f"Found {len(videos)} video(s):")
for v in videos: print("-", Path(v).name)

# ---- cell ----
det_model = YOLO("yolov8n.pt")
PERSON_CLASS_ID = 0
print("yolov8n.pt Model ready.")


# ---- cell ----
def track(
    src_path,
    save_prefix="yolo8n",
    conf=0.35,
    iou=0.45,
    tracker="bytetrack.yaml",
    take_screens=5
):
    src = Path(src_path)
    name = src.stem.strip().replace(" ", "_")

    # Probe video for size/FPS
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1: fps = 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    # Output paths
    out_video_path = OUT_DIR /"Player Detection"/ f"{save_prefix}_{name}_tracked.mp4"
    out_json_path  = OUT_DIR /"Player Detection"/f"{save_prefix}_{name}_stats.json"

    # VideoWriter (MP4 → fallback to AVI if needed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        out_video_path = OUT_DIR / f"{save_prefix}_{name}_tracked.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter for both MP4 and AVI.")

    # Screenshots positions
    shot_positions = []
    if total_frames > 0:
        take = min(take_screens, total_frames) if total_frames < take_screens else take_screens
        shot_positions = [int((i+1) * total_frames / (take+1)) for i in range(take)]
    screenshot_buffer = {}

    # Stats
    frame_idx = 0
    frame_counts = []
    track_ids_seen = set()
    id_switches = 0
    prev_frame_ids = set()

    # Stream detections (no RAM buildup)
    gen = det_model.track(
        source=str(src),
        conf=conf, iou=iou, classes=[PERSON_CLASS_ID],
        tracker=tracker,
        save=False, stream=True, verbose=False
    )

    for r in gen:
        frame = r.plot()           # annotated frame (BGR)
        writer.write(frame)

        # Stats via track IDs
        ids = []
        if r.boxes is not None and r.boxes.id is not None:
            ids = [int(x) for x in r.boxes.id.cpu().numpy().tolist()]
        frame_counts.append(len(ids))
        new_ids = set(ids)
        appeared = new_ids - prev_frame_ids
        disappeared = prev_frame_ids - new_ids
        if appeared and disappeared:
            id_switches += min(len(appeared), len(disappeared))
        prev_frame_ids = new_ids
        track_ids_seen.update(ids)

        # Save screenshots
        if frame_idx in shot_positions:
            screenshot_buffer[frame_idx] = frame.copy()
        frame_idx += 1

    writer.release()

    # Write screenshots
    for pos, img in screenshot_buffer.items():
        cv2.imwrite(str(SS_DIR / f"{save_prefix}_{name}_frame{pos}.jpg"), img)

    # Save stats
    stats = {
        "video": src.name,
        "frames_processed": len(frame_counts),
        "mean_persons_per_frame": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "max_persons_in_frame": int(np.max(frame_counts)) if frame_counts else 0,
        "unique_track_ids": len(track_ids_seen),
        "approx_id_switches": int(id_switches),
        "conf": conf, "iou": iou, "tracker": tracker,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_json_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("Saved video:", out_video_path)
    print("Saved stats:", out_json_path)
    return out_video_path, out_json_path


# ---- cell ----
generated = []
for v in videos:
    out_vid, out_json = track(v, save_prefix="yolo8n")
    generated.append((out_vid, out_json))

print("\nGenerated files:")
for v, j in generated:
    print("-", Path(v).name, "|", Path(j).name)



# ---- cell ----
rows = []
for js in glob.glob(str(OUT_DIR / "*_stats.json")):
    with open(js) as f:
        rows.append(json.load(f))
df = pd.DataFrame(rows)
if not df.empty:
    df = df[["video","frames_processed","mean_persons_per_frame","max_persons_in_frame",
             "unique_track_ids","approx_id_switches","conf","iou","tracker","timestamp"]]
    df.sort_values("video", inplace=True, ignore_index=True)

csv_path = OUT_DIR / "tracking_summary.csv"
df.to_csv(csv_path, index=False)
print("Saved:", csv_path)
df


# ---- cell ----
DATA_YAML = f"{BASE}/data/person_dataset/person.yaml"
PROJECT_TRAIN = f"{BASE}/exp_det_train"

model = YOLO("yolov8n.pt")
r = model.train(
    data=DATA_YAML, epochs=20, imgsz=640, batch=16, device=0,
    project=PROJECT_TRAIN, name="person_y8n_ft", verbose=False
)
best = Path(r.save_dir) / "weights/best.pt"
best


# ---- cell ----
import pandas as pd, matplotlib.pyplot as plt, glob, os

csv = sorted(glob.glob(f"{PROJECT_TRAIN}/person_y8n_ft/results.csv"))[-1]
df = pd.read_csv(csv)

for col in ["train/box_loss","train/cls_loss","train/dfl_loss",
            "metrics/precision(B)","metrics/recall(B)","metrics/mAP50(B)","metrics/mAP50-95(B)"]:
    if col in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df[col])
        plt.title(col)
        plt.xlabel("Epoch"); plt.tight_layout(); plt.show()


# ---- cell ----
import pandas as pd
from ultralytics import YOLO

DATA_YAML = f"{BASE}/data/person_dataset/person.yaml"
PROJECT_VAL = f"{BASE}/exp_det_eval2"
weights_to_compare = ["yolov8n.pt", str(best)]  # baseline vs fine-tuned
rows = []

for w in weights_to_compare:
    res = YOLO(w).val(data=DATA_YAML, imgsz=640, plots=True, save_json=True,
                      project=PROJECT_VAL, name=f"val_{Path(w).stem}", verbose=False)
    d = res.results_dict
    rows.append({
        "weights": Path(w).name,
        "mAP50-95": d.get("metrics/mAP50-95(B)", d.get("metrics/mAP50-95")),
        "mAP50":    d.get("metrics/mAP50(B)",    d.get("metrics/mAP50")),
        "precision":d.get("metrics/precision(B)",d.get("precision")),
        "recall":   d.get("metrics/recall(B)",   d.get("recall")),
        "inference_ms": d.get("speed/inference")
    })

df_cmp = pd.DataFrame(rows)
df_cmp


# ---- cell ----
cmp_csv = f"{BASE}/outputs/det_comparison.csv"
Path(f"{BASE}/outputs").mkdir(parents=True, exist_ok=True)
df_cmp.to_csv(cmp_csv, index=False)
print("Saved:", cmp_csv)

# ---- cell ----
import json, numpy as np, matplotlib.pyplot as plt, glob

for pr_json in glob.glob(f"{PROJECT_VAL}/val_*/PR_curve.json"):
    with open(pr_json) as f: pr = json.load(f)
    conf = np.array(pr["confidence"]); P = np.array(pr["precision"]); R = np.array(pr["recall"])
    F1 = 2*(P*R)/(P+R+1e-9)
    plt.figure(); plt.plot(conf, F1); plt.title(pr_json.split("/")[-2]+" F1 vs conf")
    plt.xlabel("Confidence"); plt.ylabel("F1"); plt.grid(True, alpha=.3); plt.show()


# ---- cell ----
from ultralytics import YOLO
import time, cv2, glob

test_imgs = glob.glob(f"{BASE}/data/person_dataset/images/val/*.jpg")[:10]
model = YOLO(str(best))  # or "yolov8n.pt"
t=[]
for im in test_imgs:
    s=time.time(); model.predict(source=im, imgsz=640, conf=0.25, classes=[0], verbose=False)
    t.append((time.time()-s)*1000)
print(f"Inference ms/img (mean±std): {np.mean(t):.1f} ± {np.std(t):.1f}")


# ---- cell ----
SS_DIR  = Path(BASE) / "screens"

for p in [OUT_DIR, SS_DIR, OUT_DIR / "Keypoint Detection"]:
    p.mkdir(parents=True, exist_ok=True)

# Person class id for COCO
PERSON_CLASS_ID = 0

# Colors (BGR)
C_RED  = (0, 0, 255)
C_BLUE = (255, 0, 0)
C_DARK = (0, 0, 0)

print("Folders ready:", OUT_DIR, SS_DIR)

# ---- cell ----
from ultralytics import YOLO

pose_model = YOLO("yolov8s-pose.pt")
print("Pose model ready.")


# ---- cell ----
# COCO-17 skeleton edges
COCO_EDGES = [
    (5,7),(7,9), (6,8),(8,10),
    (11,13),(13,15), (12,14),(14,16),
    (5,6), (11,12), (5,11),(6,12),
    (0,1),(1,2),(2,3),(3,4)
]

def get_kpts_xyc(result, det_index):

    kobj = getattr(result, "keypoints", None)
    if kobj is None:
        return None, None

    # Prefer .data
    if hasattr(kobj, "data") and kobj.data is not None:
        arr = kobj.data.cpu().numpy()  # [N, 17, 3] or [N, 17, 2]
        if det_index >= arr.shape[0]:
            return None, None
        this = arr[det_index]
        if this.shape[-1] == 3:
            xy = this[:, :2]
            cf = this[:, 2]
        else:
            xy = this[:, :2]
            cf = np.ones((xy.shape[0],), dtype=float)
        return xy, cf

    # Fallback: .xy (positions only)
    if hasattr(kobj, "xy") and kobj.xy is not None:
        arr = kobj.xy.cpu().numpy()  # [N, 17, 2]
        if det_index >= arr.shape[0]:
            return None, None
        xy = arr[det_index]
        cf = np.ones((xy.shape[0],), dtype=float)
        return xy, cf

    return None, None

def get_head_point(xy, conf, conf_thr=0.25):
    """
    A robust single "head" point:
    - prefer nose (idx 0) if confident,
    - else mean of eyes (1,2),
    - else mean of shoulders (5,6),
    - else fall back to nose.
    """
    if xy is None:
        return None
    if conf is None or (len(conf)>0 and conf[0] >= conf_thr):
        return xy[0]
    if len(xy) > 2:
        eyes = [xy[1], xy[2]]
        if eyes[0] is not None and eyes[1] is not None:
            return np.mean(np.stack(eyes), axis=0)
    if len(xy) > 6:
        sh = [xy[5], xy[6]]
        if sh[0] is not None and sh[1] is not None:
            return np.mean(np.stack(sh), axis=0)
    return xy[0]

def get_fists(xy, conf=None):
    """
    Fists == wrists (left=9, right=10).
    Returns dict {'left','right','best'}
    """
    if xy is None:
        return {"left": None, "right": None, "best": None}
    lw = xy[9]  if len(xy) > 9  else None
    rw = xy[10] if len(xy) > 10 else None
    if conf is None:
        best = lw if lw is not None else rw
    else:
        lc = conf[9]  if len(conf) > 9  else 0.0
        rc = conf[10] if len(conf) > 10 else 0.0
        best = lw if lc >= rc else rw
    return {"left": lw, "right": rw, "best": best}

def get_feet(xy):
    """
    Feet == ankles (left=15, right=16). Also returns 'lower' (closer to the floor in image coords).
    """
    if xy is None:
        return {"left": None, "right": None, "lower": None, "higher": None}
    la = xy[15] if len(xy) > 15 else None
    ra = xy[16] if len(xy) > 16 else None
    cand = [p for p in [la, ra] if p is not None]
    if not cand:
        return {"left": la, "right": ra, "lower": None, "higher": None}
    lower = max(cand, key=lambda p: p[1])   # larger y → lower on screen
    higher = min(cand, key=lambda p: p[1])
    return {"left": la, "right": ra, "lower": lower, "higher": higher}

def draw_landmarks(frame, head, fists, feet, color=(0, 0, 0)):
    """
    Draw HEAD, LEFT/RIGHT FIST (+ ring on best), LEFT/RIGHT FOOT.
    """
    # Head
    if head is not None:
        cv2.circle(frame, (int(head[0]), int(head[1])), 6, color, -1, cv2.LINE_AA)
        cv2.putText(frame, "HEAD", (int(head[0])+6, int(head[1])-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Fists
    for tag in ["left", "right"]:
        p = fists.get(tag)
        if p is not None:
            cv2.circle(frame, (int(p[0]), int(p[1])), 6, color, -1, cv2.LINE_AA)
            cv2.putText(frame, f"{tag.upper()} FIST", (int(p[0])+6, int(p[1])-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Best fist (outline ring)
    best = fists.get("best")
    if best is not None:
        cv2.circle(frame, (int(best[0]), int(best[1])), 10, color, 2, cv2.LINE_AA)

    # Feet
    for tag in ["left", "right"]:
        p = feet.get(tag)
        if p is not None:
            cv2.circle(frame, (int(p[0]), int(p[1])), 6, color, -1, cv2.LINE_AA)
            cv2.putText(frame, f"{tag.upper()} FOOT", (int(p[0])+6, int(p[1])-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# ---- cell ----
def track_pose_and_render_manual(
    src_path,
    save_prefix="yolov8s_pose",
    conf=0.35,
    iou=0.45,
    tracker="bytetrack.yaml",
    take_screens=5
):
    src = Path(src_path)
    name = src.stem.strip().replace(" ", "_")

    # Probe source video
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1: fps = 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    # Output paths
    out_dir = OUT_DIR / "Keypoint Detection"
    out_video_path = out_dir / f"{save_prefix}_{name}_pose.mp4"
    out_json_path  = out_dir / f"{save_prefix}_{name}_pose_stats.json"

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        out_video_path = out_dir / f"{save_prefix}_{name}_pose.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter for both MP4 and AVI.")

    # Screenshot positions
    shot_positions = []
    if total_frames > 0:
        take = min(take_screens, total_frames) if total_frames < take_screens else take_screens
        shot_positions = [int((i+1) * total_frames / (take+1)) for i in range(take)]
    screenshot_buffer = {}

    # Stats
    frame_idx = 0
    frame_counts = []
    track_ids_seen = set()
    id_switches = 0
    prev_frame_ids = set()

    # Stream pose tracking: persons only
    gen = pose_model.track(
        source=str(src),
        conf=conf, iou=iou, classes=[PERSON_CLASS_ID],
        tracker=tracker,
        save=False, stream=True, verbose=False
    )

    for r in gen:
        # Draw default overlays (boxes + skeletons)
        frame = r.plot()

        # --- Extract track IDs (for stats & color consistency) ---
        ids = []
        if r.boxes is not None and r.boxes.id is not None:
            ids = [int(x) for x in r.boxes.id.cpu().numpy().tolist()]
        frame_counts.append(len(ids))
        new_ids = set(ids)
        appeared = new_ids - prev_frame_ids
        disappeared = prev_frame_ids - new_ids
        if appeared and disappeared:
            id_switches += min(len(appeared), len(disappeared))
        prev_frame_ids = new_ids
        track_ids_seen.update(ids)

        # --- For each detection, read keypoints and mark HEAD/FISTS/FEET ---
        for i, tid in enumerate(ids):
            xy, cf = get_kpts_xyc(r, i)
            if xy is None:
                continue

            # Choose a per-ID color (use your Red/Blue if you have mapping; here parity as simple example)
            color = C_RED if (tid % 2 == 0) else C_BLUE

            head = get_head_point(xy, cf, conf_thr=0.25)
            fists = get_fists(xy, cf)
            feet  = get_feet(xy)

            draw_landmarks(frame, head, fists, feet, color=color)

        # Save screenshots
        if frame_idx in shot_positions:
            screenshot_buffer[frame_idx] = frame.copy()

        # Write frame
        writer.write(frame)
        frame_idx += 1

    writer.release()

    # Dump screenshots
    for pos, img in screenshot_buffer.items():
        cv2.imwrite(str(SS_DIR / f"{save_prefix}_{name}_pose_frame{pos}.jpg"), img)

    # Save stats
    stats = {
        "video": src.name,
        "frames_processed": frame_idx,
        "mean_persons_per_frame": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "max_persons_in_frame": int(np.max(frame_counts)) if frame_counts else 0,
        "unique_track_ids": len(track_ids_seen),
        "approx_id_switches": int(id_switches),
        "conf": conf, "iou": iou, "tracker": tracker,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "pose+landmarks",
    }
    with open(out_json_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("Saved pose+landmarks video:", out_video_path)
    print("Saved pose stats:", out_json_path)
    return out_video_path, out_json_path


# ---- cell ----
pose_generated = []
for v in videos:
    out_vid, out_json = track_pose_and_render_manual(v, save_prefix="yolov8s")
    pose_generated.append((out_vid, out_json))

print("\nPose files generated:")
for v, j in pose_generated:
    print("-", Path(v).name, "|", Path(j).name)


# ---- cell ----
BASE = "/content/drive/MyDrive/DS5216"
POSE_DATA = Path(BASE) / "data" / "pose_dataset"
IM_ALL   = POSE_DATA / "images" / "all"
IM_TRAIN = POSE_DATA / "images" / "train"
IM_VAL   = POSE_DATA / "images" / "val"
LB_ALL   = POSE_DATA / "labels" / "all"
LB_TRAIN = POSE_DATA / "labels" / "train"
LB_VAL   = POSE_DATA / "labels" / "val"

# ---- cell ----
DATA_YAML = str(POSE_DATA / "person_pose.yaml")
PROJECT_TRAIN = str(Path(BASE) / "exp_pose_train")

finetune_model = YOLO("yolov8s-pose.pt")
train_res = finetune_model.train(
    data=DATA_YAML, imgsz=640, epochs=20, batch=16, device=0,
    project=PROJECT_TRAIN, name="y8s_pose_finetuned", verbose=False
)
best_weights = Path(train_res.save_dir) / "weights" / "best.pt"
best_weights


# ---- cell ----
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

PROJECT_VAL = Path(BASE) / "exp_pose_eval"
PROJECT_VAL.mkdir(parents=True, exist_ok=True)

models_to_eval = [
    ("y8s_pose_pretrained", "yolov8s-pose.pt"),
    ("y8s_pose_finetuned", str(best_weights)),
]

rows = []
for tag, w in models_to_eval:
    r = YOLO(w).val(
        data=str(POSE_DATA / "person_pose.yaml"),
        imgsz=640, plots=True, save_json=True,
        project=str(PROJECT_VAL), name=f"val_{tag}", verbose=False
    )
    d = r.results_dict
    rows.append({
        "model": tag,
        "mAP50-95(OKS)": d.get("metrics/mAP50-95(B)", d.get("metrics/mAP50-95")),
        "mAP50(OKS)":    d.get("metrics/mAP50(B)", d.get("metrics/mAP50")),
        "precision":     d.get("metrics/precision(B)", d.get("precision")),
        "recall":        d.get("metrics/recall(B)", d.get("recall")),
        "inference_ms":  d.get("speed/inference")
    })

df_pose = pd.DataFrame(rows)
df_pose


# ---- cell ----
import pandas as pd, matplotlib.pyplot as plt, glob

csv_path = sorted(glob.glob(f"{PROJECT_TRAIN}/y8n_pose_finetuned/results.csv"))[-1]
df = pd.read_csv(csv_path)

for col in ["train/box_loss","train/kobj_loss","train/keypoint_loss",
            "metrics/mAP50(B)","metrics/mAP50-95(B)"]:
    if col in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df[col])
        plt.title(col); plt.xlabel("Epoch"); plt.tight_layout(); plt.show()


# ---- cell ----
import matplotlib.pyplot as plt

plt.figure()
plt.bar(df_pose["model"], df_pose["mAP50-95(OKS)"])
plt.ylabel("OKS mAP@50:95")
plt.title("Pose Model Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# ---- cell ----
# --- CONFIG ---
HORIZON_FRAMES = 2       # predict ~0.07s ahead @ 30FPS
HISTORY_FRAMES = 12      # past frames stored per fighter
EMA_ALPHA = 0.25         # smoothing for probabilities
WARMUP_FRAMES = 15       # frames to decide Red/Blue (left/right)
FPS_FALLBACK = 30.0

# Colors (BGR)
C_RED   = (0, 0, 255)
C_BLUE  = (255, 0, 0)
C_GRAY  = (160, 160, 160)

# COCO-17 skeleton edges (Ultralytics order)
COCO_EDGES = [
    (5,7),(7,9), (6,8),(8,10),
    (11,13),(13,15), (12,14),(14,16),
    (5,6), (11,12), (5,11),(6,12),
    (0,1),(1,2),(2,3),(3,4)
]

def draw_skeleton(frame, kpts_xy, color, radius=3, thick=2):
    if kpts_xy is None: return
    for a,b in COCO_EDGES:
        if a < len(kpts_xy) and b < len(kpts_xy):
            xa,ya = kpts_xy[a]; xb,yb = kpts_xy[b]
            if xa>0 and ya>0 and xb>0 and yb>0:
                cv2.line(frame,(int(xa),int(ya)),(int(xb),int(yb)), color, thick, cv2.LINE_AA)
    for (x,y) in kpts_xy:
        if x>0 and y>0:
            cv2.circle(frame,(int(x),int(y)), radius, color, -1, cv2.LINE_AA)

# dashed skeleton overlay
def draw_dashed_line(img, p1, p2, color, thickness=2, dash=10, gap=7):

    h, w = img.shape[:2]
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])

    # Clamp points to image boundaries
    x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1))
    x2 = max(0, min(w-1, x2)); y2 = max(0, min(h-1, y2))

    dist = int(np.hypot(x2 - x1, y2 - y1))
    if dist <= 0:
        return

    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist

    n = 0
    while n * (dash + gap) < dist:
        start = n * (dash + gap)
        end = min(start + dash, dist)

        xs = int(x1 + dx * start); ys = int(y1 + dy * start)
        xe = int(x1 + dx * end);   ye = int(y1 + dy * end)

        # Clip each segment before drawing
        xs = max(0, min(w-1, xs)); ys = max(0, min(h-1, ys))
        xe = max(0, min(w-1, xe)); ye = max(0, min(h-1, ye))

        cv2.line(img, (xs, ys), (xe, ye), color, thickness, cv2.LINE_AA)
        n += 1

        n += 1

def draw_skeleton_dashed(frame, kpts_xy, color, joint_radius=3, thick=2, dash=10, gap=7):
    if kpts_xy is None: return
    for a,b in COCO_EDGES:
        if a < len(kpts_xy) and b < len(kpts_xy):
            xa,ya = kpts_xy[a]; xb,yb = kpts_xy[b]
            if xa>0 and ya>0 and xb>0 and yb>0:
                draw_dashed_line(frame, (xa,ya), (xb,yb), color, thickness=thick, dash=dash, gap=gap)
    for (x,y) in kpts_xy:
        if x>0 and y>0:
            cv2.circle(frame,(int(x),int(y)), joint_radius+1, color, 2, cv2.LINE_AA)

class EMA:
    def __init__(self, alpha=0.25, init_vec=None):
        self.alpha = alpha
        self.state = None if init_vec is None else np.array(init_vec, dtype=float)
    def update(self, vec):
        v = np.array(vec, dtype=float)
        if self.state is None: self.state = v
        else: self.state = self.alpha*v + (1.0-self.alpha)*self.state
        return self.state.copy()

def softmax(z):
    z = np.array(z, dtype=float); z -= np.max(z); e = np.exp(z)
    return e / (np.sum(e) + 1e-9)

def normalize01(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))

# torso helpers (for fitting predicted pose)
def torso_center(kpts):
    if kpts is None: return None
    idxs = [5,6,11,12]
    pts = [kpts[i] for i in idxs if i < len(kpts)]
    pts = np.array([p for p in pts if p is not None])
    if len(pts) == 0: return None
    return np.nanmean(pts, axis=0)

def align_pose_to_torso(curr, fut):
    if curr is None or fut is None: return fut
    c_curr, c_fut = torso_center(curr), torso_center(fut)
    if c_curr is None or c_fut is None: return fut
    shift = c_curr - c_fut
    out = fut.copy(); out += shift
    return out


# ---- cell ----
def kpt_speed(curr, prev):
    if curr is None or prev is None: return 0.0
    diffs = curr - prev
    return float(np.linalg.norm(diffs, axis=1).mean())

def height_rel(kpt, ref_y):
    if kpt is None: return 0.0
    return float(max(0.0, ref_y - kpt[1]))  # above hips (px)

def heuristic_probs(self_hist, opp_hist, frame_h):
    curr = self_hist[-1] if len(self_hist)>0 else None
    prev = self_hist[-2] if len(self_hist)>1 else None
    opp  = opp_hist[-1] if (opp_hist and len(opp_hist)>0) else None

    def pt(k, i):
        return None if (k is None or i>=len(k)) else k[i]

    lw, rw = pt(curr, 9),  pt(curr,10)   # wrists
    la, ra = pt(curr,15), pt(curr,16)    # ankles
    lh, rh = pt(curr,11), pt(curr,12)    # hips
    nose   = pt(curr,0)

    v_all = kpt_speed(curr, prev)
    opp_head = opp[0] if opp is not None else None

    def reach_prob(hand):
        if hand is None or opp_head is None: return 0.0
        d = np.linalg.norm(hand - opp_head)
        return 1.0 - normalize01(d, 0.05*frame_h, 0.6*frame_h)

    p_punch = max(reach_prob(lw), reach_prob(rw)) * normalize01(v_all, 1.0, 25.0)

    hip_y  = np.nanmean([rh[1] if rh is not None else frame_h, lh[1] if lh is not None else frame_h])
    foot_h = max(height_rel(la, hip_y), height_rel(ra, hip_y))
    p_kick = normalize01(foot_h, 0.0, 60.0) * normalize01(v_all, 1.0, 25.0)

    def near_face(hand):
        if hand is None or nose is None: return 0.0
        d = np.linalg.norm(hand - nose)
        return 1.0 - normalize01(d, 0.03*frame_h, 0.25*frame_h)
    guard   = max(near_face(lw), near_face(rw))
    p_block = guard * (1.0 - normalize01(v_all, 10.0, 35.0))

    logits = np.array([p_kick, p_punch, p_block, 0.3*(1.0 - max(p_kick,p_punch,p_block)) + 0.05])
    probs  = softmax(logits * 3.0)
    return {"kick": float(probs[0]), "punch": float(probs[1]), "block": float(probs[2]), "none": float(probs[3])}


# ---- cell ----
def predict_future_pose(kpts_hist, horizon=HORIZON_FRAMES, dt=1.0):
    if len(kpts_hist) < 3 or kpts_hist[-1] is None or kpts_hist[-2] is None:
        return None
    p2 = kpts_hist[-1]; p1 = kpts_hist[-2]
    v  = (p2 - p1) / max(dt, 1e-9)
    return p2 + v * horizon

HEAD_FACE = [0,1,2,3,4]   # nose, eyes, ears
TORSO     = [5,6,11,12]   # shoulders & hips

def blend_future_pose(curr, fut, head_blend=0.85):
    if curr is None or fut is None:
        return fut if fut is not None else curr
    fut = fut.copy(); curr = curr.copy()
    for idx in TORSO:
        if idx < len(fut) and idx < len(curr):
            fut[idx] = curr[idx]
    for idx in HEAD_FACE:
        if idx < len(fut) and idx < len(curr):
            fut[idx] = head_blend * curr[idx] + (1.0 - head_blend) * fut[idx]
    return fut


# ---- cell ----
def draw_probs_overlay(frame, probs_by_id, id_to_role, margin=12):

    h, w = frame.shape[:2]
    x_right = w - margin
    y = margin + 10
    bar_height = 16
    bar_length = 150
    line_gap = 28

    # Title
    title = "Probabilities"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, title, (x_right - tw, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    y += line_gap

    # Order fighters consistently (Red then Blue)
    order = []
    for tid in probs_by_id.keys():
        role = id_to_role.get(tid, "Fighter")
        order.append((0 if role=="Red Fighter" else 1, tid))
    order.sort()

    for _, tid in order:
        role = id_to_role.get(tid, "Fighter")
        color = C_RED if role=="Red Fighter" else C_BLUE

        # Fighter header
        label = f"{role} (ID {tid})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(frame, label, (x_right - tw, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
        y += line_gap

        # Draw 3 bars: Kick, Punch, Block
        for action in ["kick", "punch", "block"]:
            p = float(probs_by_id[tid][action])
            bar_w = int(bar_length * p)
            x1 = x_right - bar_length
            x2 = x_right

            # Rail
            cv2.line(frame, (x1, y), (x2, y), (50,50,50), 2)
            # Value bar
            cv2.line(frame, (x2 - bar_w, y), (x2, y), color, 4)

            # Text
            txt = f"{action.capitalize()} {p:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(frame, txt, (x1 - tw - 8, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
            y += bar_height + 8

        y += 6  # gap after fighter


# ---- cell ----
from collections import defaultdict, deque

def run_overlay_video(
    src_path,
    save_prefix="overlay",
    conf=0.35, iou=0.45,
    tracker="bytetrack.yaml"
):
    src = Path(src_path)
    name = src.stem.strip().replace(" ", "_")

    # probe
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    if fps <= 1: fps = FPS_FALLBACK
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap.release()

    # writer
    out_path = OUT_DIR /"Improvements"/ f"{save_prefix}_{name}_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        out_path = OUT_DIR / f"{save_prefix}_{name}_overlay.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError("VideoWriter open failed.")

    # state
    id_to_hist = defaultdict(lambda: deque(maxlen=HISTORY_FRAMES))
    id_to_prob_ema = defaultdict(lambda: EMA(EMA_ALPHA, [0.12,0.12,0.12]))
    id_to_role = {}
    warm_pos = defaultdict(list)
    frame_idx = 0

    # stream
    gen = pose_model.track(
        source=str(src), conf=conf, iou=iou, classes=[PERSON_CLASS_ID],
        tracker=tracker, save=False, stream=True, verbose=False
    )

    for r in gen:
        frame = r.orig_img.copy() if hasattr(r, "orig_img") else r.plot()
        fh, fw = frame.shape[:2]

        # detections
        ids, boxes = [], []
        if r.boxes is not None:
            if r.boxes.id is not None:
                ids = [int(x) for x in r.boxes.id.cpu().numpy().tolist()]
            if r.boxes.xyxy is not None:
                boxes = r.boxes.xyxy.cpu().numpy().tolist()
        # keypoints (n,17,2)
        if hasattr(r, "keypoints") and r.keypoints is not None:
            try:
                kps = r.keypoints.xy.cpu().numpy()
            except:
                kps = r.keypoints.data[..., :2].cpu().numpy()
        else:
            kps = []

        # histories
        cur_by_id = {}
        for i, tid in enumerate(ids):
            this_k = kps[i] if i < len(kps) else None
            id_to_hist[tid].append(this_k)
            cur_by_id[tid] = this_k

        # roles (left/right after warmup)
        if frame_idx < WARMUP_FRAMES:
            for i, tid in enumerate(ids):
                if i < len(boxes) and boxes[i] is not None:
                    x1,y1,x2,y2 = boxes[i]
                    warm_pos[tid].append(0.5*(x1+x2))
        if frame_idx == WARMUP_FRAMES and not id_to_role:
            means = [(tid, np.mean(xs)) for tid, xs in warm_pos.items() if len(xs)>0]
            means.sort(key=lambda t: t[1])  # left→right
            if len(means) >= 1: id_to_role[means[0][0]] = "Red Fighter"
            if len(means) >= 2: id_to_role[means[1][0]] = "Blue Fighter"

        # per-id probs & predicted pose (fitted)
        ids_sorted = list(cur_by_id.keys())
        probs_for_overlay, fut_fitted = {}, {}
        for tid in ids_sorted:
            opp_tid = None
            if len(ids_sorted) > 1:
                opp_tid = ids_sorted[1] if ids_sorted[0]==tid else ids_sorted[0]

            self_hist = id_to_hist[tid]
            opp_hist  = id_to_hist[opp_tid] if opp_tid is not None else deque(maxlen=HISTORY_FRAMES)

            p = heuristic_probs(self_hist, opp_hist, fh)
            sm = id_to_prob_ema[tid].update([p["kick"], p["punch"], p["block"]])
            probs_for_overlay[tid] = {"kick": float(sm[0]), "punch": float(sm[1]), "block": float(sm[2])}

            fut = predict_future_pose(self_hist, horizon=HORIZON_FRAMES)
            fitted = blend_future_pose(cur_by_id.get(tid), fut, head_blend=0.85)
            fitted = align_pose_to_torso(cur_by_id.get(tid), fitted)
            fut_fitted[tid] = fitted

        # draw current boxes/labels/skeletons
        for i, tid in enumerate(ids):
            role = id_to_role.get(tid, "Fighter")
            col  = C_RED if role=="Red Fighter" else (C_BLUE if role=="Blue Fighter" else (0,255,0))
            if i < len(boxes):
                x1,y1,x2,y2 = map(int, boxes[i])
                cv2.rectangle(frame,(x1,y1),(x2,y2), col, 2)
                cv2.putText(frame,f"{role} (ID {tid})",(x1, max(20,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2,cv2.LINE_AA)
            if i < len(kps):
                draw_skeleton(frame, kps[i], col, radius=3, thick=2)

        # draw predicted (dashed) on top
        for tid, fut in fut_fitted.items():
            role = id_to_role.get(tid, "Fighter")
            col  = C_RED if role=="Red Fighter" else (C_BLUE if role=="Blue Fighter" else (0,180,0))
            if fut is not None:
                draw_skeleton_dashed(frame, fut, col, joint_radius=3, thick=2, dash=10, gap=7)

        # top-right probabilities
        draw_probs_overlay(frame, probs_for_overlay, id_to_role, margin=12)

        writer.write(frame)
        frame_idx += 1

    writer.release()
    print("Saved overlay video:", out_path)
    return out_path


# ---- cell ----
overlay_outputs = []
for v in videos:
    overlay_outputs.append(run_overlay_video(v, save_prefix="tri", tracker="bytetrack.yaml"))
overlay_outputs[:2]

