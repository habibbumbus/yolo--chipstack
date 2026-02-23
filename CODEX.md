# YOLOv8s Chip/Chipstack Training Project (Environment + Scripts Only)

Project root: `C:\yolo_chipstack`  
Dataset source (already prepared by RF-DETR project): `C:\rf-detr_chipstack\data\splits\{train,val,test}\...`  
Goal: Train **YOLOv8s** on the same splits, **imgsz=616**, **35 epochs**, with scripts for training + inference.  
IMPORTANT: Create environment + scripts, but **do not execute training yet**.

---

## 0) What YOLO expects (don’t mess this up)

Ultralytics YOLO does NOT train directly from COCO JSON paths like DETR.
It expects a **YOLO dataset format**:
- images in folders
- labels in `.txt` files (YOLO bbox format)
- a dataset YAML that points to those folders

So we will:
1) Convert your split COCO JSONs → YOLO labels  
2) Create `data_yolo\{train,val,test}\images` + `labels`  
3) Write a `configs\chipstack.yaml` dataset config  
4) Provide `scripts\train_yolo.py` and `scripts\infer_yolo.py`

---

## 1) Folder structure to create

Create this structure:

- `C:\yolo_chipstack\`
  - `CODEX.md` (this file)
  - `.venv\`
  - `configs\`
    - `chipstack.yaml`
  - `data_yolo\`
    - `train\images\`
    - `train\labels\`
    - `val\images\`
    - `val\labels\`
    - `test\images\`
    - `test\labels\`
  - `scripts\`
    - `check_env.py`
    - `coco_to_yolo_splits.py`
    - `train_yolo.py`
    - `infer_yolo.py`
  - `runs\` (YOLO will populate runs\detect\...)

---

## 2) Environment setup (PowerShell) — create venv + install deps

```powershell
cd C:\yolo_chipstack

# create venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

# install ultralytics + utils
python -m pip install ultralytics opencv-python numpy pyyaml tqdm

# OPTIONAL: if torch isn't installed or CUDA isn't working, install torch per pytorch.org
# (Do not run blindly; choose the correct cuXXX index URL for your system)
# python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

3) Dependency sanity check script (Codex can run this)

Create scripts\check_env.py:

import sys

def main():
    pkgs = ["torch", "ultralytics", "cv2", "numpy", "yaml"]
    print("Python:", sys.version)
    for p in pkgs:
        try:
            __import__(p if p != "cv2" else "cv2")
            print(f"[OK] import {p}")
        except Exception as e:
            print(f"[FAIL] import {p}: {e}")

    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("[WARN] torch details unavailable:", e)

    try:
        from ultralytics import YOLO
        print("[OK] ultralytics YOLO import")
    except Exception as e:
        print("[FAIL] ultralytics YOLO:", e)

if __name__ == "__main__":
    main()

Run (safe, no training):

cd C:\yolo_chipstack
.\.venv\Scripts\Activate.ps1
python scripts\check_env.py
4) Convert COCO splits → YOLO labels + mirrored dataset folders

Your COCO split sources are:

Train:

images: C:\rf-detr_chipstack\data\splits\train\images

ann: C:\rf-detr_chipstack\data\splits\train\annotations.json

Val:

images: C:\rf-detr_chipstack\data\splits\val\images

ann: C:\rf-detr_chipstack\data\splits\val\annotations.json

Test:

images: C:\rf-detr_chipstack\data\splits\test\images

ann: C:\rf-detr_chipstack\data\splits\test\annotations.json

4.1 Create scripts\coco_to_yolo_splits.py

This script:

copies images into C:\yolo_chipstack\data_yolo\<split>\images

writes YOLO label .txt into ...\labels

uses COCO bbox [x,y,w,h] and converts to YOLO normalized [cx,cy,w,h]

maps COCO category_id → contiguous 0..(nc-1) class ids

writes configs\chipstack.yaml automatically with correct class names

import argparse
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import yaml

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def coco_categories(coco):
    cats = coco.get("categories", [])
    # sort by id for stability
    cats = sorted(cats, key=lambda c: c["id"])
    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in cats}
    # map COCO category_id -> YOLO class index 0..N-1
    cat_ids = [c["id"] for c in cats]
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    names = [cat_id_to_name[cid] for cid in cat_ids]
    return cat_id_to_idx, names

def index_images(coco):
    # id -> record
    return {im["id"]: im for im in coco.get("images", [])}

def annotations_by_image(coco):
    by = defaultdict(list)
    for ann in coco.get("annotations", []):
        by[ann["image_id"]].append(ann)
    return by

def yolo_line_from_coco_bbox(bbox, img_w, img_h):
    # coco bbox: [x_min, y_min, width, height] in pixels
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    # normalize
    return cx / img_w, cy / img_h, w / img_w, h / img_h

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def convert_split(split_name, images_dir: Path, ann_path: Path, out_root: Path, cat_id_to_idx: dict):
    coco = load_json(ann_path)
    imgs = index_images(coco)
    anns_by_img = annotations_by_image(coco)

    out_images = out_root / split_name / "images"
    out_labels = out_root / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # copy images + write labels
    for img_id, rec in tqdm(imgs.items(), desc=f"Converting {split_name}"):
        file_name = Path(rec["file_name"]).name
        src = images_dir / file_name
        if not src.exists():
            raise FileNotFoundError(f"Missing image: {src}")

        # copy image
        safe_copy(src, out_images / file_name)

        # labels
        w = rec["width"]
        h = rec["height"]
        lines = []
        for ann in anns_by_img.get(img_id, []):
            cid = ann["category_id"]
            if cid not in cat_id_to_idx:
                # if categories differ across splits, that's a dataset bug
                raise ValueError(f"Unknown category_id={cid} in {ann_path}")
            cls = cat_id_to_idx[cid]
            bbox = ann["bbox"]
            cx, cy, bw, bh = yolo_line_from_coco_bbox(bbox, w, h)

            # clamp slightly to [0,1] to avoid tiny numeric drift
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)

            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = out_labels / (Path(file_name).stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def write_dataset_yaml(yaml_path: Path, data_root: Path, names: list[str]):
    y = {
        "path": str(data_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: n for i, n in enumerate(names)},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf_root", type=str, default=r"C:\rf-detr_chipstack")
    ap.add_argument("--out_root", type=str, default=r"C:\yolo_chipstack\data_yolo")
    ap.add_argument("--out_yaml", type=str, default=r"C:\yolo_chipstack\configs\chipstack.yaml")
    args = ap.parse_args()

    rf_root = Path(args.rf_root)
    out_root = Path(args.out_root)
    out_yaml = Path(args.out_yaml)

    # Use train split categories as canonical
    train_ann = rf_root / r"data\splits\train\annotations.json"
    if not train_ann.exists():
        raise FileNotFoundError(f"Missing: {train_ann}")

    train_coco = load_json(train_ann)
    cat_id_to_idx, names = coco_categories(train_coco)

    # Convert all splits
    splits = {
        "train": (rf_root / r"data\splits\train\images", rf_root / r"data\splits\train\annotations.json"),
        "val":   (rf_root / r"data\splits\val\images",   rf_root / r"data\splits\val\annotations.json"),
        "test":  (rf_root / r"data\splits\test\images",  rf_root / r"data\splits\test\annotations.json"),
    }

    for split, (img_dir, ann_path) in splits.items():
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {img_dir}")
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing ann json: {ann_path}")
        convert_split(split, img_dir, ann_path, out_root, cat_id_to_idx)

    # Write dataset yaml for ultralytics
    write_dataset_yaml(out_yaml, out_root, names)

    print("\nDONE.")
    print("YOLO dataset root:", out_root)
    print("Dataset YAML:", out_yaml)
    print("Classes:", names)

if __name__ == "__main__":
    main()
4.2 Run conversion (safe, no training)
cd C:\yolo_chipstack
.\.venv\Scripts\Activate.ps1

python scripts\coco_to_yolo_splits.py --rf_root "C:\rf-detr_chipstack" --out_root "C:\yolo_chipstack\data_yolo" --out_yaml "C:\yolo_chipstack\configs\chipstack.yaml"
5) Training script (YOLOv8s) — create but DO NOT RUN

Create scripts\train_yolo.py:

import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=r"C:\yolo_chipstack\configs\chipstack.yaml")
    ap.add_argument("--model", type=str, default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=616)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=16)   # adjust down if VRAM OOM
    ap.add_argument("--device", type=str, default="0") # "0" = first GPU, or "cpu"
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--project", type=str, default=r"C:\yolo_chipstack\runs")
    ap.add_argument("--name", type=str, default="yolov8s_chipstack_616")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    from ultralytics import YOLO

    model = YOLO(args.model)

    # NOTE: This call runs training. Do not execute this script until you are ready.
    model.train(
        data=str(data_path),
        imgsz=int(args.imgsz),
        epochs=int(args.epochs),
        batch=int(args.batch),
        device=args.device,
        workers=int(args.workers),
        project=str(args.project),
        name=str(args.name),
        # keep defaults for now; you can lock augmentations later for fairness
    )

if __name__ == "__main__":
    main()

When ready later (DO NOT RUN NOW):

python scripts\train_yolo.py --data C:\yolo_chipstack\configs\chipstack.yaml --model yolov8s.pt --imgsz 616 --epochs 35
6) Inference script (YOLOv8s) — post-training

Create scripts\infer_yolo.py:

import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to best.pt or last.pt")
    ap.add_argument("--source", type=str, required=True, help="Image/video/folder")
    ap.add_argument("--imgsz", type=int, default=616)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default=r"C:\yolo_chipstack\runs")
    ap.add_argument("--name", type=str, default="infer")
    ap.add_argument("--save_txt", action="store_true")
    ap.add_argument("--save_crop", action="store_true")
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.exists():
        raise FileNotFoundError(f"Weights not found: {w}")
    s = Path(args.source)
    if not s.exists():
        raise FileNotFoundError(f"Source not found: {s}")

    from ultralytics import YOLO
    model = YOLO(str(w))

    # This runs inference and writes outputs under runs\detect\...
    model.predict(
        source=str(s),
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        device=args.device,
        project=str(args.project),
        name=str(args.name),
        save=True,
        save_txt=bool(args.save_txt),
        save_crop=bool(args.save_crop),
    )

if __name__ == "__main__":
    main()

Example (after training):

python scripts\infer_yolo.py --weights C:\yolo_chipstack\runs\detect\yolov8s_chipstack_616\weights\best.pt --source C:\some_image_or_video.mp4 --imgsz 616 --conf 0.25
7) Quick sanity checks (no training)
cd C:\yolo_chipstack
.\.venv\Scripts\Activate.ps1

python scripts\check_env.py

# verify dataset yaml exists
type C:\yolo_chipstack\configs\chipstack.yaml

# verify a few labels exist
dir C:\yolo_chipstack\data_yolo\train\labels | select -first 5
dir C:\yolo_chipstack\data_yolo\train\images | select -first 5
Notes (important)

Keep imgsz=616 for YOLO to match RF-DETR’s training resolution.

Batch default is set to 16, but on an RTX 3050 4GB you might need --batch 8 or --batch 4.

This setup keeps your RF-DETR split as the “source of truth,” and builds a separate YOLO-ready dataset in C:\yolo_chipstack\data_yolo\....

Training is not executed unless you explicitly run scripts\train_yolo.py.