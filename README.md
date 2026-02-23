# YOLOv8s Chipstack Training Setup

This repository prepares a YOLOv8s training environment for the Chip/Chipstack dataset split from:
`C:\rf-detr_chipstack\data\splits\{train,val,test}`.

Training is not started by these setup steps.

## Project Layout

- `scripts/check_env.py` validates imports and CUDA visibility.
- `scripts/coco_to_yolo_splits.py` converts COCO split annotations to YOLO labels and writes `configs/chipstack.yaml`.
- `scripts/train_yolo.py` launches training when you are ready.
- `scripts/infer_yolo.py` runs post-training inference.

## Environment Setup (PowerShell)

```powershell
cd C:\yolo_chipstack
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
# GPU-enabled torch build (recommended for RTX 3050)
python -m pip install --force-reinstall torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Safe Checks (No Training)

```powershell
cd C:\yolo_chipstack
.\.venv\Scripts\Activate.ps1
python scripts\check_env.py
python scripts\coco_to_yolo_splits.py --rf_root "C:\rf-detr_chipstack" --out_root "C:\yolo_chipstack\data_yolo" --out_yaml "C:\yolo_chipstack\configs\chipstack.yaml"
```

## Train Command (when ready)

```powershell
python scripts\train_yolo.py --data C:\yolo_chipstack\configs\chipstack.yaml --model yolov8s.pt --imgsz 616 --epochs 35 --batch 16 --device 0 --workers 2 --project C:\yolo_chipstack\runs --name yolov8s_chipstack_616
```

## Resume from Existing Run

Resume from best checkpoint in the same run:

```powershell
python scripts\train_yolo.py --data C:\yolo_chipstack\configs\chipstack.yaml --imgsz 616 --epochs 35 --batch 16 --device 0 --workers 2 --project C:\yolo_chipstack\runs --name yolov8s_chipstack_616 --resume --resume_ckpt best --exist_ok
```

Resume from last checkpoint with full optimizer/scheduler state:

```powershell
python scripts\train_yolo.py --data C:\yolo_chipstack\configs\chipstack.yaml --imgsz 616 --epochs 35 --batch 16 --device 0 --workers 2 --project C:\yolo_chipstack\runs --name yolov8s_chipstack_616 --resume --resume_ckpt last --exist_ok
```

You can also provide an explicit checkpoint path:

```powershell
python scripts\train_yolo.py --data C:\yolo_chipstack\configs\chipstack.yaml --resume_path C:\yolo_chipstack\runs\yolov8s_chipstack_616\weights\best.pt --exist_ok
```
