import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--source", type=str, required=True, help="Image/video/folder")
    parser.add_argument("--imgsz", type=int, default=616)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default=r"C:\yolo_chipstack\runs")
    parser.add_argument("--name", type=str, default="infer")
    parser.add_argument("--save_txt", action="store_true")
    parser.add_argument("--save_crop", action="store_true")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.predict(
        source=str(source),
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

