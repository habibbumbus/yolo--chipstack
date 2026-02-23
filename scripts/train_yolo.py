import argparse
from pathlib import Path


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_path:
        ckpt = Path(args.resume_path)
    elif args.resume:
        run_dir = Path(args.project) / args.name
        ckpt = run_dir / "weights" / f"{args.resume_ckpt}.pt"
    else:
        return None

    if not ckpt.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt}")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=r"C:\yolo_chipstack\configs\chipstack.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=616)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=str, default=r"C:\yolo_chipstack\runs")
    parser.add_argument("--name", type=str, default="yolov8s_chipstack_616")
    parser.add_argument("--exist_ok", action="store_true", help="Allow writing into an existing run folder.")

    parser.add_argument("--resume", action="store_true", help="Resume from an existing run under --project/--name.")
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        choices=["best", "last"],
        default="best",
        help="Checkpoint to use when --resume is enabled.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="",
        help="Explicit checkpoint path. Overrides --resume/--resume_ckpt when set.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    from ultralytics import YOLO

    resume_ckpt = resolve_resume_checkpoint(args)
    model_source = str(resume_ckpt) if resume_ckpt else args.model
    model = YOLO(model_source)

    train_kwargs = {
        "data": str(data_path),
        "imgsz": int(args.imgsz),
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "device": args.device,
        "workers": int(args.workers),
        "project": str(args.project),
        "name": str(args.name),
        "exist_ok": bool(args.exist_ok),
    }

    # Full optimizer/scheduler resume is supported with last.pt.
    # best.pt resume still starts from the best learned weights.
    if resume_ckpt and resume_ckpt.name.lower() == "last.pt":
        train_kwargs["resume"] = True

    # NOTE: This call runs training. Do not execute this script until you are ready.
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()

