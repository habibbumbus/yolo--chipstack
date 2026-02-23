import sys


def main() -> None:
    packages = ["torch", "ultralytics", "cv2", "numpy", "yaml"]
    print("Python:", sys.version)
    for package in packages:
        try:
            __import__(package)
            print(f"[OK] import {package}")
        except Exception as exc:  # pragma: no cover
            print(f"[FAIL] import {package}: {exc}")

    try:
        import torch

        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
    except Exception as exc:  # pragma: no cover
        print("[WARN] torch details unavailable:", exc)

    try:
        from ultralytics import YOLO  # noqa: F401

        print("[OK] ultralytics YOLO import")
    except Exception as exc:  # pragma: no cover
        print("[FAIL] ultralytics YOLO:", exc)


if __name__ == "__main__":
    main()

