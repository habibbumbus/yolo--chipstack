import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
import yaml


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def coco_categories(coco: dict) -> tuple[dict[int, int], list[str]]:
    categories = sorted(coco.get("categories", []), key=lambda item: item["id"])
    cat_id_to_name = {item["id"]: item.get("name", str(item["id"])) for item in categories}
    cat_ids = [item["id"] for item in categories]
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    names = [cat_id_to_name[cat_id] for cat_id in cat_ids]
    return cat_id_to_idx, names


def index_images(coco: dict) -> dict[int, dict]:
    return {image["id"]: image for image in coco.get("images", [])}


def annotations_by_image(coco: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        grouped[ann["image_id"]].append(ann)
    return grouped


def yolo_line_from_coco_bbox(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x_min, y_min, width, height = bbox
    cx = x_min + width / 2.0
    cy = y_min + height / 2.0
    return cx / img_w, cy / img_h, width / img_w, height / img_h


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def clear_split_output(out_root: Path, split_name: str) -> None:
    split_root = out_root / split_name
    if split_root.exists():
        shutil.rmtree(split_root)


def convert_split(
    split_name: str,
    images_dir: Path,
    ann_path: Path,
    out_root: Path,
    cat_id_to_idx: dict[int, int],
    clean_split: bool,
) -> None:
    if clean_split:
        clear_split_output(out_root, split_name)

    coco = load_json(ann_path)
    images_by_id = index_images(coco)
    anns_by_image = annotations_by_image(coco)

    out_images = out_root / split_name / "images"
    out_labels = out_root / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for img_id, image_record in tqdm(images_by_id.items(), desc=f"Converting {split_name}"):
        file_name = Path(image_record["file_name"]).name
        src_image = images_dir / file_name
        if not src_image.exists():
            raise FileNotFoundError(f"Missing image: {src_image}")

        safe_copy(src_image, out_images / file_name)

        img_w = image_record["width"]
        img_h = image_record["height"]
        label_lines: list[str] = []
        for ann in anns_by_image.get(img_id, []):
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_idx:
                raise ValueError(f"Unknown category_id={cat_id} in {ann_path}")
            cls_id = cat_id_to_idx[cat_id]
            cx, cy, bw, bh = yolo_line_from_coco_bbox(ann["bbox"], img_w, img_h)
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = out_labels / (Path(file_name).stem + ".txt")
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")


def write_dataset_yaml(yaml_path: Path, data_root: Path, names: list[str]) -> None:
    payload = {
        "path": str(data_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {idx: name for idx, name in enumerate(names)},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf_root", type=str, default=r"C:\rf-detr_chipstack")
    parser.add_argument("--out_root", type=str, default=r"C:\yolo_chipstack\data_yolo")
    parser.add_argument("--out_yaml", type=str, default=r"C:\yolo_chipstack\configs\chipstack.yaml")
    parser.add_argument(
        "--no_clean",
        action="store_true",
        help="Do not delete existing converted split folders before conversion.",
    )
    args = parser.parse_args()

    rf_root = Path(args.rf_root)
    out_root = Path(args.out_root)
    out_yaml = Path(args.out_yaml)
    clean_split = not args.no_clean

    train_ann = rf_root / r"data\splits\train\annotations.json"
    if not train_ann.exists():
        raise FileNotFoundError(f"Missing: {train_ann}")

    train_coco = load_json(train_ann)
    cat_id_to_idx, names = coco_categories(train_coco)

    splits: dict[str, tuple[Path, Path]] = {
        "train": (rf_root / r"data\splits\train\images", rf_root / r"data\splits\train\annotations.json"),
        "val": (rf_root / r"data\splits\val\images", rf_root / r"data\splits\val\annotations.json"),
        "test": (rf_root / r"data\splits\test\images", rf_root / r"data\splits\test\annotations.json"),
    }

    for split_name, (images_dir, ann_path) in splits.items():
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {images_dir}")
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing ann json: {ann_path}")
        convert_split(split_name, images_dir, ann_path, out_root, cat_id_to_idx, clean_split=clean_split)

    write_dataset_yaml(out_yaml, out_root, names)

    print("\nDONE.")
    print("YOLO dataset root:", out_root)
    print("Dataset YAML:", out_yaml)
    print("Classes:", names)


if __name__ == "__main__":
    main()

