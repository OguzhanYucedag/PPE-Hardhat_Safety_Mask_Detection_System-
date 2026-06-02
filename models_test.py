from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


@dataclass(frozen=True)
class ModelScore:
    path: Path
    map50: float
    map5095: float


def find_models_in_models_dir(kkd_project_dir: Path) -> list[Path]:
    """
    Bu projede modelleri elle kopyalayıp şu klasöre koyuyorsun:

      runs/detect/KKD_Projesi/models/best1.pt
      runs/detect/KKD_Projesi/models/best2.pt
      runs/detect/KKD_Projesi/models/best3.pt

    Bu nedenle yalnızca `models/` altındaki `best*.pt` dosyalarını test ediyoruz.
    """

    models_dir = kkd_project_dir / "models"
    if not models_dir.exists():
        return []

    # Deterministik sırada listele (best1, best2, best3 ...)
    return sorted(p.resolve() for p in models_dir.glob("best*.pt") if p.exists())


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def eval_one_model(model_path: Path, data_yaml: Path, imgsz: int = 640) -> ModelScore:
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, split="val", verbose=False)

    # Ultralytics Metrics nesnesi sürüme göre değişebiliyor; birkaç güvenli yol deniyoruz.
    map50 = safe_float(getattr(getattr(metrics, "box", None), "map50", None))
    map5095 = safe_float(getattr(getattr(metrics, "box", None), "map", None))

    # Yedek: results_dict varsa
    if map50 != map50 or map5095 != map5095:  # NaN check
        d = getattr(metrics, "results_dict", None)
        if isinstance(d, dict):
            map50 = safe_float(d.get("metrics/mAP50(B)", map50))
            map5095 = safe_float(d.get("metrics/mAP50-95(B)", map5095))

    return ModelScore(path=model_path, map50=map50, map5095=map5095)


def print_table(rows: Iterable[ModelScore], root: Path) -> None:
    rows = list(rows)
    if not rows:
        print("Model bulunamadı.")
        return

    name_w = max(len(str(r.path.relative_to(root))) for r in rows)
    name_w = max(name_w, len("model"))
    header = f"{'model'.ljust(name_w)}  mAP50   mAP50-95"
    print(header)
    print("-" * len(header))
    for r in rows:
        rel = str(r.path.relative_to(root))
        print(f"{rel.ljust(name_w)}  {r.map50:0.4f}  {r.map5095:0.4f}")


def main() -> int:
    root = Path(__file__).resolve().parent
    data_yaml = root / "dataset/data.yaml"
    kkd_project_dir = root / "runs/detect/KKD_Projesi"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Bulunamadı: {data_yaml}")
    if not kkd_project_dir.exists():
        raise FileNotFoundError(f"Bulunamadı: {kkd_project_dir}")

    candidates = find_models_in_models_dir(kkd_project_dir)
    if not candidates:
        print("Hiç model bulunamadı: runs/detect/KKD_Projesi/models/best*.pt")
        return 1

    print(f"Toplam {len(candidates)} model test edilecek (val setinde).")
    scores: list[ModelScore] = []
    for p in candidates:
        print(f"\nTest ediliyor: {p.relative_to(root)}")
        scores.append(eval_one_model(p, data_yaml=data_yaml, imgsz=640))

    scores_sorted = sorted(scores, key=lambda s: (s.map50, s.map5095), reverse=True)
    print("\nSonuçlar (büyükten küçüğe):")
    print_table(scores_sorted, root=root)

    best = scores_sorted[0]
    print(
        f"\nEn iyi model: {best.path.relative_to(root)} "
        f"(mAP50={best.map50:0.4f}, mAP50-95={best.map5095:0.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

