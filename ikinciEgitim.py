# from pathlib import Path

# from ultralytics import YOLO

# ROOT = Path(__file__).resolve().parent

# # İlk eğitimden en iyi ağırlıklar (yoksa last.pt kullanılır)
# ILK_EGITIM_AGIRLIK = ROOT / "runs/detect/KKD_Projesi/ilk_egitim/weights/best.pt"
# YEDEK_AGIRLIK = ROOT / "runs/detect/KKD_Projesi/ilk_egitim/weights/last.pt"

# model_path = ILK_EGITIM_AGIRLIK if ILK_EGITIM_AGIRLIK.exists() else YEDEK_AGIRLIK
# if not model_path.exists():
#     raise FileNotFoundError(
#         f"İlk eğitim ağırlığı bulunamadı.\n"
#         f"  Beklenen: {ILK_EGITIM_AGIRLIK}\n"
#         f"Önce egitim.py ile ilk eğitimi tamamlayın."
#     )

# print(f"İkinci eğitim başlıyor... Yüklenecek model: {model_path.name}")

# model = YOLO(model_path)

# results = model.train(
#     data=str(ROOT / "dataset/data.yaml"),
#     epochs=60,       # Üst limit (patience dolunca erken durabilir)
#     patience=15,     # Val metrik 15 tur iyileşmezse dur (ezber riskini azaltır)
#     imgsz=640,
#     batch=16,
#     device="mps",
#     project="KKD_Projesi",
#     name="ikinci_egitim",
# )

# print("İkinci eğitim tamamlandı!")
# print(f"En iyi model: {ROOT / 'runs/detect/KKD_Projesi/ikinci_egitim/weights/best.pt'}")


results = model.train(
    data=str(ROOT / "dataset/data.yaml"),
    epochs=60,       
    patience=15,     
    imgsz=640,
    batch=16,
    device="mps",
    project="KKD_Projesi",
    name="ikinci_egitim_devam",
    resume=True  # Eğitimi tam olarak kaldığı optimizer state ve learning rate ile sürdürür
)