from pathlib import Path
from ultralytics import YOLO

# Projenin ana dizin yolunu al
ROOT = Path(__file__).resolve().parent

# İkinci eğitimden elde edilen en iyi ağırlıklar hedef alınır
IKINCI_EGITIM_AGIRLIK = ROOT / "runs/detect/KKD_Projesi/ikinci_egitim/weights/best.pt"
YEDEK_AGIRLIK = ROOT / "runs/detect/KKD_Projesi/ikinci_egitim/weights/last.pt"

# Hangi ağırlık mevcutsa onu seçer
model_path = IKINCI_EGITIM_AGIRLIK if IKINCI_EGITIM_AGIRLIK.exists() else YEDEK_AGIRLIK

if not model_path.exists():
    raise FileNotFoundError(
        f"Üçüncü eğitime temel oluşturacak ikinci eğitim ağırlığı bulunamadı.\n"
        f"  Beklenen konum: {IKINCI_EGITIM_AGIRLIK}\n"
        f"Lütfen runs/detect/ klasöründe ikinci eğitimin tamamlandığından emin ol."
    )

print(f"3. Eğitim (Hassas İnce Ayar) başlıyor... Yüklenecek model: {model_path.name}")

# Modeli ikinci eğitimin başarıyla öğrendiği ağırlıklarla başlatıyoruz
model = YOLO(model_path)

# 3. Eğitimi Çalıştır
results = model.train(
    data=str(ROOT / "dataset/data.yaml"),
    epochs=60,         # 3. eğitim için planlanan maksimum tur sayısı
    patience=15,       # Doğrulama metrikleri 15 tur boyunca gelişmezse eğitimi sonlandır
    imgsz=640,
    batch=16,
    device="mps",      # Apple Silicon GPU hızlandırması aktif
    project="KKD_Projesi",
    name="ucuncu_egitim", # Sonuçlar bu isimle ayrı bir klasöre kaydedilecek
    
    # ÖNEMLİ AYARLAR: İkinci grafikteki sert dalgalanmaları engellemek ve 
    # modeli ürkütmeden eğitmek için başlangıç öğrenme oranını (lr0) çok küçük tutuyoruz.
    lr0=0.0005,        # Varsayılan değer olan 0.01 yerine çok daha küçük adımlar atmasını sağlıyoruz
    lrf=0.01           # Eğitimin sonundaki nihai öğrenme oranı çarpanı
)

print("Üçüncü eğitim başarıyla tamamlandı!")
print(f"Bu eğitimin en iyi modeli: {ROOT / 'runs/detect/KKD_Projesi/ucuncu_egitim/weights/best.pt'}")