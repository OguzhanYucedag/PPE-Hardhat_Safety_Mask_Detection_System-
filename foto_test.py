from pathlib import Path

import cv2
from ultralytics import YOLO

# 1. Adım: Eğittiğimiz en zeki beyni (best.pt) projeye çağırıyoruz.
# egitim.py'deki project='KKD_Projesi', name='ilk_egitim' -> runs/detect/KKD_Projesi/ilk_egitim/
ROOT = Path(__file__).resolve().parent
CIKTI_KLASORU = ROOT / "runs" / "detect" / "predict"
model_path = ROOT / "runs/detect/KKD_Projesi/ilk_egitim/weights/best.pt"
model = YOLO(model_path)

# 2. Adım: Test etmek istediğimiz resmin adını veriyoruz
test_edilecek_resim = ROOT / "test" / "test_resmi03.jpg"
if not test_edilecek_resim.exists():
    raise FileNotFoundError(f"Test resmi bulunamadı: {test_edilecek_resim}")

print("Model resme bakıyor, tespitler yapılıyor...")

# 3. Adım: Tahmin (Prediction) işlemini başlatıyoruz
# conf=0.5 -> "Eğer bir kutunun baret olduğundan %50'den daha az eminsen onu ekrana çizme" demek.
# save=True -> Çizilmiş halini bilgisayara kaydet (runs/detect/predict/).
# exist_ok=True -> predict-2, predict-3... yerine hep aynı klasöre yazar (aynı dosya adı üzerine yazar).
sonuclar = model.predict(
    source=test_edilecek_resim,
    conf=0.5,
    save=True,
    show=False,
    project=str(CIKTI_KLASORU.parent),
    name=CIKTI_KLASORU.name,
    exist_ok=True,
)
#Ekrana çizilmiş halini gösterme işlemi
pencere_adi = "KKD Tespit Sonucu"
for sonuc in sonuclar:
    cizili_resim = sonuc.plot()
    cv2.imshow(pencere_adi, cizili_resim)
    print("Sonuç ekranda. Kapatmak için pencereyi kapatın veya herhangi bir tuşa basın...")
    while True:
        tus = cv2.waitKey(100)
        if tus != -1:
            break
        try:
            if cv2.getWindowProperty(pencere_adi, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
cv2.destroyAllWindows()
#Sonuçları ekrana yazdırma işlemi
print(f"İşlem tamam! Sonuçlar: {CIKTI_KLASORU}")