# PPE (KKD) Tespit Sistemi

Kişisel koruyucu donanım (KKD / PPE) öğelerini görüntülerde ve canlı kamera akışında tespit eden bir **YOLOv8** projesidir. Baret, maske, yelek, eldiven, güvenlik botu ve kişi sınıfları için nesne algılama (object detection) yapılır.

## Tespit edilen sınıflar

| Sınıf | Açıklama |
|-------|----------|
| `Hard_hat` | Baret |
| `Mask` | Maske |
| `Vest` | Yelek |
| `Gloves` | Eldiven |
| `Safety_boots` | Güvenlik botu |
| `Person` | Kişi |

## Kullanılan teknolojiler

- [Ultralytics YOLOv8] — model eğitimi ve tahmin
- [OpenCV] — görüntü işleme
- [Streamlit] — web arayüzü
- [streamlit-webrtc]— canlı kamera

## Kurulum

```bash
# Sanal ortam (önerilir)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Bağımlılıklar
pip install -r requirements.txt
```

**Gereksinimler:** Python 3.10+, Mac kullanıcıları için Apple Silicon (`mps`) destekli PyTorch.

## Proje yapısı

```
PPE(Hardhat_Safety_Mask_Detection_System)/
├── dataset/
│   ├── data.yaml          # Veri seti yapılandırması
│   ├── train/             # Eğitim görüntüleri ve etiketleri
│   └── val/               # Doğrulama görüntüleri ve etiketleri
├── test/                  # Manuel test görselleri
├── runs/detect/
│   ├── KKD_Projesi/       # Eğitim çıktıları (ilk_egitim, ikinci_egitim, …)
│   │   └── models/        # Karşılaştırma için kopyalanan best1.pt, best2.pt, …
│   ├── predict/           # foto_test.py tahmin çıktıları
│   └── val, val-2, …      # model.val() rapor klasörleri (silinebilir)
├── ilkEgitim.py           # İlk model eğitimi
├── ikinciEgitim.py        # İkinci eğitim (önceki ağırlıklardan devam)
├── ucuncuEgitim.py         # Üçüncü eğitim (ince ayar)
├── foto_test.py           # Tek resimde masaüstü testi
├── models_test.py         # models/ altındaki modelleri val setinde karşılaştırma
├── web_app.py             # Streamlit web arayüzü
├── veri_indir.py          # Roboflow’dan veri indirme
├── veri_duzenle.py        # Ham veriyi train/val’e bölme
└── requirements.txt
```

## Veri seti

- **Kaynak:** Roboflow PPE Detection (v9)
- **Boyut:** ~1708 eğitim / ~427 doğrulama görüntüsü
- **Format:** YOLOv8

Veri henüz yoksa:

```
python veri_indir.py    # Roboflow API anahtarı gerekir
python veri_duzenle.py  # İndirilen veriyi dataset/ yapısına böler
```



## Model eğitimi

### 1. İlk eğitim

```
python ilkEgitim.py
```

Çıktı: `runs/detect/KKD_Projesi/ilk_egitim/weights/best.pt`

### 2. İkinci eğitim

İlk eğitimden sonra, daha fazla tur ve erken durdurma (`patience`) ile:

```
python ikinciEgitim.py
```

### 3. Üçüncü eğitim

İkinci eğitimin `best.pt` dosyası üzerinden ince ayar:

```
python ucuncuEgitim.py
```

Eğitim metrikleri her klasörde `results.csv` ve `results.png` olarak kaydedilir.

### Modelleri karşılaştırma klasörüne kopyalama

Farklı eğitimlerin `best.pt` dosyalarını şuraya kopyalayın:

```
runs/detect/KKD_Projesi/models/best1.pt
runs/detect/KKD_Projesi/models/best2.pt
runs/detect/KKD_Projesi/models/best3.pt
```

## Model testi ve karşılaştırma

Tüm `models/best*.pt` dosyalarını aynı **validation** setinde değerlendirir; mAP50 ve mAP50-95’e göre sıralar:

```bash
python models_test.py
```

Bu komut `runs/detect/val`, `val-2`, … klasörlerini de oluşturabilir (YOLO doğrulama raporu).

## Tek resim testi (masaüstü)

`foto_test.py` içinde model ve test resmi yolunu düzenleyin, ardından:

```
python foto_test.py
```

Sonuçlar `runs/detect/predict/` altına kaydedilir.

## Web arayüzü

```
streamlit run web_app.py
```



| Sekme | Özellik |
|-------|---------|
| **Resim yükle** | Dosyadan tek görüntü analizi |
| **Canlı kamera** | Web kamerasından akış; model seçilen aralıkta (0.5–5 sn) kare analiz eder |

> Uygulamayı `python web_app.py` ile değil, mutlaka `streamlit run web_app.py` ile başlatın.

Model seçimi: mümkünse eğitim `results.csv` dosyalarından en yüksek mAP50’ye sahip `best.pt` yüklenir; bulunamazsa `models/` ve eğitim klasörlerindeki yedek listeye bakılır.

## `runs/detect/` altındaki klasörler

| Klasör | Açıklama |
|--------|----------|
| `KKD_Projesi/ilk_egitim/` … | Eğitim çıktıları, ağırlıklar, grafikler |
| `KKD_Projesi/models/` | Karşılaştırma için kopyalanan modeller |
| `predict/` | `foto_test.py` tahmin görselleri |
| `val`, `val-2`, `val-3` | `model.val()` veya `models_test.py` raporları (isteğe bağlı silinebilir) |

Bunlar **`dataset/val`** verisinden farklıdır: `dataset/val` ham etiketli veridir; `runs/detect/val*` klasörleri model değerlendirme raporudur.

## Ezberleme (overfitting) kontrolü

- `results.png` içinde train ve val loss eğrileri.
- Val mAP düşerken train loss düşmeye devam ediyorsa eğitimi durdurun (`patience` kullandık).
- Yeni, görülmemiş fotoğraflarla (`test/`) manuel test yaptık.


## Geliştirici notları

- Mac (M1/M2/M3): eğitim scriptlerinde `device='mps'` kullanılır.
- NVIDIA GPU varsa `device='0'` veya `device='cuda'` olarak değiştirilebilir.
- Büyük model dosyaları (`.pt`) Git’e eklenmemiş olabilir; eğitim sonrası yerel olarak oluşur.
