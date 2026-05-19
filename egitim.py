from ultralytics import YOLO

# 1. Adım: YOLOv8'in 'nano' modelini (en hafif ve en hızlı versiyon) indir/yükle
model = YOLO('yolov8n.pt')

print("Eğitim başlıyor... Kemerleri bağlayın!")

# 2. Adım: Eğitimi başlat
results = model.train(
    data='dataset/data.yaml',  # Yol haritamız olan konfigürasyon dosyası
    epochs=25,                 # Modelin veri setini kaç kere baştan sona çalışacağı (Tur sayısı)
    imgsz=640,                 # Resimlerin yeniden boyutlandırılma ölçüsü (YOLO standardı)
    batch=16,                  # Aynı anda GPU'ya gönderilecek resim sayısı
    device='mps',              # Mac M1/M2/M3 gücünü (Apple Silicon) kullan!
    project='KKD_Projesi',     # Sonuçların kaydedileceği ana klasör
    name='ilk_egitim'          # Bu eğitime verdiğimiz isim
)

print("Eğitim başarıyla tamamlandı!")