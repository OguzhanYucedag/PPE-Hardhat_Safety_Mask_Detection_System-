import os
import shutil
import random

# Klasör yolları (Görseldeki isimlere göre ayarlandı)
kaynak_klasor = "PPE-DETECTION-9"
hedef_klasor = "dataset"

# Hedef alt klasörleri oluştur (Yoksa yaratır)
klasorler = [
    f"{hedef_klasor}/train/images", f"{hedef_klasor}/train/labels",
    f"{hedef_klasor}/val/images", f"{hedef_klasor}/val/labels"
]

for klasor in klasorler:
    os.makedirs(klasor, exist_ok=True)

# Tüm resim ve etiketleri toplayacağımız ana liste
tum_resimler = []

# Roboflow'un oluşturduğu alt klasörleri tara
alt_klasorler = ["train", "valid", "test"]

for alt in alt_klasorler:
    resim_klasoru = os.path.join(kaynak_klasor, alt, "images")
    etiket_klasoru = os.path.join(kaynak_klasor, alt, "labels")

    if os.path.exists(resim_klasoru):
        for resim_adi in os.listdir(resim_klasoru):
            if resim_adi.endswith(('.jpg', '.jpeg', '.png')):
                # Resmin adından yola çıkarak etiketinin adını bul
                etiket_adi = resim_adi.rsplit('.', 1)[0] + '.txt'
                
                resim_yolu = os.path.join(resim_klasoru, resim_adi)
                etiket_yolu = os.path.join(etiket_klasoru, etiket_adi)
                
                # Sadece etiketi (koordinatları) olan eksiksiz resimleri al
                if os.path.exists(etiket_yolu):
                    tum_resimler.append({
                        'resim': resim_yolu,
                        'etiket': etiket_yolu,
                        'resim_adi': resim_adi,
                        'etiket_adi': etiket_adi
                    })

# Modeli ezberden kurtarmak için verileri rastgele karıştır
random.shuffle(tum_resimler)

# Tam olarak %80 ve %20 oranında böl
toplam_veri = len(tum_resimler)
train_siniri = int(toplam_veri * 0.8)

train_verileri = tum_resimler[:train_siniri]
val_verileri = tum_resimler[train_siniri:]

# Dosyaları kopyalama işlemi
def dosyalari_kopyala(veri_listesi, hedef_alt_klasor):
    for veri in veri_listesi:
        shutil.copy(veri['resim'], os.path.join(hedef_klasor, hedef_alt_klasor, "images", veri['resim_adi']))
        shutil.copy(veri['etiket'], os.path.join(hedef_klasor, hedef_alt_klasor, "labels", veri['etiket_adi']))

print(f"Toplam {toplam_veri} eksiksiz veri bulundu. %80 / %20 oranında dağıtılıyor...")

# Verileri yeni evlerine taşı
dosyalari_kopyala(train_verileri, "train")
print(f"✅ {len(train_verileri)} resim ve etiket 'train' klasörüne yerleşti.")

dosyalari_kopyala(val_verileri, "val")
print(f"✅ {len(val_verileri)} resim ve etiket 'val' klasörüne yerleşti.")

print("\nHarika! Veri seti organizasyonu başarıyla tamamlandı.")