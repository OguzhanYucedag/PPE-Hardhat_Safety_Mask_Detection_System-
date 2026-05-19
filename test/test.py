import torch
import cv2
from ultralytics import YOLO

print("OpenCV Versiyonu:", cv2.__version__)
print("PyTorch Versiyonu:", torch.__version__)

# Mac M1/M2/M3 işlemcilerdeki GPU (MPS) kontrolü
if torch.backends.mps.is_available():
    print("Harika haber! Mac GPU'su (MPS) aktif ve YOLO için kullanıma hazır. Eğitimlerimiz çok daha hızlı olacak.")
else:
    print("MPS bulunamadı, işlemler CPU üzerinden yapılacak.")