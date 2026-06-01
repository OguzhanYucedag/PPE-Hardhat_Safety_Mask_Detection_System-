from pathlib import Path
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
model = YOLO(ROOT / "runs/detect/KKD_Projesi/ilk_egitim/weights/best.pt")

st.title("KKD (PPE) Tespit Sistemi")
conf = st.slider("Güven eşiği", 0.1, 1.0, 0.5)
uploaded = st.file_uploader("Resim yükle", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    results = model.predict(img, conf=conf, verbose=False)
    annotated = results[0].plot()
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Tespit sonucu")

    for box in results[0].boxes:
        cls = int(box.cls[0])
        st.write(f"- {model.names[cls]}: %{float(box.conf[0])*100:.1f}")