from ultralytics.models.yolo.model import YOLO


import time
from pathlib import Path
import sys

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO

if get_script_run_ctx() is None:
    print("Bu uygulamayı şu komutla başlatın:\n  streamlit run web_app.py", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
KKD_PROJE = ROOT / "runs/detect/KKD_Projesi"


def en_iyi_modeli_bul():
    en_iyi_skor = -1.0
    en_iyi_model_yolu = None
    model_dosyalari = list(KKD_PROJE.glob("**/weights/best.pt"))

    if not model_dosyalari:
        return None, None

    for model_path in model_dosyalari:
        egitim_klasoru = model_path.parent.parent
        csv_yolu = egitim_klasoru / "results.csv"
        if not csv_yolu.exists():
            continue
        try:
            df = pd.read_csv(csv_yolu)
            df.columns = df.columns.str.strip()
            if "metrics/mAP50(B)" in df.columns:
                en_yuksek_map = df["metrics/mAP50(B)"].max()
            elif "metrics/mAP50" in df.columns:
                en_yuksek_map = df["metrics/mAP50"].max()
            else:
                en_yuksek_map = 0.0
            if en_yuksek_map > en_iyi_skor:
                en_iyi_skor = en_yuksek_map
                en_iyi_model_yolu = model_path
        except Exception:
            pass

    return en_iyi_model_yolu, en_iyi_skor


@st.cache_resource
def load_model():
    en_iyi_yol, skor = en_iyi_modeli_bul()
    if not en_iyi_yol:
        model_adaylari = [
            KKD_PROJE / "ucuncu_egitim/weights/best.pt",
            
        ]
        for path in model_adaylari:
            if path.exists():
                return YOLO(path), f"{path.relative_to(ROOT)} (yedek listeden)"
        raise FileNotFoundError("Hiçbir model dosyası bulunamadı!")
    return (
        YOLO(en_iyi_yol),
        f"{en_iyi_yol.relative_to(ROOT)} (en yüksek mAP50: %{skor * 100:.1f})",
    )


model, model_yolu = load_model()


class KKDVideoProcessor(VideoProcessorBase):
    """Web kamerasından gelen akışta belirli aralıklarla YOLO tahmini yapar."""

    def __init__(self, yolo_model: YOLO, conf: float, interval_sec: float):
        self.yolo_model = yolo_model
        self.conf = conf
        self.interval_sec = interval_sec
        self._son_analiz = 0.0
        self._son_cizim: np.ndarray | None = None
        self.son_tespitler: list[tuple[str, float]] = []
        self.son_analiz_zamani: str = "-"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        simdi = time.time()

        if simdi - self._son_analiz >= self.interval_sec:
            sonuc = self.yolo_model.predict(img, conf=self.conf, verbose=False)[0]
            self._son_cizim = sonuc.plot()
            self._son_analiz = simdi
            self.son_analiz_zamani = time.strftime("%H:%M:%S")

            self.son_tespitler = []
            if sonuc.boxes is not None:
                for box in sonuc.boxes:
                    cls = int(box.cls[0])
                    self.son_tespitler.append(
                        (self.yolo_model.names[cls], float(box.conf[0]))
                    )

        goster = self._son_cizim if self._son_cizim is not None else img
        return av.VideoFrame.from_ndarray(goster, format="bgr24")


def tespit_yap(img_bgr: np.ndarray, conf: float):
    results = model.predict(img_bgr, conf=conf, verbose=False)
    annotated = results[0].plot()
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    tespitler = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            tespitler.append((model.names[cls], float(box.conf[0])))
    return rgb, tespitler


def sonuclari_goster(rgb_img: np.ndarray, tespitler: list, baslik: str):
    st.image(rgb_img, caption=baslik, use_container_width=True)
    if tespitler:
        st.subheader("Tespitler")
        for ad, skor in tespitler:
            st.write(f"- {ad}: %{skor * 100:.1f}")
    else:
        st.info("Bu karede tespit yapılmadı.")


st.title("KKD (PPE) Tespit Sistemi")
st.caption(f"Model: `{model_yolu}`")

conf = st.slider("Güven eşiği", 0.1, 1.0, 0.5)
analiz_araligi = st.slider(
    "Canlı kamera analiz aralığı (saniye)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5,
    help="Model her bu kadar saniyede bir kareyi analiz eder.",
)

tab_resim, tab_canli = st.tabs(["Resim yükle", "Canlı kamera"])

with tab_resim:
    uploaded = st.file_uploader("Resim yükle", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Resim okunamadı.")
        else:
            rgb, tespitler = tespit_yap(img, conf)
            sonuclari_goster(rgb, tespitler, "Yüklenen resim — tespit sonucu")
    else:
        st.info("Analiz için bir resim yükleyin.")

with tab_canli:
    st.markdown(
        "**START** ile kamerayı açın. Model saniyede bir (veya seçtiğiniz aralıkta) "
        "kareyi analiz eder; kutular canlı görüntüde kalır."
    )

    def video_processor_factory():
        return KKDVideoProcessor(model, conf, analiz_araligi)

    ctx = webrtc_streamer(
        key="kkd-canli-kamera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=video_processor_factory,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        async_processing=False,
    )

    if ctx.video_processor:
        ctx.video_processor.conf = conf
        ctx.video_processor.interval_sec = analiz_araligi

    @st.fragment(run_every=1)
    def son_analiz_paneli():
        if not ctx.video_processor:
            return
        vp = ctx.video_processor
        st.caption(f"Son analiz: {vp.son_analiz_zamani}")
        if vp.son_tespitler:
            st.subheader("Son karedeki tespitler")
            for ad, skor in vp.son_tespitler:
                st.write(f"- {ad}: %{skor * 100:.1f}")
        else:
            st.info("Son analizde tespit yok.")

    son_analiz_paneli()
