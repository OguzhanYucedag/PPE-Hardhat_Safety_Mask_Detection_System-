from roboflow import Roboflow

rf = Roboflow(api_key="vdQQJVALjiva7lxxZPT8")
proje = rf.workspace("sdp-lfigk").project("ppe-detection-ozhfb")
surum = proje.version(9)
veri_kumesi = surum.download("yolov8")