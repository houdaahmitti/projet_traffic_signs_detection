import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image

# ================== MODELS ==================
from ultralytics import YOLO

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Traffic Signs Detection",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #0f172a; }
h1, h2, h3 { color: #38bdf8; }
.stButton>button {
    background-color: #38bdf8;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
section[data-testid="stSidebar"] {
    background-color: #020617;
}
</style>
""", unsafe_allow_html=True)

# ================== TITLE ==================
st.title("🚦 Traffic Signs Detection Platform")
st.markdown("**YOLOv8 & Detectron2 – Image & Video Detection**")

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    st.markdown("---")

    model_choice = st.selectbox(
        "🧠 Detection Model",
        ["YOLOv8", "Detectron2"]
    )

    input_type = st.radio(
        "📂 Input Type",
        ["Image", "Video"],
        horizontal=True
    )

    confidence = st.slider(
        "🎯 Confidence Threshold",
        0.1, 1.0, 0.4, 0.05
    )

    run_detection = st.button("🚀 Run Detection")

    st.markdown("---")
    st.markdown(
        "<center><small>© 2026 • Traffic Signs Detection<br>YOLO & Detectron2</small></center>",
        unsafe_allow_html=True
    )

# ================== LOAD MODELS ==================
@st.cache_resource
def load_yolo():
    return YOLO("models/yolo/best.pt")

@st.cache_resource
def load_detectron(conf):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
    cfg.MODEL.WEIGHTS = "models/detectron2/model_final.pth"
    cfg.MODEL.DEVICE = "cpu"
    cfg.DATASETS.TRAIN = ("traffic_signs",)

    MetadataCatalog.get("traffic_signs").set(thing_classes=[])

    predictor = DefaultPredictor(cfg)
    return predictor, cfg

# ================== INFERENCE ==================
def yolo_detect(image):
    model = load_yolo()
    results = model(image, conf=confidence)
    return results[0].plot()

def detectron_detect(image):
    predictor, cfg = load_detectron(confidence)
    outputs = predictor(image)

    v = Visualizer(
        image[:, :, ::-1],
        MetadataCatalog.get("traffic_signs"),
        scale=1.2
    )

    out = v.draw_instance_predictions(
        outputs["instances"].to("cpu")
    )

    return out.get_image()[:, :, ::-1]

# ================== IMAGE ==================
if input_type == "Image":
    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Original Image")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("🎯 Detection Result")

            if run_detection:
                if model_choice == "YOLOv8":
                    result = yolo_detect(image_np)
                else:
                    result = detectron_detect(image_np)

                st.image(result, use_column_width=True)
            else:
                st.info("👈 Click **Run Detection**")

# ================== VIDEO ==================
else:
    uploaded_video = st.file_uploader(
        "📤 Upload Video",
        type=["mp4", "avi"]
    )

    if uploaded_video and run_detection:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if model_choice == "YOLOv8":
                frame = yolo_detect(frame)
            else:
                frame = detectron_detect(frame)

            stframe.image(frame, channels="BGR")

        cap.release()
