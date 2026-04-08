import streamlit as st
import torch
from PIL import Image
import pathlib
import pandas as pd
import os

# -------- WINDOWS FIX --------
pathlib.PosixPath = pathlib.WindowsPath

# -------- PAGE --------
st.set_page_config(page_title="AI PCB Inspector", layout="wide")

st.markdown("""
<style>
.main {background: #0F172A;}
h1 {color:#00FFAA;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI PCB DEFECT INSPECTOR")

# -------- SIDEBAR --------
option = st.sidebar.selectbox(
    "Inspection Stage",
    ["Bare PCB Inspection", "Soldering Stage Inspection"]
)

# -------- UPLOAD --------
uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg","png","jpeg"])

# =====================================
# SIMPLE PREPROCESS (DON’T DISTURB IMAGE)
# =====================================
def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    return img

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model(path):

    if not os.path.exists(path):
        st.error(f"❌ Model not found:\n{path}")
        st.stop()

    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=path,
        force_reload=False
    )

    # 🔥 VERY LOW CONFIDENCE → FORCE DETECTION
    model.conf = 0.05

    return model

# =====================================
# DETECTION (YOLO BUILT-IN RENDER)
# =====================================
def detect(model, image):

    results = model(image)

    # 🔥 YOLO draws boxes (clean & accurate)
    rendered = results.render()[0]

    output_img = Image.fromarray(rendered)

    df = results.pandas().xyxy[0]

    return output_img, df

# =====================================
# MAIN
# =====================================
if uploaded_file:

    image = preprocess_image(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    # -------- YOUR SAME PATHS --------
    if option == "Bare PCB Inspection":
        model_path = r"C:\Users\bambo\OneDrive\Desktop\Desktop\MiniProject\BarePcb-stage\Work Done\yolov5-(good accuarcy)\bestt.pt"
    else:
        model_path = r"C:\Users\bambo\OneDrive\Desktop\Desktop\MiniProject\Soldering-stage\Work Done\Soldering (good accuarcy)\weights\best.pt"

    model = load_model(model_path)

    with st.spinner("Detecting defects..."):
        output, df = detect(model, image)

    with col2:
        st.subheader("Detection Output")
        st.image(output, use_container_width=True)

    st.markdown("---")

    # -------- RESULTS --------
    if len(df) == 0:
        st.warning("⚠️ Model is not confident, but system is working")
    else:
        st.dataframe(df)
        st.success(f"🔥 Total Detections: {len(df)}")

    # -------- DOWNLOAD --------
    output.save("pcb_result.jpg")
    with open("pcb_result.jpg", "rb") as f:
        st.download_button("Download Result", f, "pcb_result.jpg")