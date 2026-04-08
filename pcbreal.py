import streamlit as st
import torch
from PIL import Image
import pandas as pd
import os
from yolov5 import load

# -------- PAGE --------
st.set_page_config(page_title="AI PCB Inspector", layout="wide")
st.title("🧠 AI PCB DEFECT INSPECTOR")

# -------- SIDEBAR --------
option = st.sidebar.selectbox(
    "Inspection Stage",
    ["Bare PCB Inspection", "Soldering Stage Inspection"]
)

# -------- UPLOAD --------
uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg","png","jpeg"])

# =====================================
# LOAD MODEL (FAST + STABLE)
# =====================================
@st.cache_resource
def load_model(model_path):

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        st.stop()

    model = load(model_path)   # 🔥 LOCAL LOAD (NO INTERNET)
    return model

# =====================================
# DETECTION
# =====================================
def detect(model, image):

    results = model(image)

    results.render()
    output_img = Image.fromarray(results.ims[0])

    df = results.pandas().xyxy[0]

    return output_img, df

# =====================================
# MAIN
# =====================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    # -------- MODEL PATH --------
    if option == "Bare PCB Inspection":
        model_path = "best_bare.pt"
    else:
        model_path = "best_solder.pt"

    # LOAD MODEL
    with st.spinner("🚀 Loading model (first time ~10 sec)..."):
        model = load_model(model_path)

    # BUTTON (IMPORTANT)
    if st.button("Run Detection"):

        with st.spinner("🔍 Detecting defects..."):
            output, df = detect(model, image)

        with col2:
            st.subheader("Detection Output")
            st.image(output, use_container_width=True)

        st.markdown("---")

        if len(df) == 0:
            st.warning("⚠️ No defects detected")
        else:
            st.dataframe(df)
            st.success(f"🔥 Total Detections: {len(df)}")

        # DOWNLOAD
        output.save("pcb_result.jpg")
        with open("pcb_result.jpg", "rb") as f:
            st.download_button("Download Result", f, "pcb_result.jpg")
