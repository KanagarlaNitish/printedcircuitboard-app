import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import os

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
# PREPROCESS
# =====================================
def preprocess_image(file):
    return Image.open(file).convert("RGB")

# =====================================
# LOAD MODEL (FAST + FIXED)
# =====================================
@st.cache_resource
def load_model(model_path):

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        st.stop()

    model = YOLO(model_path)   # 🔥 FAST LOADING
    return model

# =====================================
# DETECTION
# =====================================
def detect(model, image):

    results = model(image)

    plotted = results[0].plot()  # draw boxes
    output_img = Image.fromarray(plotted)

    # convert detections to dataframe
    boxes = results[0].boxes
    if boxes is None:
        return output_img, pd.DataFrame()

    data = boxes.data.cpu().numpy()
    df = pd.DataFrame(data, columns=["x1","y1","x2","y2","confidence","class"])

    return output_img, df

# =====================================
# MAIN
# =====================================
if uploaded_file is not None:

    image = preprocess_image(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    # -------- MODEL PATH (FIXED) --------
    if option == "Bare PCB Inspection":
        model_path = "bestt.pt"
    else:
        model_path = "best.pt"

    # DEBUG CHECK
    if not os.path.exists(model_path):
        st.error(f"❌ {model_path} not found in repo")
        st.stop()

    # LOAD MODEL
    with st.spinner("🚀 Loading model (first time takes 10-20 sec)..."):
        model = load_model(model_path)

    # RUN BUTTON (IMPORTANT FOR SPEED)
    if st.button("Run Detection"):

        with st.spinner("🔍 Detecting defects..."):
            output, df = detect(model, image)

        with col2:
            st.subheader("Detection Output")
            st.image(output, use_container_width=True)

        st.markdown("---")

        if df.empty:
            st.warning("⚠️ No defects detected")
        else:
            st.dataframe(df)
            st.success(f"🔥 Total Detections: {len(df)}")

        # DOWNLOAD
        output.save("pcb_result.jpg")
        with open("pcb_result.jpg", "rb") as f:
            st.download_button("Download Result", f, "pcb_result.jpg")
