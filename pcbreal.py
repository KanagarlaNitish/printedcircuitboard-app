import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import os

st.set_page_config(page_title="AI PCB Inspector", layout="wide")
st.title("🧠 AI PCB DEFECT INSPECTOR")

option = st.sidebar.selectbox(
    "Inspection Stage",
    ["Bare PCB Inspection", "Soldering Stage Inspection"]
)

uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg","png","jpeg"])

# LOAD MODEL
@st.cache_resource
def load_model(model_path):

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        st.stop()

    model = YOLO(model_path)
    return model

# DETECT
def detect(model, image):

    results = model(image)

    plotted = results[0].plot()
    output_img = Image.fromarray(plotted)

    boxes = results[0].boxes
    if boxes is None:
        return output_img, pd.DataFrame()

    data = boxes.data.cpu().numpy()
    df = pd.DataFrame(data, columns=["x1","y1","x2","y2","confidence","class"])

    return output_img, df

# MAIN
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    if option == "Bare PCB Inspection":
        model_path = "bestt.pt"
    else:
        model_path = "best.pt"

    model = load_model(model_path)

    if st.button("Run Detection"):

        with st.spinner("🔍 Detecting..."):
            output, df = detect(model, image)

        with col2:
            st.subheader("Output")
            st.image(output, use_container_width=True)

        if df.empty:
            st.warning("No defects detected")
        else:
            st.dataframe(df)
            st.success(f"Detections: {len(df)}")
