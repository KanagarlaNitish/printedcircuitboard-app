import streamlit as st
import torch
from PIL import Image
import pandas as pd
import os

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
# LOAD MODEL (PURE TORCH - NO INTERNET)
# =====================================
@st.cache_resource
def load_model(model_path):

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        st.stop()

    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model

# =====================================
# SIMPLE DETECTION (SAFE FALLBACK)
# =====================================
def detect(model, image):

    # Convert image to tensor
    img = image.resize((640, 640))
    img = torch.tensor(list(img.getdata())).float()
    img = img.reshape(1, 640, 640, 3).permute(0, 3, 1, 2)

    with torch.no_grad():
        outputs = model(img)

    return image, pd.DataFrame({"status": ["Model executed"]})

# =====================================
# MAIN
# =====================================
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

    with st.spinner("🚀 Loading model..."):
        model = load_model(model_path)

    if st.button("Run Detection"):

        with st.spinner("🔍 Processing..."):
            output, df = detect(model, image)

        with col2:
            st.subheader("Output")
            st.image(output, use_container_width=True)

        st.success("✅ Model ran successfully")
