import os
import gdown
import streamlit as st
import torch
from PIL import Image
import numpy as np
from model import UNet
MODEL_PATH = "best_model_hy.pth"
FILE_ID = "1LoiR4InZtpVQHBg986rdkYdzcemF-qav"

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    download_model()
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Preprocess function
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image

# Predict
def predict(image):
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
    return output.squeeze().numpy()

# UI
st.title("🧠 Brain Tumor Segmentation App")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    input_tensor = preprocess_image(image)
    prediction = predict(input_tensor)

    # Resize original image to 256x256
    original = np.array(image.resize((256, 256)))

    # Make sure mask is 0 or 1
    prediction = (prediction > 0.5).astype(np.uint8)

    # Create overlay
    overlay = original.copy()
    overlay[prediction == 1] = [255, 0, 0]

    # Calculate tumor percentage
    tumor_pixels = np.sum(prediction)
    total_pixels = prediction.size
    percentage = (tumor_pixels / total_pixels) * 100

    st.subheader("Segmentation Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original MRI", width=250)

    with col2:
        st.image(prediction * 255, caption="Predicted Mask", width=250)

    with col3:
        st.image(overlay, caption="Overlay Result", width=250)

    st.markdown("---")

    if percentage > 1:
        st.error(f"⚠ Tumor Detected: {percentage:.2f}% of brain area affected.")
    else:
        st.success("✅ You are safe. No tumor detected.")