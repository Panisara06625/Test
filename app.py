import os
import gdown
import streamlit as st
import tensorflow as tf

@st.cache_resource
def download_and_load_model():
    model_path = "model_epoch_94.h5"

    if not os.path.exists(model_path):
        # Use only the file ID in the download URL
        file_id = "1U_Il1ynl7R3kFtJjlotS4JXE-I5W3nZw"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

    return tf.keras.models.load_model(model_path)

# App UI
st.title("Model Deployment Example")

if st.button("Load Model and Predict"):
    model = download_and_load_model()
    st.success("Model loaded!")

