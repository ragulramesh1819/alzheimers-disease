# %%
!pip install streamlit
!pip install tensorflow

# %%
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# %%

# Load the trained model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\ragul\Downloads\alzheimers_prediction.h5")
    return model



# %%
model = load_model()

# Define class labels
CLASS_NAMES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Streamlit UI
st.title("Alzheimer's Disease Prediction")
st.write("Upload an MRI image to predict the stage of Alzheimer's disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((128, 128))  # Resize as per model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    # Display prediction
    st.write("### Prediction:")
    st.success(predicted_class)


# %%
!streamlit run  c:/Users/ragul/miniconda3/envs/tvm-build-venv/Lib/site-packages/ipykernel_launcher.py

# %% [markdown]
# 


