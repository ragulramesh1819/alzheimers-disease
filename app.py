# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image


# # Load the trained model
# @st.cache_resource()
# def load_model():
#     model = tf.keras.models.load_model(r"C:\Users\ragul\Downloads\alzheimers_prediction.h5")
#     return model

# model = load_model()

# # Define class labels
# CLASS_NAMES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# # Streamlit UI
# st.title("Alzheimer's Disease Prediction")
# st.write("Upload an MRI image to predict the stage of Alzheimer's disease.")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...")

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess the image
#     img = image.resize((128, 128))  # Resize as per model input size
#     img_array = np.array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     # Make prediction
#     prediction = model.predict(img_array)
#     predicted_class = CLASS_NAMES[np.argmax(prediction)]

#     # Display prediction
#     st.write("### Prediction:")
#     st.success(predicted_class)


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\ragul\Downloads\alzheimers_prediction.h5")
    return model

model = load_model()

# Define class labels
CLASS_NAMES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Streamlit UI
st.title("Alzheimer's Disease Prediction")
st.write("Upload an MRI image to predict the stage of Alzheimer's disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    target_size = model.input_shape[1:3]  # Get expected height & width
    img = image.resize(target_size)  # Resize as per model input size
    img_array = np.array(img)

    # Ensure correct shape & type
    if img_array.shape[-1] == 4:  # Convert RGBA to RGB if needed
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimension

    # Make prediction
    try:
        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display prediction
        st.write("### Prediction:")
        st.success(f"{predicted_class} ({confidence:.2f}% confidence)")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
