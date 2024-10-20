import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    # Load EfficientNetB0 model in .keras format
    effb0_model = tf.keras.models.load_model("efficientnetb0_sickle_cell_model.keras")
    
    # Load the ensemble model using joblib
    ensemble_model = joblib.load("ensemble_weights.joblib")
    
    return effb0_model, ensemble_model

# Preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to (224x224) for EfficientNetB0 input size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize image to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)
    return img_array

# Load models
effb0_model, ensemble_model = load_models()

# Streamlit UI
st.title("Sickle Cell Disease Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg", "tiff"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Classification with EfficientNetB0
    effb0_prediction = effb0_model.predict(preprocessed_image)
    effb0_class = np.argmax(effb0_prediction, axis=1)[0]

    # Classification with Ensemble model (assuming ensemble_model is a sklearn model)
    ensemble_prediction = ensemble_model.predict(preprocessed_image.reshape(1, -1))  # Flatten image
    ensemble_class = ensemble_prediction[0]

    # Display Results
    st.write("### Model Predictions:")
    st.write(f"EfficientNetB0 Prediction: {'Sickle Cell Disease' if effb0_class == 1 else 'Normal'}")
    st.write(f"Ensemble Model Prediction: {'Sickle Cell Disease' if ensemble_class == 1 else 'Normal'}")
