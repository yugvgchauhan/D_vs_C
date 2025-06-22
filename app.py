import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
@st.cache_resource
def load_cnn_model():
    return load_model("dogs_vs_cats_model.h5")

model = load_cnn_model()

# App title
st.title("🐶 Dog vs Cat Classifier")
st.write("Upload an image and let the model predict whether it's a **Dog** or a **Cat**!")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image',  use_container_width=True)

    # Preprocess image
    img = image.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Show result
    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
