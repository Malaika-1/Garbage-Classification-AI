
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("Garbage_Classifier.h5")


classes = [
    "cardboard","clothes",
    "glass","metal","paper","plastic","shoes"
]

st.title("♻️ Garbage Classification")
st.write("Upload an image of waste and the model will classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((128,128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader("Prediction:")
    st.write(predicted_class)

    st.subheader("Confidence:")
    st.write(f"{confidence*100:.2f}%")

