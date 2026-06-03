import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

model = tf.keras.models.load_model("mnist_production.h5")

st.title("Digit Tester (Image Upload)")

uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])

def preprocess(img):
    img = img.convert("L")
    img = np.array(img)

    img = 255 - img

    # resize directly (simple & stable for test images)
    img = Image.fromarray(img).resize((28, 28))

    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img

if uploaded_file:
    img = Image.open(uploaded_file)

    st.image(img, caption="Input Image", width=150)

    processed = preprocess(img)

    pred = model.predict(processed)[0]

    st.subheader(f"Prediction: {np.argmax(pred)}")
    st.bar_chart(pred)