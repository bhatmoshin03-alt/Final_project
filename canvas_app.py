import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.title("✍️ Draw a Digit Recognition App (0-9)")
st.write("Draw a digit in the box below")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.h5", compile=False)

model = load_model()

canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:

    img = Image.fromarray(canvas_result.image_data.astype("uint8"))
    img = img.convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.image(img, caption="Processed Image (28x28)", width=150)
    st.success(f"Predicted Digit: {digit}")