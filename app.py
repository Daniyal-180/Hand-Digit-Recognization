import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('mnist_model.keras')

# Set title
st.title('MNIST Digit Classification')

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    # Load image
    img = image.load_img(uploaded_file, target_size=(28, 28), color_mode="grayscale")
    img = image.img_to_array(img)
    img = img.reshape(1, 784).astype('float32') / 255  # Normalize
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)

    # Display the prediction
    st.write(f'Predicted Digit: {predicted_class[0]}')

    # Optional: Show the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
