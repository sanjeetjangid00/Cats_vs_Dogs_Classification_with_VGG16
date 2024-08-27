import streamlit as st
import tensorflow as tf
import keras
from PIL import Image
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def centered_content():
    col1,col2,col3 = st.columns([1,12,1])
    return col1, col2, col3
with centered_content()[1]:
    st.title(":blue[Cat & Dog Image Classifier]")

model = tf.keras.models.load_model('model.h5')
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    resized_image = image.resize((300,150))
    st.image(resized_image, caption='Uploaded Image', use_column_width=False)
    resized_img = image.resize((150,150))
    image_array = np.array(resized_img) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    with st.spinner("loading..."):
        image_pred = model.predict(image_array)
        if image_pred>0.5:
            st.header(":green[It's a Dog Image]")
            st.write(":red[It can predict 97% accurate only...]")
        else:
            st.header(":green[It's a Cat Image]")
            st.write(":red[It can predict 97% accurate only...]")
else:
    st.warning(":red[Please upload an image....]")
