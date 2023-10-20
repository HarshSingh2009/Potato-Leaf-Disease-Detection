from PIL import Image
import tensorflow as tf
import streamlit as st
from deep_learning_pipeline import PredictionPipeline

st.title('Potato Leaf Disease Detection')
st.write('This Project is built using CNN (Convolutional Neural Networks) and help in predicts if the potato leaf is Early Blight, Healthy or Late Blight')

st.write('')
st.write('')


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Process the uploaded image here\
    with st.container():
        col1, col2 = st.columns([3, 2])
        col1.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        # leave some extra space at the top
        col2.text('')
        col2.text('')

        if st.button('Predict!!'):
            pipeline = PredictionPipeline()
            y_pred, y_probs = pipeline.predict(input_img=uploaded_file)
            st.write(y_pred, y_probs)
            col2.balloons()
            col2.subheader(f'{pipeline.CLASS_NAMES[int(y_pred)]}!!')
            acc = '{:.2f}'.format(100*(float(y_probs)))
            col2.success(f'Accuracy: {acc}%')
            if int(y_pred) == -1:
                col2.error('Error!! Model needs shape (224, 224, 3), but your image is of shape (224, 224,4)')