import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image


class PredictionPipeline():
    def __init__(self) -> None:
        self.CLASS_NAMES = ['Early Blight Leaf', 'Healthy Leaf', 'Late Blight Leaf']
        self.IMG_SIZE = 224

    def predict(self, input_img):
        model = load_model('Potato_Leaf_Disease_detection_model.h5')
        image = Image.open(input_img)
        # Changing the dtype to float32
        image = tf.cast(image, dtype=tf.float32)
        # Normalize the image data to [0, 1]
        image = image / 255.
        input_tensor = tf.expand_dims(tf.image.resize(image, [self.IMG_SIZE, self.IMG_SIZE]), axis=0)
        # Making Predictions
        try:
            y_probs = model.predict(input_tensor)
        except ValueError as err:
            return [[-1]], err
        else:
            return tf.argmax(y_probs, axis=1), tf.reduce_max(y_probs)