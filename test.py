import tensorflow as tf
from keras.models import load_model

# Check TensorFlow and Keras versions
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# Load the saved model
print("Loading model...")
model = load_model('model1.keras')
print("Model loaded successfully.")