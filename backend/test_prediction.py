# test_prediction.py
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('models/cancer_classifier.h5')

# Load and preprocess image
img = Image.open('path/to/your/image.jpg').convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]
print(f"Prediction: {prediction:.4f}")
print(f"Classification: {'MALIGNANT' if prediction > 0.5 else 'BENIGN'}")