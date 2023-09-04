import json
import os
import numpy as np
import tensorflow as tf

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "mnist_model.keras")
    model = tf.keras.models.load_model(model_path)

def run(raw_data):
    data = json.loads(raw_data)["data"]
    predictions = []
    
    for image in data:
        # Preprocess the image
        image = np.array(image)
        image = image.reshape((1, 28, 28, 1))
        image = image / 255.0
        
        # Make predictions
        result = model.predict(image)
        predicted_label = np.argmax(result, axis=1)
        predictions.append(predicted_label.tolist())
    
    return json.dumps(predictions)