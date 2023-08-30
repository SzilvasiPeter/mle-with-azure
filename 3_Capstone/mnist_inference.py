import json

import numpy as np
import tensorflow as tf
import requests

from utils import visualize


(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
test_images = test_images / 255.0

# Select a few random samples for testing
num_samples = 4
sample_indices = np.random.choice(len(test_images), num_samples, replace=False)
sample_images = test_images[sample_indices]

# Prepare the data for inference
data = {"data": sample_images.tolist()}
with open("data_example.json", 'w') as file:
    file.write(json.dumps(data))

# Endpoint information
endpoint_url = "YOUR_ENDPOINT_URL"
authentication_key = "YOUR_AUTHENTICATION_KEY"

# Headers for authentication
headers = {"Content-Type": "application/json"}
headers["Authorization"] = f"Bearer {authentication_key}"

# Send the inference request
response = requests.post(endpoint_url, json=data, headers=headers)
if response.status_code == 200:
    result = response.json()
    print("Inference Result:", result)
else:
    print("Inference request failed with status code:", response.status_code)
    print("Response:", response.text)

predicted_labels = ...

for idx in sample_indices:
    visualize(test_images[idx])
    print("Predicted labels: ", predicted_labels[idx])
    print("Ground truth: ", test_labels[idx])
