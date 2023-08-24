import requests
import json
import numpy as np
from tensorflow.keras.datasets import cifar10

from utils import visualize

(_, _), (test_images, test_labels) = cifar10.load_data()

# Select a few random samples for testing
num_samples = 3
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

# TODO: Get the predicted labels from the response
pred_labels = ...

# Convert predicted_labels and test_labels to class names
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
predicted_class_names = [class_names[label] for label in pred_labels]
true_class_names = [class_names[label] for label in test_labels]

# Checkout the example
for idx in sample_indices:
    visualize(test_images[idx])
    print("Predicted labels: ", predicted_class_names[idx])
    print("Ground truth: ", true_class_names[idx])
