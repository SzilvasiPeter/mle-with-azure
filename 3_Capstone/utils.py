import json
import os

import numpy as np
import matplotlib.pyplot as plt


def get_difficult_examples(model, test_images, test_labels, sample_size=10) -> None:
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    differing_idx = np.where(predicted_labels != test_labels)[0]
    sampled_indexes = np.random.choice(differing_idx, size=sample_size, replace=False)

    return predicted_labels, sampled_indexes

def visualize(image: np.array) -> None:
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def save_to_json(file_name, model_history, test_accuracy, training_time):
    history = model_history.history
    results = {
        "test_accuracy": test_accuracy,
        "training_time": training_time
        }
    
    os.makedirs("./logs", exist_ok=True)
    with open(f"./logs/{file_name}_history.json", "w") as file:
        json.dump(history, file)

    with open(f"./logs/{file_name}_result.json", "w") as file:
        json.dump(results, file, indent=4)