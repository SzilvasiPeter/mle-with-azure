import argparse
import json
import time

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

import numpy as np
from azureml.core.run import Run

def save_to_json(model_history, test_accuracy, training_time):
    history = model_history.history
    results = {
        "test_accuracy": test_accuracy,
        "training_time": training_time
        }
    
    with open("./models/training_history.json", "w") as file:
        json.dump(history, file)

    with open("./models/cifar_model_dropout.json", "w") as file:
        json.dump(results, file, indent=4)

def main():
    # Add arguments for HyperDrive hyper parameter sampling
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dropout1",
        type=float,
        default=0.05,
        help="The first layer dropout value for the CNN."
    )
    parser.add_argument(
        "--dropout2",
        type=str,
        default=0.125,
        help="The second layer dropout value for the CNN."
    )
    parser.add_argument(
        "--dropout3",
        type=str,
        default=0.25,
        help="The third layer dropout value for the CNN."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model"
    )

    args = parser.parse_args()
    dropout1 = args.dropout1
    dropout2 = args.dropout2
    dropout3 = args.dropout3
    epochs = args.epochs

    run = Run.get_context()
    run.log("First layer dropout value:", np.float(dropout1))
    run.log("Second layer dropout value:", np.float(dropout2))
    run.log("Third layer dropout value:", np.float(dropout3))
    run.log("Number of Epoch:", np.int(epochs))

    # Use seed for reproducibility
    tf.random.set_seed(42)

    # Load and preprocess CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels, test_labels = (to_categorical(train_labels, 10),
                                to_categorical(test_labels, 10)
                                )

    # Build the CNN model with Dropout
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout1),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout3),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    # Train the model
    start_time = time.time()
    history = model.fit(train_images, train_labels,
              epochs=epochs, validation_data=(test_images, test_labels)
              )
    end_time = time.time()

    training_time = end_time - start_time

    # Evaluate the model
    _, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)

    run.log("accuracy", np.float(test_accuracy))
    run.log("training_time", np.float(training_time))

    save_to_json(history, test_accuracy, training_time)

    # Save the model
    model.save(f"./models/cifar_model_dropout.keras")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
