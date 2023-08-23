import argparse
import json
import time

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


def main():
    # Add arguments for HyperDrive hyper parameter sampling
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dropout_values",
        type=str,
        default="0.05,0.125,0.25",
        help="The dropout values seperated by commas for a three layer CNN."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model"
    )
    args = parser.parse_args()
    dropout_values = [float(value) for value in args.dropout_values.split(',')]
    epochs = args.epochs

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
        layers.Dropout(dropout_values[0]),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_values[1]),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_values[2]),
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

    # Save the training history to a JSON file
    history_file = "./models/training_history.json"
    with open(history_file, "w") as file:
        json.dump(history.history, file)

    # Save the test accuracy and training time to a JSON file
    results = {
        "test_accuracy": test_accuracy,
        "training_time": training_time
        }

    with open(f"./models/cifar_model_dropout_{dropout_values}.json", "w") as file:
        json.dump(results, file, indent=4)

    # Save the model
    model.save(f"./models/cifar_model_dropout_{dropout_values}.keras")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
