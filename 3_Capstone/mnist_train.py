import argparse
import os
import time

from azureml.core.run import Run

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


def main():
    # Add arguments for HyperDrive hyper parameter sampling
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dropout1",
        type=float,
        default=0.1,
        help="The first layer dropout value for the CNN."
    )
    parser.add_argument(
        "--dropout2",
        type=float,
        default=0.3,
        help="The second layer dropout value for the CNN."
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
    epochs = args.epochs

    run = Run.get_context()
    run.log("First layer dropout value:", dropout1)
    run.log("Second layer dropout value:", dropout2)
    run.log("Number of Epoch:", epochs)
    
    # Use seed for reproducibility
    tf.random.set_seed(42)

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout1),
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout2),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    start_time = time.time()
    model.fit(train_images, train_labels,
                        epochs=epochs, validation_data=(test_images, test_labels)
                        )
    end_time = time.time()

    training_time = end_time - start_time
    _, test_accuracy = model.evaluate(test_images, test_labels)

    run.log("accuracy", test_accuracy)
    run.log("training_time", training_time)

    # Hyperdrive can only save the model into the `outputs` folder
    os.makedirs("./outputs", exist_ok=True)
    model.save("./outputs/mnist_model.keras")


if __name__ == "__main__":
    main()