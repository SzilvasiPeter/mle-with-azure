import time

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

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
    layers.Dropout(0.05),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.125),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
start_time = time.time()
history = model.fit(train_images, train_labels,
                    epochs=10, validation_data=(test_images, test_labels)
                    )
end_time = time.time()

training_time = end_time - start_time

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Write the test accuracy and training time to a file
with open("./models/cifar_model_dropout.log", "w") as file:
    file.write(f"Test Accuracy: {test_acc:.4f}")
    file.write(f"Training time: {training_time:.4f} seconds")

# Save the model
model.save("./models/cifar_model_dropout.keras")
print("Model saved successfully.")