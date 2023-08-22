import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Use seed for reproducibility
tf.random.set_seed(42)

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

num_epochs = 10
model.fit(train_images, train_labels,
          epochs=num_epochs, validation_data=(test_images, test_labels)
          )

# Save the model
model.save("./models/mnist_model.keras")
print("Model saved successfully.")
