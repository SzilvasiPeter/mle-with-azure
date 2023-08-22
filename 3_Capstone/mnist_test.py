import tensorflow as tf

from utils import get_difficult_examples, visualize


(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
test_images = test_images / 255.0
sample_size = 10

model = tf.keras.models.load_model("./models/mnist_model.keras")
predicted_labels, sample_indexes = get_difficult_examples(model,
                                                          test_images,
                                                          test_labels,
                                                          sample_size)

for idx in sample_indexes:
    visualize(test_images[idx])
    print("Predicted labels: ", predicted_labels[idx])
    print("Ground truth: ", test_labels[idx])
