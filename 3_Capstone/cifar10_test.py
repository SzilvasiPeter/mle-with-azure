import tensorflow as tf

from utils import get_difficult_examples, visualize


(_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
test_images = test_images / 255.0
sample_size = 10

# Extracting the first column, the ground truth labels
test_labels = test_labels[:, 0]

model = tf.keras.models.load_model("./models/cifar_model_dropout.keras")
pred_labels, missed_samples = get_difficult_examples(model,
                                                     test_images,
                                                     test_labels,
                                                     sample_size)

# Convert predicted_labels and test_labels to class names
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
predicted_class_names = [class_names[label] for label in pred_labels]
true_class_names = [class_names[label] for label in test_labels]

# random_samples = np.random.choice(len(test_labels), sample_size, replace=False)
for idx in missed_samples:
    visualize(test_images[idx])
    print("Predicted labels: ", predicted_class_names[idx])
    print("Ground truth: ", true_class_names[idx])
