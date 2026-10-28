import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input
)
from tensorflow.keras.layers import GlobalAveragePooling2D

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

# 1. Dataset folder path
folder = "sample_dataset"

image_size = (224, 224)
batch_size = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    folder,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="int"
)

# 3. Load testing dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    folder,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="int"
)

class_names = train_ds.class_names

print("\nClasses:", class_names)

# 4. Load pretrained VGG16 without its classification layer
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
# 5. Build feature extractor using Global Average Pooling
inputs = tf.keras.Input(shape=(224, 224, 3))

x = preprocess_input(inputs)
x = base_model(x, training=False)
outputs = GlobalAveragePooling2D()(x)

feature_model = tf.keras.Model(inputs, outputs)

# 6. Extract training features
X_train = []
y_train = []

for images, labels in train_ds:
    features = feature_model.predict(images, verbose=0)
    X_train.extend(features)
    y_train.extend(labels.numpy())

X_train = np.array(X_train)
y_train = np.array(y_train)

print("\nTraining feature shape:", X_train.shape)
print("Feature vector dimension:", X_train.shape[1])

# 7. Extract testing features
X_test = []
y_test = []

for images, labels in test_ds:
    features = feature_model.predict(images, verbose=0)
    X_test.extend(features)
    y_test.extend(labels.numpy())

X_test = np.array(X_test)
y_test = np.array(y_test)

print("Testing feature shape:", X_test.shape)

# 8. Train a new classification layer
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# 9. Predict test classes
y_pred = classifier.predict(X_test)

print("\nActual labels:")
print([class_names[i] for i in y_test])

print("\nPredicted labels:")
print([class_names[i] for i in y_pred])

# 10. Calculate evaluation metrics
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(
    y_test, y_pred, average="macro", zero_division=0
))

print("Recall:", recall_score(
    y_test, y_pred, average="macro", zero_division=0
))

print("F1-score:", f1_score(
    y_test, y_pred, average="macro", zero_division=0
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
))

# 11. Generate confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(class_names))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 12. Display a sample test image
for images, labels in test_ds.take(1):
    plt.figure(figsize=(5, 5))
    plt.imshow(images[0].numpy().astype("uint8"))
    plt.title("Original Class: " + class_names[int(labels[0])])
    plt.axis("off")
    plt.show()

# 13. Display one feature vector

plt.figure(figsize=(10, 3))
plt.plot(X_train[0])
plt.title("VGG16 Feature Vector")
plt.xlabel("Feature Index")
plt.ylabel("Feature Value")
plt.show()

# 14. Summary
print("\n--- SUMMARY ---")
print("CNN: VGG16")
print("Feature extraction: Pretrained VGG16")
print("Dimensionality reduction: Global Average Pooling")
print("Feature vector length:", X_train.shape[1])
print("Number of classes:", len(class_names))
print("Classifier: Logistic Regression")