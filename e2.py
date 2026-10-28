import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input
)

# 1. Load an image from Experiment 1
folder = "dataset"

files = [
    f for f in os.listdir(folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
files.sort()

file = files[0]
path = os.path.join(folder, file)

img = load_img(path, target_size=(224, 224))
image = img_to_array(img)

print("Selected image:", file)
print("Original image shape:", image.shape)

# 2. Preprocess image for VGG16
image_batch = np.expand_dims(image, axis=0)
processed = preprocess_input(image_batch.copy())

print("Preprocessed image shape:", processed.shape)

# 3. Load pretrained VGG16
model = VGG16(weights="imagenet", include_top=True)

# 4. Select shallow, intermediate and deep convolutional layers
layer_names = [
    "block1_conv1",
    "block3_conv1",
    "block5_conv3"
]

# 5. Extract feature maps
outputs = [model.get_layer(name).output for name in layer_names]

feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=outputs
)

feature_maps = feature_model.predict(processed, verbose=0)

# 6. Display original image
plt.figure(figsize=(4, 4))
plt.imshow(image.astype("uint8"))
plt.title("Input Image: " + file)
plt.axis("off")
plt.show()

# 7. Visualize feature maps from each layer
for layer_name, feature_map in zip(layer_names, feature_maps):

    print("\nLayer:", layer_name)
    print("Feature map dimensions:", feature_map.shape)

    # Display first 8 feature maps
    plt.figure(figsize=(12, 6))

    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(feature_map[0, :, :, i], cmap="viridis")
        plt.title("Map " + str(i + 1))
        plt.axis("off")

    plt.suptitle("Feature Maps: " + layer_name)
    plt.tight_layout()
    plt.show()

# 8. Visualize learned convolutional filters
filter_layer = model.get_layer("block1_conv1")
filters, biases = filter_layer.get_weights()

print("\nFilter layer:", filter_layer.name)
print("Filter dimensions:", filters.shape)

# Display first 8 filters
plt.figure(figsize=(12, 6))

for i in range(8):
    f = filters[:, :, :, i]

    # Normalize filter values for display
    f_min, f_max = f.min(), f.max()
    f = (f - f_min) / (f_max - f_min + 1e-8)

    plt.subplot(2, 4, i + 1)
    plt.imshow(f)
    plt.title("Filter " + str(i + 1))
    plt.axis("off")

plt.suptitle("Learned Convolutional Filters")
plt.tight_layout()
plt.show()

# 9. Compare layer dimensions
print("\n--- LAYER COMPARISON ---")

for layer_name, feature_map in zip(layer_names, feature_maps):
    print(layer_name, ":", feature_map.shape)

print("\n--- INTERPRETATION ---")
print("Early layers: edges, colours and simple textures.")
print("Intermediate layers: patterns and object parts.")
print("Deep layers: complex shapes and semantic features.")
print("Spatial dimensions decrease with network depth.")
print("The number of channels increases with network depth.")