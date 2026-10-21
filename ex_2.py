import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Load pretrained VGG16
model = VGG16(weights="imagenet")

# Custom dataset path
folder = "dataset"

# Select three convolutional layers
layers = ["block1_conv1", "block3_conv1", "block5_conv1"]

# Process each image
for file in os.listdir(folder):
    if file.endswith((".png", ".jpg", ".jpeg")):

        path = os.path.join(folder, file)

        # Load and resize image
        img = tf.keras.utils.load_img(
            path, target_size=(224, 224)
        )
        img = tf.keras.utils.img_to_array(img)

        # Preprocess image
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x.copy())

        print("\nImage:", file)
        print("Input shape:", img.shape)
        print("Preprocessed shape:", x.shape)

        # Extract feature maps from three layers
        feature_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(l).output for l in layers]
        )
        
        features = feature_model.predict(x, verbose=0)

        # Display original image
        plt.figure(figsize=(4, 4))
        plt.imshow(img.astype("uint8"))
        plt.title(file)
        plt.axis("off")
        plt.show()

        # Display feature maps
        for name, feature in zip(layers, features):

            print(name, "Feature map shape:", feature.shape)

            plt.figure(figsize=(12, 3))

            for i in range(6):
                plt.subplot(1, 6, i + 1)
                plt.imshow(feature[0, :, :, i], cmap="gray")
                plt.axis("off")

            plt.suptitle(name + " Feature Maps")
            plt.show()

        # Visualize learned filters from first layer
        filters, bias = model.get_layer(
            "block1_conv1"
        ).get_weights()

        print("Filter shape:", filters.shape)

        plt.figure(figsize=(8, 4))

        for i in range(6):
            f = filters[:, :, :, i]
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)

            plt.subplot(2, 3, i + 1)
            plt.imshow(f)
            plt.title("Filter " + str(i + 1))
            plt.axis("off")

        plt.suptitle("Learned CNN Filters")
        plt.tight_layout()
        plt.show()