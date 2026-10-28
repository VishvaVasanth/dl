import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.model_selection import train_test_split

# 1. Load dataset
folder = "dataset"

files = sorted([
    f for f in os.listdir(folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

if len(files) < 2:
    raise ValueError("Add at least 2 images to the dataset folder.")

labels = [os.path.splitext(f)[0] for f in files]

print("Total images:", len(files))
print("Image labels:", labels)

# 2. Split into training and testing sets
train, test = train_test_split(
    files, test_size=0.25, random_state=42
)

print("\nTraining images:", len(train))
print("Testing images:", len(test))

# 3. Load and resize all images
def load_images(file_list):
    imgs = []
    for file in file_list:
        img = load_img(
            os.path.join(folder, file),
            target_size=(224, 224)
        )
        imgs.append(img_to_array(img))
    return np.array(imgs, dtype="float32")

train_images = load_images(train)
test_images = load_images(test)

print("\nTraining shape:", train_images.shape)
print("Testing shape:", test_images.shape)

# 4. Preprocess images for VGG16
train_processed = preprocess_input(train_images.copy())
test_processed = preprocess_input(test_images.copy())

print("Preprocessed test shape:", test_processed.shape)
print("Sample pixel values:", test_processed[0, 0, 0, :])

# 5. Load pretrained VGG16
model = VGG16(weights="imagenet", include_top=True)

# 6. Create model for intermediate feature maps
feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("block1_conv1").output
)

features = feature_model.predict(test_processed, verbose=0)

print("Feature map shape:", features.shape)

# 7. Predict test images
predictions = model.predict(test_processed, verbose=0)

print("Prediction shape:", predictions.shape)

# 8. Display test images and top-5 predictions
for i, file in enumerate(test):
    label = os.path.splitext(file)[0]

    top5 = decode_predictions(
        predictions[i:i+1], top=5
    )[0]

    print("\nImage:", file)
    print("Original filename label:", label)
    print("Top-5 predictions:")

    for rank, (_, name, score) in enumerate(top5, 1):
        print(f"{rank}. {name}: {score * 100:.2f}%")

    print("Top-1 prediction:", top5[0][1])

    # Display original image
    plt.figure(figsize=(5, 4))
    plt.imshow(test_images[i].astype("uint8"))
    plt.title("Original: " + label)
    plt.axis("off")
    plt.show()

    # Display feature map
    plt.figure(figsize=(5, 4))
    plt.imshow(features[i, :, :, 0], cmap="gray")
    plt.title("VGG16 Feature Map")
    plt.axis("off")
    plt.show()

# 9. Summary
print("\n--- PREPROCESSING SUMMARY ---")
print("Training images:", len(train_images))
print("Testing images:", len(test_images))
print("Resized image shape:", test_images[0].shape)
print("Preprocessed shape:", test_processed.shape)
print("Feature map shape:", features.shape)
print("Prediction shape:", predictions.shape)
print("Model: VGG16 pretrained on ImageNet")
print("Preprocessing: VGG16 preprocess_input()")