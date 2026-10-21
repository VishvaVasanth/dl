import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load custom dataset
classes = ["bird", "car", "cat", "dog", "horse"]
images = []

for name in classes:
    img = tf.keras.utils.load_img(
        "dataset/" + name + ".png",
        target_size=(64, 64)
    )
    img = tf.keras.utils.img_to_array(img)
    images.append(img)

images = np.array(images) / 127.5 - 1

# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(105,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(64 * 64 * 3, activation="tanh"),
    tf.keras.layers.Reshape((64, 64, 3))
])

# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

discriminator.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# Generate an image for a class
noise = np.random.randn(1, 100)
label = tf.keras.utils.to_categorical([3], 5)  # 2 = cat

input_data = np.concatenate([noise, label], axis=1)
image = generator.predict(input_data, verbose=0)[0]

# Display generated image
plt.imshow(np.clip((image + 1) / 2, 0, 1))
plt.title("Generated Image: dog")
plt.axis("off")
plt.show()