import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# Load images
names = ["bird", "car", "cat", "dog", "horse"]
images = []

for name in names:
    img = tf.keras.utils.load_img(
        "dataset/" + name + ".png", target_size=(64, 64))
    images.append(tf.keras.utils.img_to_array(img))

images = np.array(images) / 127.5 - 1
labels = tf.keras.utils.to_categorical(range(5), 5)

# Generator
z = Input(shape=(100,))
c = Input(shape=(5,))
x = Concatenate()([z, c])
x = Dense(256, activation="relu")(x)
x = Dense(64 * 64 * 3, activation="tanh")(x)
fake = Reshape((64, 64, 3))(x)
G = Model([z, c], fake)

# Discriminator
img = Input(shape=(64, 64, 3))
label = Input(shape=(5,))
x = Flatten()(img)
x = Concatenate()([x, label])
x = Dense(128, activation="relu")(x)
out = Dense(1, activation="sigmoid")(x)
D = Model([img, label], out)
D.compile(optimizer="adam", loss="binary_crossentropy")

# Combined GAN
D.trainable = False
z2 = Input(shape=(100,))
c2 = Input(shape=(5,))
fake2 = G([z2, c2])
gan = Model([z2, c2], D([fake2, c2]))
gan.compile(optimizer="adam", loss="binary_crossentropy")

# Training
for epoch in range(100):
    noise = np.random.randn(5, 100)
    fake_images = G.predict([noise, labels], verbose=0)

    D.train_on_batch([images, labels], np.ones((5, 1)))
    D.train_on_batch([fake_images, labels], np.zeros((5, 1)))
    gan.train_on_batch([noise, labels], np.ones((5, 1)))

print("Training completed!")

# Generate image from text/class
prompt = input("Enter class (bird/car/cat/dog/horse): ").lower()
if prompt not in names:
    prompt = "dog"

i = names.index(prompt)
noise = np.random.randn(1, 100)
condition = tf.keras.utils.to_categorical([i], 5)
image = G.predict([noise, condition], verbose=0)[0]

plt.imshow(np.clip((image + 1) / 2, 0, 1))
plt.title("Generated: " + prompt)
plt.axis("off")
plt.show()