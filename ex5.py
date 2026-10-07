import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Concatenate
from tensorflow.keras.models import Model


# ---------------------------------------
# 1. Images and text labels
# ---------------------------------------

images = [
    "bird.png",
    "car.png",
    "cat.png",
    "dog.png",
    "horse.png"
]

texts = [
    "bird",
    "car",
    "cat",
    "dog",
    "horse"
]


# ---------------------------------------
# 2. Load real images
# ---------------------------------------

X = []

for image in images:

    img = load_img(
        image,
        target_size=(64, 64)
    )

    img = img_to_array(img)

    img = img / 127.5 - 1

    X.append(img)

X = np.array(X)

print("Real images:", X.shape)


# ---------------------------------------
# 3. Simple text encoding
# ---------------------------------------

text_vectors = np.array([
    [1, 0, 0, 0, 0],   # bird
    [0, 1, 0, 0, 0],   # car
    [0, 0, 1, 0, 0],   # cat
    [0, 0, 0, 1, 0],   # dog
    [0, 0, 0, 0, 1]    # horse
])


# ---------------------------------------
# 4. Generator
# ---------------------------------------

noise = Input(shape=(100,))
text = Input(shape=(5,))

x = Concatenate()([noise, text])

x = Dense(256, activation="relu")(x)

x = Dense(512, activation="relu")(x)

x = Dense(64 * 64 * 3, activation="tanh")(x)

output = Reshape((64, 64, 3))(x)

generator = Model(
    [noise, text],
    output
)

print("Generator created")


# ---------------------------------------
# 5. Discriminator
# ---------------------------------------

image = Input(shape=(64, 64, 3))
text = Input(shape=(5,))

x = Flatten()(image)

x = Concatenate()([x, text])

x = Dense(512, activation="relu")(x)

x = Dense(256, activation="relu")(x)

output = Dense(
    1,
    activation="sigmoid"
)(x)

discriminator = Model(
    [image, text],
    output
)

discriminator.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Discriminator created")


# ---------------------------------------
# 6. Create GAN
# ---------------------------------------

discriminator.trainable = False

noise = Input(shape=(100,))
text = Input(shape=(5,))

fake_image = generator(
    [noise, text]
)

result = discriminator(
    [fake_image, text]
)

gan = Model(
    [noise, text],
    result
)

gan.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)


# ---------------------------------------
# 7. Train GAN
# ---------------------------------------

epochs = 1000

for epoch in range(epochs):

    # Select random image
    i = np.random.randint(0, 5)

    real_image = X[i:i+1]

    real_text = text_vectors[i:i+1]

    # Random noise
    z = np.random.normal(
        0,
        1,
        (1, 100)
    )

    # Generate fake image
    fake_image = generator.predict(
        [z, real_text],
        verbose=0
    )

    # Train discriminator

    discriminator.train_on_batch(
        [real_image, real_text],
        np.ones((1, 1))
    )

    discriminator.train_on_batch(
        [fake_image, real_text],
        np.zeros((1, 1))
    )

    # Train generator

    gan.train_on_batch(
        [z, real_text],
        np.ones((1, 1))
    )

    if epoch % 100 == 0:

        print("Epoch:", epoch)


# ---------------------------------------
# 8. Generate image from text
# ---------------------------------------

# Example:
# "a photo of a dog"

dog_text = np.array([
    [0, 0, 0, 1, 0]
])


z = np.random.normal(
    0,
    1,
    (1, 100)
)

generated = generator.predict(
    [z, dog_text],
    verbose=0
)


# Convert image from [-1,1] to [0,1]

generated = (
    generated[0] + 1
) / 2


# ---------------------------------------
# 9. Display generated image
# ---------------------------------------

plt.imshow(generated)

plt.title(
    'Text: "a photo of a dog"'
)

plt.axis("off")
 
plt.show()