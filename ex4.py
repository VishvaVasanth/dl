import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate
from tensorflow.keras.models import Model


# 1. Images

images = [
    "bird.png",
    "car.png",
    "cat.png",
    "dog.png",
    "horse.png"
]


# 2. Captions

captions = [
    "<start> a photo of a bird <end>",
    "<start> a photo of a car <end>",
    "<start> a photo of a cat <end>",
    "<start> a photo of a dog <end>",
    "<start> a photo of a horse <end>"
]


# 3. VGG16

vgg = VGG16(
    weights="imagenet",
    include_top=False
)

cnn = Model(
    vgg.input,
    tf.keras.layers.GlobalAveragePooling2D()(vgg.output)
)

print("VGG16 loaded")


# 4. Extract image features

features = []

for image in images:

    img = load_img(image, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature = cnn.predict(img, verbose=0)

    features.append(feature[0])

features = np.array(features)

print("Feature shape:", features.shape)


# 5. Tokenize captions

tokenizer = Tokenizer(filters="")

tokenizer.fit_on_texts(captions)

seq = tokenizer.texts_to_sequences(captions)

max_length = max(len(x) for x in seq)

vocab = len(tokenizer.word_index) + 1

print("Vocabulary:", tokenizer.word_index)


# 6. Create training data

X_image = []
X_text = []
Y = []
print(seq)
for i in range(len(images)):

    s = seq[i]

    for j in range(1, len(s)):

        X_image.append(features[i])

        X_text.append(
            pad_sequences(
                [s[:j]],
                maxlen=max_length,
                padding="pre"
            )[0]
        )

        Y.append(s[j])
print(X_image,X_text,Y )

X_image = np.array(X_image)
X_text = np.array(X_text)
Y = np.array(Y)



# 7. CNN + LSTM

image_input = Input(shape=(512,))
text_input = Input(shape=(max_length,))

image = Dense(128, activation="relu")(image_input)

image = tf.keras.layers.RepeatVector(max_length)(image)

text = Embedding(vocab, 128)(text_input)

x = Concatenate()([image, text])

x = LSTM(128)(x)

output = Dense(
    vocab,
    activation="softmax"
)(x)

model = Model(
    [image_input, text_input],
    output
)


# 8. Train

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

model.fit(
    [X_image, X_text],
    Y,
    epochs=100,
    verbose=1
)


# 9. Generate caption

words = {
    v: k
    for k, v in tokenizer.word_index.items()
}


def generate_caption(feature):

    caption = "<start>"

    for i in range(max_length):

        s = tokenizer.texts_to_sequences(
            [caption]
        )[0]

        s = pad_sequences(
            [s],
            maxlen=max_length,
            padding="pre"
        )

        prediction = model.predict(
            [feature.reshape(1, 512), s],
            verbose=0
        )

        word = words.get(
            np.argmax(prediction),
            ""
        )

        if word == "<end>":
            break

        caption += " " + word

    return caption.replace(
        "<start>",
        ""
    ).strip()


# 10. Show images and captions

for i in range(len(images)):

    result = generate_caption(
        features[i]
    )

    print("\nImage:", images[i])
    print("Expected:", captions[i])
    print("Generated:", result)

    plt.imshow(
        load_img(images[i])
    )

    plt.axis("off")

    plt.title(
        "Expected: " + captions[i] +
        "\nGenerated: " + result
    )

    plt.show()