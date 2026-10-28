import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# 1. Load images
files = ["bird", "cat", "dog", "car"]
captions = ["start a photo of " + name + " end" for name in files]
images = []

for name in files:
    img = tf.keras.utils.load_img(
        "dataset/" + name + ".png", target_size=(224, 224)
    )
    images.append(tf.keras.utils.img_to_array(img))

images = preprocess_input(np.array(images))

# 2. CNN feature extraction
cnn = VGG16(weights="imagenet", include_top=False, pooling="avg")
features = cnn.predict(images, verbose=0)
print("Feature shape:", features.shape)

# 3. Tokenize captions
tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(captions)
seq = tokenizer.texts_to_sequences(captions)

X = pad_sequences([s[:-1] for s in seq])
y = pad_sequences([s[1:] for s in seq])
vocab = len(tokenizer.word_index) + 1

# 4. Build LSTM model
image_input = Input(shape=(512,))
text_input = Input(shape=(X.shape[1],))

img = tf.keras.layers.RepeatVector(X.shape[1])(
    Dense(64, activation="relu")(image_input)
)
text = Embedding(vocab, 64)(text_input)

combined = tf.keras.layers.Concatenate()([img, text])
lstm = LSTM(64, return_sequences=True)(combined)
output = Dense(vocab, activation="softmax")(lstm)

model = Model([image_input, text_input], output)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy")

# 5. Train model
model.fit([features, X], np.expand_dims(y, -1), epochs=10)

print("Caption model trained successfully!")
import matplotlib.pyplot as plt

# Display image with caption
for i, name in enumerate(files):
    plt.figure(figsize=(5, 5))
    plt.imshow(plt.imread("dataset/" + name + ".png"))
    plt.title("A photo of " + name)
    plt.axis("off")
    plt.show()