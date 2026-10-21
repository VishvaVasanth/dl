import os 
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input

from tensorflow.keras.layers import Embedding,LSTM,Dense,Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

file = ['bird','cat','car','dog','horse']

caption = ["Start a photo of "+ name + " end" for name in file]
images = []

for name in file:
    img = tf.keras.utils.load_img("dataset/"+name+".png", target_size=(224,224))
    
    images.append(tf.keras.utils.img_to_array(img))
images = preprocess_input(np.array(images))

cnn = VGG16(weights="imagenet",include_top=False,pooling ="avg")
features = cnn.predict(images,verbose=0)
print(features.shape)

#token

token = Tokenizer(filters="")
token.fit_on_texts(caption)
seq = token.texts_to_sequences(caption)

X = pad_sequences([s[:-1] for s in seq])
y = pad_sequences([s[1:] for s in seq])

vocab = len(token.word_index)+1

#lstm

img_input = Input(shape=(512,))

text_input = Input(shape=(X.shape[1],))

img = tf.keras.layers.RepeatVector(X.shape[1])(
    Dense(64,activation="relu")(img_input)
    )
text = Embedding(vocab,64)(text_input)
combined = tf.keras.layers.Concatenate()([img,text])
lstm = LSTM(64,return_sequences=True)(combined)
output = Dense(vocab,activation="softmax")(lstm)

model = Model([img_input,text_input],output)
model.compile(
    optimizer="adam",
    loss = "sparse_categorical_crossentropy"
)
model.fit([features,X],
          np.expand_dims(y,axis = -1),
          epochs=20)

print("model Train over")

for name in file:
    plt.figure(figsize = (5,5))
    plt.imshow(plt.imread("dataset/"+name+".png"))
    plt.title("a photo of "+name+".")
    plt.axis("off")
    plt.show()


