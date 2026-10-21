import os
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input

from tensorflow.keras.layers import GlobalAveragePooling2D,Dense
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

folder = 'dataset'

classes = ['bird','car','cat','dog','horse']

X = []
y = []

for  lable,name in enumerate(classes):
    path= os.path.join(folder,name+".png")
    
    img = tf.keras.utils.load_img(path,target_size=(224,224))
    img = tf.keras.utils.img_to_array(img)
    
    X.append(img)
    y.append(lable)
    
X = np.array(X)
y = np.array(y)

X = preprocess_input(X)

base = VGG16(weights="imagenet",include_top=False)
base.trainable = False

feature_model = Sequential([
    base,
    GlobalAveragePooling2D()
])
features = feature_model.predict(X,verbose=0)

X_train, X_test, y_train,y_test = train_test_split(
    features,y, train_size = 0.8, random_state= 42
)

model=Sequential([
    tf.keras.Input(shape=(512,)),
    Dense(32,activation="relu"),
    Dense(5,activation="softmax")
])

model.compile(
    optimizer = 'adam',
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
              )

model.fit(X_train,y_train,epochs = 20, verbose = 1)

prediction = model.predict(X_test,verbose = 0)
y_prd = np.argmax(prediction,axis =1)

print(confusion_matrix(y_test,y_prd,labels=range(5)))

print(classification_report(
    y_test,y_prd,labels=range(5),target_names= classes,
    zero_division=0
))
