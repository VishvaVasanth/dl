import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


from tensorflow.keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)

model = VGG16(weights='imagenet')

folder = 'dataset'

feature_model = tf.keras.Model(
    inputs = model.input,
    outputs = model.get_layer('block1_conv1').output
)

for file in os.listdir(folder):
    if file.endswith((".png",".jpg",".jpeg")):
        path = os.path.join(folder,file)
        
        img = tf.keras.utils.load_img(path,target_size=(224,224))
        img = tf.keras.utils.img_to_array(img)
        
        x = np.expand_dims(img,axis=0)
        x = preprocess_input(x.copy())
        
        prd = model.predict(x,verbose = 0)
        top5 = decode_predictions(prd,top=5)[0]
        
        for  _,lable,score in top5:
            
            print(lable, " : ", round(score*100,2), "%")
        
        
        feature = feature_model.predict(x,verbose=0)
        
        plt.figure(figsize=(8,4))
        
        plt.subplot(1,2,1)
        plt.imshow(img.astype('uint8'))
        plt.title(file)
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(feature[0,:,:,0],cmap='gray')
        plt.axis('off')
        plt.show()
    
    
