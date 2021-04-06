"""
Author : ianha
25/03/2021
Colorization autoencoder
"""

from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave
from skimage.io import imshow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image, ImageOps

##########################################################################
#GENERAL COMMAND
##########################################################################
Create_Dataset = False
Train = False
Transfer_learning=True
pretrained_model='colorize_autoencoder_Apple200.model'
Test = True
path = 'dataset'
##########################################################################





##########################################################################
#Create dataset by downloading online using API library
##########################################################################

if(Create_Dataset == True):
        
    from bing_image_downloader import downloader
    
    downloader.download("portrait", limit=20,  output_dir='dataset', 
                        adult_filter_off=True, force_replace=False, timeout=60)
##########################################################################





##########################################################################
if (Train == True):
    #Normalize images : divide by 255
    #pixel value from [0:255] -> [0:1]
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    
    #Resize images to the correct shape for input the autoencoders
    train = train_datagen.flow_from_directory(path, 
                                              target_size=(256, 256), 
                                              class_mode=None)
    #print keras model structure
    print(train)
    
    #Convert from RGB to Lab
    
    X =[] #input vector
    Y =[] #output vector
    for img in train[0]:
      try:
          lab = rgb2lab(img)            #convert to Lab image
          X.append(lab[:,:,0])          #the input is only the gray scale -> b&w image
          Y.append(lab[:,:,1:] / 128)   #The output is the colorization matrices range from -127 to 128
          #normalize pixels colo from [-127;128] to [-1;1] by dividing by 128
      except:
         print('error')
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape+(1,)) #dimensions to be the same for X and Y
    print(X.shape)
    print(Y.shape)
    
    
    if (Transfer_learning==True):
        
        #load model
        model = tf.keras.models.load_model(pretrained_model,
                                       custom_objects=None,
                                       compile=True)
    else:
                                  
        #NETWORK CREATION
        model = Sequential()
        
        #Encoder
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(256, 256, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3,3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3,3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
        
        #Decoder
        #Note: For the last layer we use tanh instead of Relu. 
        #This is because we are colorizing the image in this layer using 2 filters, A and B.
        #A and B values range between -1 and 1 so tanh (or hyperbolic tangent) is used
        #as it also has the range between -1 and 1. 
        #Other functions go from 0 to 1.
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.compile(optimizer='adam', loss='mse' , metrics=['accuracy'])
        model.summary()
    
    #Train model
    model.fit(X,Y,validation_split=0.1, epochs=50)
    #Save model
    model.save('colorize_autoencoder.model')

##########################################################################
    
    
def Gris(image):
    print(image.shape) #393, 700, 4 (composants de l'image : lignes, colonne, plans 4 pour rouge, vert, bleu, transparence)
    n=image.shape[0]       #accés au premier composant : n est le nb de lignes (393)
    m=image.shape[1]       #accés au second composant : m est le nb de colonnes (700)
    IMG=np.zeros((n,m)) #création d'une matrice IMG de n lignes et de m colonnes de valeurs 0
    for i in range(n):
        for j in range(m):  #pour chaque pixel de chaque ligne de chaque colonne
            pix=image[i,j]     #on retrouve 4 composants rangés dans une liste avec pix[0]=valeur rouge ...
                            # In [44]: IM[200,100]
                            # Out[44]: Image([126, 139, 117, 255], dtype=uint8)            
            IMG[i,j]=0.299*pix[0]+0.587*pix[1]+0.114*pix[2]#on reconstruit IMG avec les valeurs de pix et
    return IMG              # la formule donnée dans l'énoncé pour retrouver la valeur de gris
     
    

##########################################################################
if (Test == True):
    
    #load model
    model = tf.keras.models.load_model('colorize_autoencoder_sunset500.model',
                                   custom_objects=None,
                                   compile=True)
    #image path to test
    path += '/forest/2131.jpg'   #relative path to image
    
    img1_color=[]               #empty list
    
    img1=img_to_array(load_img(path))   #load as array instead of object
    img1 = resize(img1 ,(256,256))      #resize to input layer size 256x256
    img1_color.append(img1)
    
    #convert to lab space and extrat only L matrix [0]
    img1_color = np.array(img1_color, dtype=float)
    img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
    img1_color = img1_color.reshape(img1_color.shape+(1,))
    
    #predict with grey scale image
    output1 = model.predict(img1_color)
    output1 = output1*128       #un normalize : [-1:1] -> [-128:128]
    
    #recompose output image with L,A & B matrices for lab space
    result = np.zeros((256, 256, 3))
    result[:,:,0] = img1_color[0][:,:,0]
    result[:,:,1:] = output1[0]
    imsave("result.png", lab2rgb(result))
    
    #imshow(lab2rgb(result))
    
    fig = plt.figure(figsize=[15,15])
    plt.subplot(1,3,1)
    image = img.imread(path)
    plt.imshow(image)
    plt.title("Original image")
    
    
    plt.subplot(1,3,2)
    image = img.imread(path)
    
    plt.imshow(Gris(image),cmap="gray")
    plt.title("Gray scale image")
    
    plt.subplot(1,3,3)
    image = img.imread("result.png")
    plt.imshow(image)
    plt.title("Colorized image \n output of autoencoder")