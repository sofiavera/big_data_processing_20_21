############################################################################### IMPORT IMAGES ###############################################################################

from google.colab import drive
drive.mount('/content/gdrive')

import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/MyDrive/Kaggle"

!kaggle datasets download -d kmader/rsna-bone-age

!ls
!unzip \*.zip  && rm *.zip

############################################################################## IMPORT LIBRARIES #############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from skimage.io import imread

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split

plt.rcParams['image.cmap'] = 'gray'

################################################################################ PREPROCESSING ###############################################################################


data_df = pd.read_csv('boneage-training-dataset.csv')
#test_df = pd.read_csv('boneage-test-dataset.csv')
#print(len(train_df), len(test_df))

image_size=224
add_png= lambda x : str(x) + '.png'
data_df['id']= data_df['id'].apply(add_png)

#data_df.head(3)

## SEPARATE DATA_DF INTO TRAIN AND TEST
train_df, test_df = train_test_split(data_df,test_size = 0.2,random_state = 2018)
print('train', train_df.shape[0], 'test', test_df.shape[0])

#TRAIN IMAGE GENERATOR
data_gen=ImageDataGenerator(
               rescale = 1./255,
               validation_split=0.20
            )

#TEST IMAGE GENERATOR
data_gen_test=ImageDataGenerator(
                preprocessing_function=preprocess_input,
            )

## BIGGER BATCH SIZE 
## BATCH NORMALIZATION (The inputs to individual layers in a neural network can be normalized to speed up training. This process, called Batch Normalization, attempts to resolve an issue in neural networks called internal covariate shift.)

train_generator=data_gen.flow_from_dataframe(
    dataframe=train_df,
    directory="boneage-training-dataset/boneage-training-dataset/",
    x_col="id",
    y_col="boneage",
    subset="training",
    batch_size=32, 
    seed=42,
    shuffle=True,
    target_size=(image_size,image_size),
    class_mode='raw',
    )

validation_generator=data_gen.flow_from_dataframe(
    dataframe=train_df,
    directory="boneage-training-dataset/boneage-training-dataset/",
    x_col="id",
    y_col="boneage",
    subset="validation",
    batch_size=256,
    seed=42,
    shuffle=True,
    target_size=(image_size,image_size),
    class_mode='raw'
)

test_generator=data_gen_test.flow_from_dataframe(
    dataframe=test_df,
    directory="boneage-training-dataset/boneage-training-dataset/",
    x_col="id", #NO SE SI LO DE LAS COLUMNAS ES NECESARIO 
    y_col="boneage",
    #subset="validation",
    #batch_size=256,
    #seed=42,
    #shuffle=True,
    target_size=(image_size,image_size),
    #class_mode='raw',
    class_mode=None
)



################################################################################# LOAD VGG16 ################################################################################


## freze pretrained weights: not train existing weights
vgg= VGG16(input_shape= (image_size,image_size, 3), weights= 'imagenet', include_top = False)
for layer in vgg.layers:
    layer.trainable= False

## keras function API to add more layers
x= Flatten()(vgg.output)
x= Dropout(0.5)(x)
x= Dense(128, activation='relu')(x)
x= BatchNormalization()(x)
prediction= Dense(1, activation='linear')(x)

bone_age_model = Model(inputs= vgg.input, outputs= prediction)
bone_age_model.compile(loss='mse', optimizer='adam', metrics=['MeanSquaredError'])
bone_age_model.summary()

##early stopping, model check point
mc = ModelCheckpoint('model_weights.h5', monitor='val_loss', mode='max', save_best_only=True)
es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

############################################################################# TRAIN MODIFIED VGG16 ############################################################################

time0 = time.time()
 
history = bone_age_model.fit(train_generator, 
                    validation_data = validation_generator,
                    epochs = 15,
                    callbacks=[mc,es])
  
time1 = time.time()

GPUtime = time1 - time0
print(GPUtime)

######################################################################### TEST ACCURACY MODIFIED VGG16 ########################################################################

## ACCURACY

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

## LOSS
plt.plot(history.history['mean_squared_error'], label='train mse')
plt.plot(history.history['val_mean_squared_error'], label='val mse')
plt.legend()
plt.show()

y_pred=bone_age_model.predict(test_generator)

y_pred

test_df

y_real= test_df["boneage"]

fig, ax1 = plt.subplots(1,1, figsize = (6,6))
ax1.plot(y_real, y_pred, 'r.', label = 'predictions')
ax1.plot(y_real, y_real, 'b-', label = 'actual')
ax1.legend()
ax1.set_xlabel('Actual Age (Months)')
ax1.set_ylabel('Predicted Age (Months)')
ax1.set_ylim(-500,500)
