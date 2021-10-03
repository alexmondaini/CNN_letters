import os
import natsort as ns
import numpy as np
import pandas as pd
import cv2 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Read in images
path = 'questions'
n_folders = len(os.listdir(path))
images = []

for folder in range(1,n_folders+1):
    pic_list = ns.natsorted(os.listdir(path+'/'+str(folder)))
    for i in pic_list:
        current_image = cv2.imread(path+'/'+str(folder)+'/'+i)
        current_image = cv2.resize(current_image, (256,256))
        images.append(current_image)
        

# Transform images list into numpy array
images = np.array(images)

# Read in excel labels then get a numpy flat array with labels row-wise. 
df = pd.read_excel('label.xlsx',index_col=None,usecols=[*range(1,113)],skiprows=[*range(1,4)],engine='openpyxl')
# we have ambiguoues labels which we can set to missing values and then impute those missing values by the most frequent letter.
df = df.replace('BLANK',np.nan)
df = df.replace('A,D',np.nan)
df = df.replace('E',np.nan)
# get np array of labels
labels = df.to_numpy().ravel('C')

# How many unique labels we have, considering nan is not a label
n_labels = len(set(filter(lambda x: x==x,set(labels))))

# Split data

X_train, X_test, y_train, y_test = train_test_split(images,labels,test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train,test_size=0.2)

# Basic image preprocessing grayscale + normalize
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255
    return img

# Apply function
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

# Add a 1 depth to the CNN
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

# Image augmentation step with keras

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

# generate the images as we go along the training process
dataGen.fit(X_train)

# One-hot encode our labels
categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent',missing_values=np.nan)),
    ('encoder',OneHotEncoder(sparse=False))
])

y_train = categorical_transformer.fit_transform(y_train.reshape(-1,1))
y_test = categorical_transformer.fit_transform(y_test.reshape(-1,1))
y_validation = categorical_transformer.fit_transform(y_validation.reshape(-1,1))

# Define CNN model
def CNNModel():
    n_Filters= 60
    sizeof_filter1= (5,5)
    sizeof_filter2= (3,3)
    sizeofPool= (2,2)
    n_nodes= 500

    model = keras.Sequential(
        [
            layers.Conv2D(n_Filters,sizeof_filter1,input_shape=(256,256,1),activation='relu'),
            layers.Conv2D(n_Filters,sizeof_filter1,activation='relu'),
            layers.MaxPool2D(pool_size=sizeofPool),
            layers.Conv2D(n_Filters//2,sizeof_filter2,activation='relu'),
            layers.Conv2D(n_Filters//2,sizeof_filter2,activation='relu'),
            layers.MaxPool2D(pool_size=sizeofPool),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(n_labels,activation='softmax')
        ]
    )
    model.compile(optimizer="Adam",loss='categorical_crossentropy',metrics=['accuracy'])
    return model


model = CNNModel()
print(model.summary())

### Time to train

batch_size_value = 50
steps_per_Epoch_value = len(X_train)//batch_size_value
epochs_value = 10

history = model.fit(dataGen.flow(X_train,y_train,
                                batch_size=batch_size_value),
                                steps_per_epoch=steps_per_Epoch_value,
                                epochs=epochs_value,
                                validation_data=(X_validation,y_validation),
                                shuffle=1
                                )

## Evaluate and test accuracy
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy = ',score[1])

model.save('CNN_model')