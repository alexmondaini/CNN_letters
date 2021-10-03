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


# Read in excel labels then get a numpy flat array with labels row-wise. 
df = pd.read_excel('label.xlsx',index_col=None,usecols=[*range(1,113)],skiprows=[*range(1,4)],engine='openpyxl')
# we have ambiguoues labels which we can set to missing values and then impute those missing values by the most frequent letter.
df = df.replace('BLANK',np.nan)
df = df.replace('A,D',np.nan)
df = df.replace('E',np.nan)

# to numpy
labels = df.to_numpy().ravel('C')
# How many unique labels we have, considering nan is not a label
n_labels = len(set(filter(lambda x: x==x,set(labels))))

# Impute
imputer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent',missing_values=np.nan)),
    ('encoder',OneHotEncoder(sparse=False))
])
# transform
labels = imputer.fit_transform(labels.reshape(-1,1))

# Read in images
path = 'questions'
n_folders = len(os.listdir(path))
images = []
IMG_SIZE = (160,160)

for folder in range(1,n_folders+1):
    pic_list = ns.natsorted(os.listdir(path+'/'+str(folder)))
    for i in pic_list:
        current_image = cv2.imread(path+'/'+str(folder)+'/'+i)
        current_image = cv2.resize(current_image, (160,160))
        images.append(current_image)
      
# Transform images list into numpy array
images = np.array(images)

# Split data

X_train, X_test, y_train, y_test = train_test_split(images,labels,test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train,test_size=0.2)

# Define train and validation dataset and Batch images for performance

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(32)
validation_dataset = tf.data.Dataset.from_tensor_slices((X_validation, y_validation))
validation_dataset = validation_dataset.batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(32)

#Autotune

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Data Augmentation

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Rescale pixel values to MobilenetV2

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Create the base model from the pre-trained model MobileNet V2

IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# Feature extract 

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)

# Freeze convolutional base

base_model.trainable = False

# Take a look at the base model architecture
base_model.summary()

# Add classification head

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

# Add dense layer with activation function
prediction_layer = tf.keras.layers.Dense(n_labels,activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)

# Build model

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# Compile

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# Inital loss and accuracy

initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# Train

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

# Fine-tune by un-freezing the top layers

base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

# Compile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001/10),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# Train the fine-tuned model
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)

# Finallly test the model

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', loss)
print('Test accuracy :', accuracy)
