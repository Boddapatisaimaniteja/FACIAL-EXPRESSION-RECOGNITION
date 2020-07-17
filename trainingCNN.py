# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:11:18 2020

@author: bodda
"""
import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Activation,BatchNormalization

num_classes=5
batch_size=64
image_rows,image_cols=48,48

training_data_dir=r'E:\faceexpression\train'
validation_data_dir=r'E:\faceexpression\validation'

#Data preprocessing

train_datagen=ImageDataGenerator(rescale=1./255
                                 ,rotation_range=30,
                                 shear_range=0.3,
                                 zoom_range=0.4,
                                 horizontal_flip=True)

validation_datagen=ImageDataGenerator(rescale=1./255)

training_data=train_datagen.flow_from_directory(training_data_dir,color_mode='grayscale',target_size=(image_rows,image_cols),batch_size=batch_size,class_mode='categorical',shuffle=True)
validation_data=validation_datagen.flow_from_directory(validation_data_dir,color_mode='grayscale',target_size=(image_rows,image_cols),batch_size=batch_size,class_mode='categorical',shuffle=True)

#CREATING A CNN 

model=Sequential()

#block1(CONVOLUTION LAYER 1)
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(image_rows,image_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(image_rows,image_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block2(CONVOLUTION LAYER2)
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block3( CONVOLUTION LAYER 3)
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block4(CONVOLUTION LAYER 4)
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block5(FLATTENING)

model.add(Flatten())
model.add(Dense(128,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#block6(FULLY CONNECTED LAYER)
model.add(Dense(256,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#block 7(output layer)
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

# earlystopping
from keras.optimizers import RMSprop,Adam,SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

checkpoint=ModelCheckpoint(r'E:\faceexpression\fe.h5',monitor='val_loss',mode='min',save_best_only=True,verbose=1)

earlystop=EarlyStopping(monitor='val_loss',min_delta=0,patience=3,verbose=1,restore_best_weights=True)

reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)

callbacks=[earlystop,checkpoint,reduce_lr]

#compilation
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

no_training_s=24176
no_validating_s=3006
epochs=25

history=model.fit_generator(training_data,steps_per_epoch=no_training_s//batch_size,epochs=epochs,callbacks=callbacks,validation_data=validation_data,validation_steps=no_validating_s//batch_size)

plt.plot(history.history['accuracy'],c='b',label='training_acc')
plt.plot(history.history['val_accuracy'],c='r',label='val_acc')
plt.legend()