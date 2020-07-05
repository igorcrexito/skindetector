from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, concatenate, Dropout, Lambda, Reshape, Flatten, Dense, average, Conv3D, subtract, multiply, add
from keras.models import Model
import numpy as np
from keras.optimizers import *
from keras import backend as K
from keras.optimizers import Adam
import random
from keras.models import model_from_json
from utils.keras_contrib import InstanceNormalization
from keras.models import load_model
import json
from keras.utils import plot_model
import cv2


def create_unet_model(width, height, channels):
    inputs = Input(shape=(width,height,channels), name='input')
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.1)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid', padding = 'same')(conv9)

    model = Model([inputs], [conv10])
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2 = 0.999)
    
    model.compile(loss='mse', optimizer=opt)
    model.summary()
    
    return model


def define_composite_model(background_model, skin_model, image_shape):
    
    #defining input dimensions
    input_skin = Input(shape=image_shape)   
    input_background = Input(shape=image_shape)
    
    #gathering outputs
    skin_out = skin_model([input_skin])
    background_out = background_model([input_background])
    
    composite_out = Lambda(dice_layer)([skin_out, background_out])
    composite_shape = np.shape(composite_out)
    composite_out = Reshape((composite_shape[1],composite_shape[2],1))(composite_out)
    
    composite_model = Model([input_skin, input_background], [skin_out, background_out, composite_out])
    opt = Adam(lr=0.001, beta_1=0.9, beta_2 = 0.999)
    
    #huber loss on autoencoders -> low converging behavior // mse considering dice difference -> fast convergence
    composite_model.compile(loss=['mse', 'mse', 'mse'], optimizer=opt)
    composite_model.summary()
    return composite_model, skin_model, background_model

def fit_skin_model(model, rgb_sequence, output_vector, background_vector, width, height, channels, batch_size, number_of_epochs):
    
    #reshaping inputs to be suitable for the model
    rgb_sequence = np.reshape(rgb_sequence, (len(rgb_sequence), width, height, channels))
    output_vector = np.reshape(output_vector, (len(output_vector), width, height, 1)) #channels are always 1 => binary masks
    background_vector = np.reshape(background_vector, (len(background_vector), width, height, 1)) #channels are always 1 => binary masks
    composite_out = np.zeros((len(rgb_sequence), width, height, 1))
    
    model.fit([np.array(rgb_sequence), np.array(rgb_sequence)], [np.array(output_vector), np.array(background_vector), np.array(composite_out)], 
              epochs=number_of_epochs, 
              batch_size=batch_size, 
              shuffle=True)
                
    return model


def inter_coef(y_true, y_pred):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    return K.sum(K.abs(y_true * y_pred), axis=-1)
    
    
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)

def dice_loss(y_true, y_pred):
    return dice_coef(y_true, y_pred)

def dice_layer(tensor):
    #return dice_coef(tensor[0], tensor[1])
    return inter_coef(tensor[0], tensor[1])
    
def save_weights(model, model_type):
    model_json = model.to_json()
    with open("model_"+ model_type + ".json", "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights("model_" + model_type + ".h5")
    print("Saved model to disk")

def load_weights(model_type, model):
    
    #json_file = open("model_"+ model_type + ".json", 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    
    cust = {'InstanceNormalization': InstanceNormalization}
    model = model.load_weights("model_"+ model_type + ".h5", cust)

    return model    


#this method defines the training of a composite model
def train_skin_background_models(composite_model, skin_model, background_model, rgb_sequence, output_vector, background_vector, width, height, channels, batch_size, number_of_epochs):

    composite_model = fit_skin_model(composite_model, rgb_sequence, output_vector, background_vector, width, height, channels, batch_size, number_of_epochs)
    
    #saving weights of both models
    skin_model_filepath = "model_" + "unet" + ".h5"
    skin_model.save(skin_model_filepath)
    
    background_model_filepath = "model_" + "unet_back" + ".h5"
    background_model.save(background_model_filepath)
    
    return skin_model, background_model
