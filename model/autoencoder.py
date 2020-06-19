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

def create_skin_detector_model(width, height, channels, supportive_width, supportive_height, supportive_channels, max_number_of_supportive_images):
    input_rgb = Input(shape=(width,height,channels), name='rgb_input')
    input_hsv = Input(shape=(width,height,channels), name='hsv_input')
    input_ycrcb = Input(shape=(width,height,channels), name='ycrcb_input')
    
    #contractive path of the architecture -------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------
    encoding_rgb = input_merging(input_rgb)
    encoding_hsv = input_merging(input_hsv)
    encoding_ycrcb = input_merging(input_ycrcb)
    
    mixed_representation = concatenate([encoding_rgb, encoding_hsv, encoding_ycrcb], axis = 3) #mixed = 112x112x128
    print(np.shape(mixed_representation))
    
    #first convolutional block
    encoding = Conv2D(32, (3,3), activation='relu', padding='same')(mixed_representation)
    encoding = Conv2D(32, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(32, (3,3), activation='relu', padding='same')(encoding)
    encoding_first = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           border_mode='valid')(encoding) #encoding 56x56x64
    
    #second convolutional block
    encoding = Conv2D(64, (3,3), activation='relu', padding='same')(encoding_first)
    encoding = Conv2D(64, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(64, (3,3), activation='relu', padding='same')(encoding)
    encoding_second = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           border_mode='valid')(encoding) #encoding 28x28x64
    
    #third convolutional block
    encoding = Conv2D(128, (3,3), activation='relu', padding='same')(encoding_second)
    encoding = Conv2D(128, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(128, (3,3), activation='relu', padding='same')(encoding)
    encoding_third = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           border_mode='valid')(encoding) #encoding 14x14x128
    
    #expansive path of the architecture -------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------
    
    #first convolutional block
    encoding = Conv2D(128, (3,3), activation='relu', padding='same')(encoding_third)
    encoding = Conv2D(128, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(64, (3,3), activation='relu', padding='same')(encoding)
    encoding = UpSampling2D(size=(2, 2), name='up1')(encoding) #encoding 28x28x128
    
    #second convolutional block
    encoding = concatenate([encoding_second, encoding], axis = 3) #28x28x128
    encoding = Conv2D(128, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(64, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(64, (3,3), activation='relu', padding='same')(encoding)
    encoding = UpSampling2D(size=(2, 2), name='up2')(encoding) #encoding 56x56x64
    
    #third convolutional block
    encoding = concatenate([encoding_first, encoding], axis = 3) #56x56x128
    encoding = Conv2D(128, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(96, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(96, (3,3), activation='relu', padding='same')(encoding)
    encoding = UpSampling2D(size=(2, 2), name='up3')(encoding) #encoding 112x112x32
    encoding = InstanceNormalization(axis=-1)(encoding) #instance normalization before merging with standard patterns
    
    #last convolutional block + auxiliary input
    input_supportive = Input(shape=(max_number_of_supportive_images, supportive_width, supportive_height, supportive_channels), name='supportive_hsv') #1024x14x14x3
    supportive_encoding = Conv3D(12, (3,3,3), activation='sigmoid', padding='same')(input_supportive)
    supportive_encoding = Conv3D(12, (3,3,3), activation='sigmoid', padding='same')(supportive_encoding)
    supportive_encoding = Conv3D(6, (3,3,3), activation='sigmoid', padding='same')(supportive_encoding) #1204224
    supportive_encoding = Reshape((112, 112, 96))(supportive_encoding)
    encoding = subtract([encoding, supportive_encoding]) #112x112x96
    encoding = UpSampling2D(size=(2, 2), name='up4')(encoding) #encoding 224x224x96
    
    encoding = Dropout(0.5)(encoding)
    encoding = Conv2D(32, (3,3), activation='sigmoid', padding='same')(encoding)
    encoding = Dropout(0.5)(encoding)
    encoding = Conv2D(8, (3,3), activation='sigmoid', padding='same')(encoding)
    output = Conv2D(1, (3,3), activation='sigmoid', padding='same')(encoding)
    
    autoencoder = Model([input_rgb, input_hsv, input_ycrcb, input_supportive], [output])

    return autoencoder


def fit_skin_model(model, rgb_sequence, hsv_sequence, ycrcb_sequence, skin_hsv_sequence, no_skin_hsv_sequence, output_vector, background_vector, width, height, channels, batch_size, number_of_epochs):
    
    #reshaping inputs to be suitable for the model
    rgb_sequence = np.reshape(rgb_sequence, (len(rgb_sequence), width, height, channels))
    hsv_sequence = np.reshape(hsv_sequence, (len(hsv_sequence), width, height, channels))
    ycrcb_sequence = np.reshape(ycrcb_sequence, (len(ycrcb_sequence), width, height, channels))
    output_vector = np.reshape(output_vector, (len(output_vector), width, height, 1)) #channels are always 1 => binary masks
    background_vector = np.reshape(background_vector, (len(background_vector), width, height, 1)) #channels are always 1 => binary masks
    composite_out = np.logical_and(output_vector, background_vector)
    
    #adjusting skin/no skin samples to fit the model
    no_skin_hsv_vector = np.zeros(((len(rgb_sequence), len(no_skin_hsv_sequence), 14, 14, 3)))
    skin_hsv_vector = np.zeros(((len(rgb_sequence), len(no_skin_hsv_sequence), 14, 14, 3)))
    
    for i in range(0, len(rgb_sequence)):
        no_skin_hsv_vector[i] = np.array(no_skin_hsv_sequence)
        skin_hsv_vector[i] = np.array(skin_hsv_sequence)
    
    model.fit([np.array(rgb_sequence),np.array(hsv_sequence),np.array(ycrcb_sequence), np.array(skin_hsv_vector), np.array(rgb_sequence), np.array(hsv_sequence), np.array(ycrcb_sequence), np.array(no_skin_hsv_vector)], [np.array(output_vector), np.array(background_vector), np.array(composite_out)], 
              epochs=number_of_epochs, 
              batch_size=batch_size, 
              shuffle=True)
                
    return model

def input_merging(input):
    
    #encoding inputs and performing a merging considering a multi-scale conv. strategy
    encoding1 = Conv2D(32, (3,3), activation='relu', padding='same')(input)
    encoding1 = Conv2D(32, (3,3), activation='relu', padding='same')(encoding1)
    encoding1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           border_mode='valid')(encoding1)
    
    encoding2 = Conv2D(16, (5,5), activation='relu', padding='same')(input)
    encoding2 = Conv2D(16, (5,5), activation='relu', padding='same')(encoding2)
    encoding2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           border_mode='valid')(encoding2)
    
    encoding3 = Conv2D(32, (7,7), activation='relu', padding='same')(input)
    encoding3 = Conv2D(32, (7,7), activation='relu', padding='same')(encoding3)
    encoding3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           border_mode='valid')(encoding3)
    
    encoding4 = Conv2D(32, (11,11), activation='relu', padding='same')(input)
    encoding4 = Conv2D(32, (11,11), activation='relu', padding='same')(encoding4)
    encoding4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           border_mode='valid')(encoding4)
    encoding = concatenate([encoding1, encoding2, encoding3, encoding4], axis = 3)
    
    return encoding

def define_composite_model(background_model, skin_model, image_shape, supportive_image_shape):
    
    #defining input dimensions
    input_skin1 = Input(shape=image_shape)
    input_skin2 = Input(shape=image_shape)
    input_skin3 = Input(shape=image_shape)
    input_skin_supportive = Input(shape=supportive_image_shape)
    
    input_background1 = Input(shape=image_shape)
    input_background2 = Input(shape=image_shape)
    input_background3 = Input(shape=image_shape)
    input_background_supportive = Input(shape=supportive_image_shape)
    
    #gathering outputs
    skin_out = skin_model([input_skin1,input_skin2, input_skin3, input_skin_supportive]) #setting dimension for the 4 inputs
    background_out = background_model([input_background1, input_background2, input_background3, input_background_supportive])
    composite_out = Lambda(dice_layer)([skin_out, background_out])
    composite_shape = np.shape(composite_out)
    composite_out = Reshape((composite_shape[1],composite_shape[2],1))(composite_out)
    
    composite_model = Model([input_skin1, input_skin2, input_skin3, input_skin_supportive, input_background1, input_background2, input_background3, input_background_supportive], [skin_out, background_out, composite_out])
    opt = Adam(lr=0.0005, beta_1=0.9, beta_2 = 0.999)
    
    #huber loss on autoencoders -> low converging behavior // mse considering dice difference -> fast convergence
    composite_model.compile(loss=[huber_loss, huber_loss, 'mse'], optimizer=opt)
    composite_model.summary()
    return composite_model, skin_model, background_model

#this method defines the training of a composite model
def train_skin_background_models(composite_model, skin_model, background_model, rgb_sequence, hsv_sequence, ycrcb_sequence, skin_hsv_sequence, no_skin_hsv_sequence, output_vector, background_vector, width, height, channels, batch_size, number_of_epochs):

    composite_model = fit_skin_model(composite_model, rgb_sequence, hsv_sequence, ycrcb_sequence, skin_hsv_sequence, no_skin_hsv_sequence, output_vector, background_vector, width, height, channels, batch_size, number_of_epochs)
    
    #saving weights of both models
    skin_model_filepath = "model_" + "skin" + ".h5"
    skin_model.save(skin_model_filepath)
    
    background_model_filepath = "model_" + "background" + ".h5"
    background_model.save(background_model_filepath)
    
    return skin_model, background_model
    
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
    return 1-dice_coef(y_true, y_pred) 

def dice_layer(tensor):
    return 1-dice_coef(tensor[0], tensor[1])
            
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
