from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, concatenate, Dropout, Lambda, Reshape
from keras.models import Model
import numpy as np
from keras.optimizers import *
from keras import backend as K
from keras.optimizers import Adam
import random
from keras.models import model_from_json

def create_skin_detector_model(width, height, channels):
    input_rgb = Input(shape=(width,height,channels), name='rgb_input')
    input_hsv = Input(shape=(width,height,channels), name='hsv_input')
    input_ycrcb = Input(shape=(width,height,channels), name='ycrcb_input')
    
    encoding_rgb = encoding_layers(input_rgb)
    encoding_hsv = encoding_layers(input_hsv)
    encoding_ycrcb = encoding_layers(input_ycrcb)
    
    mixed_representation = concatenate([encoding_rgb, encoding_hsv, encoding_ycrcb], axis = 3)
    
    encoding = Conv2D(16, (3,3), activation='relu', padding='same')(mixed_representation)
    encoding = Conv2D(16, (3,3), activation='relu', padding='same')(encoding)
    
    encoding = Conv2D(12, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(12, (3,3), activation='relu', padding='same')(encoding)
    encoding = concatenate([mixed_representation, encoding], axis = 3)
    encoding = Conv2D(12, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(12, (3,3), activation='relu', padding='same')(encoding)
    
    encoding = UpSampling2D(size=(2, 2), name='up3')(encoding)
    
    encoding = Conv2D(12, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(6, (3,3), activation='sigmoid', padding='same')(encoding)
    output = Conv2D(1, (3,3), activation='sigmoid', padding='same')(encoding)
    
    autoencoder = Model([input_rgb, input_hsv, input_ycrcb], [output])

    
    return autoencoder


def fit_skin_model(model, rgb_sequence, hsv_sequence, ycrcb_sequence, output_vector, background_vector, width, height, channels, batch_size, number_of_epochs):
    
    #reshaping inputs to be suitable for the model
    rgb_sequence = np.reshape(rgb_sequence, (len(rgb_sequence), width, height, channels))
    hsv_sequence = np.reshape(hsv_sequence, (len(hsv_sequence), width, height, channels))
    ycrcb_sequence = np.reshape(ycrcb_sequence, (len(ycrcb_sequence), width, height, channels))
    output_vector = np.reshape(output_vector, (len(output_vector), width, height, 1)) #channels are always 1 => binary masks
    background_vector = np.reshape(background_vector, (len(background_vector), width, height, 1)) #channels are always 1 => binary masks
    composite_out = np.logical_and(output_vector, background_vector)
    
    model.fit([rgb_sequence,hsv_sequence,ycrcb_sequence, rgb_sequence,hsv_sequence,ycrcb_sequence], [output_vector, background_vector, composite_out], 
              epochs=number_of_epochs, 
              batch_size=batch_size, 
              shuffle=True, 
              validation_data=([rgb_sequence,hsv_sequence,ycrcb_sequence, rgb_sequence, hsv_sequence, ycrcb_sequence], [output_vector, background_vector, composite_out]))
                
    return model

def encoding_layers(input):
    
    #encoding input into a higher level representation - 1st group
    encoding1 = Conv2D(16, (3,3), activation='relu', padding='same')(input)
    encoding1 = Conv2D(16, (3,3), activation='relu', padding='same')(encoding1)
    encoding1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           border_mode='valid')(encoding1)
    
    encoding2 = Conv2D(16, (5,5), activation='relu', padding='same')(input)
    encoding2 = Conv2D(16, (5,5), activation='relu', padding='same')(encoding2)
    encoding2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           border_mode='valid')(encoding2)
                           
    encoding3 = Conv2D(16, (7,7), activation='relu', padding='same')(input)
    encoding3 = Conv2D(16, (7,7), activation='relu', padding='same')(encoding3)
    encoding3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           border_mode='valid')(encoding3)
    
    encoding = concatenate([encoding1, encoding2, encoding3], axis = 3)
    
    encoding = Conv2D(12, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(12, (3,3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           border_mode='valid')(encoding)
                          
    
    encoding = Conv2D(12, (3,3), activation='relu', padding='same')(encoding)
    encoding = Conv2D(12, (3,3), activation='relu', padding='same')(encoding)
    encoding = UpSampling2D(size=(2, 2))(encoding)
    
    return encoding

def define_composite_model(background_model, skin_model, image_shape):
    
    #defining input dimensions
    input_skin1 = Input(shape=image_shape)
    input_skin2 = Input(shape=image_shape)
    input_skin3 = Input(shape=image_shape)
    
    input_background1 = Input(shape=image_shape)
    input_background2 = Input(shape=image_shape)
    input_background3 = Input(shape=image_shape)
    
    #gathering outputs
    skin_out = skin_model([input_skin1,input_skin2, input_skin3]) #setting dimension for the 3 inputs
    background_out = background_model([input_background1, input_background2, input_background3])
    composite_out = Lambda(dice_layer)([skin_out, background_out])
    composite_shape = np.shape(composite_out)
    composite_out = Reshape((composite_shape[1],composite_shape[2],1))(composite_out)
    
    composite_model = Model([input_skin1, input_skin2, input_skin3, input_background1, input_background2, input_background3], [skin_out, background_out, composite_out])
    opt = Adam(lr=0.0005, beta_1=0.5)
    
    # compile model with weighting of least squares loss and L1 loss
    composite_model.compile(loss=['mse', 'mse', 'mae'], optimizer=opt)
    composite_model.summary()
    return composite_model, skin_model, background_model

#this method defines the training of a composite model
def train_skin_background_models(composite_model, skin_model, background_model, rgb_sequence, hsv_sequence, ycrcb_sequence, output_vector, background_vector, width, height, channels, batch_size, number_of_epochs):

    composite_model = fit_skin_model(composite_model, rgb_sequence, hsv_sequence, ycrcb_sequence, output_vector, background_vector, width, height, channels, batch_size, number_of_epochs)
    
    #saving weights of both models
    save_weights(skin_model, "skin")    
    save_weights(background_model, "background") 
    
    return skin_model, background_model
    
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

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

def load_weights(model_type):
    
    json_file = open("model_"+ model_type + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    loaded_model.load_weights("model_" + model_type + ".h5")
 
    return loaded_model
    