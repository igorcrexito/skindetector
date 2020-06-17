import utils.file_reader as fr
import model.autoencoder as model
import cv2
import numpy as np
import utils.utils as utils
import os
import utils.tifffiles as tiff

#defining some hyperparameters/parameters
width = 224
height = 224
batch_size = 26
channels = 3
base_path = 'dataset/Pratheepan/'
test_path = 'dataset/ECU/'
number_of_epochs = 1000

if __name__ == "__main__":
    
    print('--- Reading input data ---')
    rgb_sequence, hsv_sequence, ycrcb_sequence, output_sequence, background_sequence = fr.read_training_files(base_path, width, height)
    
    #assembling skin detector autoencoder
    print('--- Assembling a skin detector autoencoder')
    autoencoder_skin = model.create_skin_detector_model(width, height, channels)
    autoencoder_background = model.create_skin_detector_model(width, height, channels)
    
    #creating a composite model to correlate both autoencoders
    print('--- Assembling composite model')
    image_shape = (height,width,channels)
    composite_model, skin_model, background_model = model.define_composite_model(autoencoder_background, autoencoder_skin, image_shape)
    
    #fitting the model
    print('--- Training model with original images and binary masks ---')
    skin_model, background_model = model.train_skin_background_models(composite_model, skin_model, background_model, rgb_sequence, hsv_sequence, ycrcb_sequence, output_sequence, background_sequence, width, height, channels, batch_size, number_of_epochs)
    
    #predicting in a different dataset
    print('--- Reading test data ---')
    rgb_sequence, hsv_sequence, ycrcb_sequence, output_sequence, background_sequence = fr.read_training_files(test_path, width, height)
    
    #loading models
    print('--- Loading pre-trained models ---')
    skin_model = model.load_weights('skin')
    background_model = model.load_weights('background')
    
    for i in range(0, len(rgb_sequence)):
        output_skin = skin_model.predict([[rgb_sequence[i]],[hsv_sequence[i]], [ycrcb_sequence[i]]])
        output_background = background_model.predict([[rgb_sequence[i]], [hsv_sequence[i]], [ycrcb_sequence[i]]])
        
        cv2.imshow('background',output_background[0])
        cv2.imshow('skin',output_skin[0])
        cv2.imshow('rgb_image',rgb_sequence[i])
        cv2.waitKey(0);                                          