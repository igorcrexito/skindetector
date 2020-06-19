import utils.file_reader as fr
import model.autoencoder as model
import cv2
import numpy as np
import utils.utils as utils
import os
import utils.tifffiles as tiff
from utils.keras_contrib import InstanceNormalization
from keras.models import load_model
import utils.metrics as metrics

#defining some hyperparameters/parameters
width = 224
height = 224
batch_size = 6
channels = 3
number_of_epochs = 40

#supportive images
supportive_width = 14
supportive_height = 14
supportive_channels = 3
max_number_of_supportive_images = 1024
augmentation = 0

#base paths to load data
Pratheephan_path = 'dataset/Pratheepan/'
ECU_path = 'dataset/ECU/'
HGR_path = 'dataset/HGR/'
support_path = 'dataset/support_dataset/'

if __name__ == "__main__":
    
    print('--- Reading input data ---')
    rgb_sequence, hsv_sequence, ycrcb_sequence, output_sequence, background_sequence = fr.read_training_files(ECU_path, width, height, augmentation)
    
    print('--- Reading support dataset ---')
    skin_rgb_sequence, skin_hsv_sequence, skin_ycrcb_sequence, no_skin_rgb_sequence, no_skin_hsv_sequence, no_skin_ycrcb_sequence = fr.read_supporting_files(support_path, supportive_width, supportive_height, max_number_of_supportive_images)
    
    #assembling skin detector autoencoder
    print('--- Assembling a skin detector autoencoder')
    autoencoder_skin = model.create_skin_detector_model(width, height, channels, supportive_width, supportive_height, supportive_channels, max_number_of_supportive_images)
    autoencoder_background = model.create_skin_detector_model(width, height, channels, supportive_width, supportive_height, supportive_channels, max_number_of_supportive_images)
    
    #creating a composite model to correlate both autoencoders
    print('--- Assembling composite model')
    image_shape = (width, height,channels)
    support_image_shape = (max_number_of_supportive_images, supportive_width, supportive_height, supportive_channels)
    composite_model, skin_model, background_model = model.define_composite_model(autoencoder_background, autoencoder_skin, image_shape, support_image_shape)
    
    #fitting the model -> comment the subsequent lines to not execute the training step
    print('--- Training model with original images and binary masks ---')
    #skin_model, background_model = model.train_skin_background_models(composite_model, skin_model, background_model, rgb_sequence, hsv_sequence, ycrcb_sequence, skin_hsv_sequence, no_skin_hsv_sequence, output_sequence, background_sequence, width, height, channels, batch_size, number_of_epochs)
    
    #predicting in a different dataset
    print('--- Reading test data ---')
    rgb_sequence, hsv_sequence, ycrcb_sequence, output_sequence, background_sequence = fr.read_training_files(Pratheephan_path, width, height)
    
    #loading models
    print('--- Loading pre-trained models ---')
    cust = {'InstanceNormalization': InstanceNormalization}
    skin_model = load_model("model_" + "skin" + ".h5", cust)
    background_model = load_model("model_" + "background" + ".h5", cust)
    
    averageIoU = 0
    
    #qualitative prediction of images -> crossdataset
    for i in range(0, len(rgb_sequence)):
        output_skin = skin_model.predict([[rgb_sequence[i]],[hsv_sequence[i]], [ycrcb_sequence[i]], [skin_hsv_sequence]])
        output_background = background_model.predict([[rgb_sequence[i]],[hsv_sequence[i]], [ycrcb_sequence[i]], [no_skin_hsv_sequence]])
                
        #cv2.imshow('groundtruth',output_sequence[i])    
        #cv2.imshow('background',output_background[0])
        #cv2.imshow('skin',output_skin[0])
        #cv2.imshow('rgb_image',rgb_sequence[i])
        
        skin_image = output_skin[0]*255
        groundtruth_image = output_sequence[i]
        
        IoU = metrics.computeIoU(skin_image.astype(np.uint8), groundtruth_image.astype(np.uint8), width, height)
        print('-----------------------------------------------------------')
        print('IoU: ' + str(IoU))
        
        #cv2.waitKey(0);                                          
        averageIoU+= IoU
        
    print('Average IoU: '+ str(averageIoU/len(rgb_sequence)))                       
