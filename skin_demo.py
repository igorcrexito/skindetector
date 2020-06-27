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
batch_size = 10
channels = 3

#supportive images
supportive_width = 14
supportive_height = 14
supportive_channels = 3
max_number_of_supportive_images = 1024

#base paths to load data
support_path = 'dataset/support_dataset/'

if __name__ == "__main__":

    #loading models
    print('--- Loading pre-trained models ---')
    cust = {'InstanceNormalization': InstanceNormalization}
    autoencoder_skin = load_model("model_" + "skin" + ".h5", cust)
    
    print('--- Reading support dataset ---')
    skin_rgb_sequence, skin_hsv_sequence, skin_ycrcb_sequence, no_skin_rgb_sequence, no_skin_hsv_sequence, no_skin_ycrcb_sequence = fr.read_supporting_files(support_path, supportive_width, supportive_height, max_number_of_supportive_images)
    
    print('--- Starting camera do capture videos ---')
    cap = cv2.VideoCapture(0) #starts the custom camera of your computer
    while(True): 
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        #resizing frame and computing complementary color spaces
        frame = cv2.resize(frame,(width,height))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        #normalizing inputs
        hsv_frame[0] = hsv_frame[0]/179.0
        hsv_frame[1] = hsv_frame[1]/255.0
        hsv_frame[2] = hsv_frame[2]/255.0
        frame = frame/255
        ycrcb_frame = ycrcb_frame/255 
        
        #predicting skin
        predicted_skin = autoencoder_skin.predict([[frame], [hsv_frame], [ycrcb_frame], [skin_hsv_sequence]])
        predicted_skin = predicted_skin[0]*255
        
        # Display the resulting frame
        cv2.imshow('detected_skin',predicted_skin)
        cv2.imshow('original image',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    