import cv2
import os
import numpy as np

def read_training_files(base_path, resized_width, resized_height):
    
    #defining list to store input/output data
    rgb_sequence = []
    hsv_sequence = []
    ycrcb_sequence = []
    output_sequence = []
    background_sequence = []
    
    #producing composite paths to find images
    composite_color_path = base_path + 'color/'
    composite_gt_path = base_path + 'gt/'
    
    #retrieving list of images
    image_list = os.listdir(composite_color_path)
    groundtruth_list = os.listdir(composite_gt_path)

    #getting number of images inside the folders
    number_of_images = len(image_list)
    
    #iterating over images
    for i in range(0, number_of_images):
        rgb_image = cv2.imread(composite_color_path + image_list[i]) #reading original input data     
        output_mask = cv2.cvtColor(cv2.imread(composite_gt_path + groundtruth_list[i]), cv2.COLOR_BGR2GRAY)  #reading GT data
        
        #converting images to different colorspaces
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        ycrcb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YCrCb)
        
        #resizing images to fit model input
        rgb_image = cv2.resize(rgb_image,(resized_width,resized_height))
        hsv_image = cv2.resize(hsv_image,(resized_width,resized_height))
        ycrcb_image = cv2.resize(ycrcb_image,(resized_width,resized_height))  
        output_mask = cv2.resize(output_mask,(resized_width,resized_height))
        
        
        #normalizando HSV image
        hsv_image[0] = hsv_image[0]/179.0
        hsv_image[1] = hsv_image[1]/255.0
        hsv_image[2] = hsv_image[2]/255.0
        
        rgb_image = rgb_image/255
        ycrcb_image = ycrcb_image/255 
        output_mask = cv2.normalize(output_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        rgb_sequence.append(rgb_image)
        hsv_sequence.append(hsv_image)
        ycrcb_sequence.append(ycrcb_image)
        output_sequence.append(output_mask)
        background_sequence.append(1-output_mask)
        
    return rgb_sequence, hsv_sequence, ycrcb_sequence, output_sequence, background_sequence
