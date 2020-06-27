import cv2
import os
import numpy as np
from PIL import Image

FLIP, TRANSLATE = (0,1)

def return_images_path(base_path):
    
    #defining list to store input/output data
    rgb_path_sequence = []
    output_path_sequence = []
    
    #producing composite paths to find images
    composite_color_path = base_path + 'color/'
    composite_gt_path = base_path + 'gt/'
    
    #retrieving list of images
    image_list = os.listdir(composite_color_path)
    groundtruth_list = os.listdir(composite_gt_path)

    for i in range(0, len(image_list)):
        rgb_path_sequence.append(composite_color_path + image_list[i])
        output_path_sequence.append(composite_gt_path + groundtruth_list[i])
    
    return rgb_path_sequence, output_path_sequence

def read_training_files(base_path, resized_width, resized_height, number_of_steps = 99, current_step = 0, augmentation = 0):
    
    #defining list to store input/output data
    rgb_sequence = []
    output_sequence = []
    
    #producing composite paths to find images
    composite_color_path = base_path + 'color/'
    composite_gt_path = base_path + 'gt/'
    
    #retrieving list of images
    image_list = os.listdir(composite_color_path)
    groundtruth_list = os.listdir(composite_gt_path)

    number_of_images = len(image_list)
    images_per_step = 0
    
    #checking current step
    if number_of_steps == 99:
        images_per_step = number_of_images
    else:
        images_per_step = int(number_of_images/number_of_steps)
    
    #iterating over images
    for i in range(current_step*images_per_step, (current_step+1)*images_per_step):
        rgb_image = cv2.imread(composite_color_path + image_list[i]) #reading original input data     
        output_mask = cv2.cvtColor(cv2.imread(composite_gt_path + groundtruth_list[i]), cv2.COLOR_BGR2GRAY)  #reading GT data
        ret,output_mask = cv2.threshold(output_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        rgb_image = cv2.resize(rgb_image,(resized_width,resized_height))
        output_mask = cv2.resize(output_mask,(resized_width,resized_height))
        
        #augmenting images
        if augmentation == 1:
            rgb_flip_image, rgb_flip_output = augment_image(rgb_image, output_mask, FLIP)
            rgb_trans_image, rgb_trans_output = augment_image(rgb_image, output_mask, TRANSLATE)
            
            rgb_sequence.append(rgb_flip_image)
            rgb_sequence.append(rgb_trans_image)
            
            output_sequence.append(rgb_flip_output)
            output_sequence.append(rgb_trans_output)
            
        rgb_sequence.append(rgb_image)     
        output_sequence.append(output_mask)
        
    return normalize_and_convert_spaces(rgb_sequence, output_sequence)

def normalize_and_convert_spaces(rgb_sequence, output_sequence):
    
    rgb_list = []
    output_list = []
    hsv_sequence = []
    ycrcb_sequence = []
    background_sequence = []
    
    for i in range(0, len(rgb_sequence)):
        
        rgb_image = rgb_sequence[i]
        output_mask = output_sequence[i]
        
        try:
            #converting images to different colorspaces
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            ycrcb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YCrCb)

            #normalizando HSV image
            hsv_image[0] = hsv_image[0]/179.0
            hsv_image[1] = hsv_image[1]/255.0
            hsv_image[2] = hsv_image[2]/255.0

            rgb_image = rgb_image/255
            ycrcb_image = ycrcb_image/255 
            output_mask = output_mask/255

            rgb_list.append(rgb_image)
            hsv_sequence.append(hsv_image)
            ycrcb_sequence.append(ycrcb_image)
            output_list.append(output_mask)
            background_sequence.append(1-output_mask)
        except:
            print('Wrong number of channels')
            
    return rgb_list, hsv_sequence, ycrcb_sequence, output_list, background_sequence

def normalize_and_convert_images(rgb_image, output_image):
    
    try:
        #converting images to different colorspaces
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        ycrcb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YCrCb)

        #normalizando HSV image
        hsv_image[0] = hsv_image[0]/179.0
        hsv_image[1] = hsv_image[1]/255.0
        hsv_image[2] = hsv_image[2]/255.0

        rgb_image = rgb_image/255
        ycrcb_image = ycrcb_image/255 
        output_mask = output_mask/255

    except:
        print('Wrong number of channels')
            
    return rgb_image, hsv_image, ycrcb_image, output_mask, 1-output_mask

def augment_image(image, output_mask, transformation):
    
    if transformation == FLIP:
        return cv2.flip(image, 1), cv2.flip(output_mask, 1)
    
    elif transformation == TRANSLATE:
        
        height, width = image.shape[:2]
        offset_height, offset_width = height / 8, width / 8
        T = np.float32([[1, 0, offset_width], [0, 1, offset_height]]) 
        
        return cv2.warpAffine(image, T, (width, height)), cv2.warpAffine(output_mask, T, (width, height))
    
def read_supporting_files(base_path, resized_width, resized_height, max_number_of_skin_instances):
    
    #defining list to store skin/background data
    skin_rgb_sequence = []
    skin_hsv_sequence = []
    skin_ycrcb_sequence = []
    no_skin_rgb_sequence = []
    no_skin_hsv_sequence = []
    no_skin_ycrcb_sequence = []
    
    #producing composite paths to find images
    composite_skin_path = base_path + 'skin_samples/'
    composite_no_skin_path = base_path + 'no_skin_samples/'
    
    #retrieving list of skin samples
    skin_list = os.listdir(composite_skin_path)
    no_skin_list = os.listdir(composite_no_skin_path)

    #getting number of images inside the folders
    number_of_images = len(skin_list)
    
    #iterating over images
    for i in range(0, max_number_of_skin_instances):
        skin_image = cv2.imread(composite_skin_path + skin_list[i]) #reading skin samples   
        no_skin_image = cv2.imread(composite_no_skin_path + no_skin_list[i])  #reading no-skin samples
        
        #resizing images to fit model input
        skin_image = cv2.resize(skin_image,(resized_width,resized_height))
        no_skin_image = cv2.resize(no_skin_image,(resized_width,resized_height))
        
        #---------------------------------------------------------------
        #adjusting skin samples ----------------------------------------
        #converting skin images to different colorspaces
        skin_hsv_image = cv2.cvtColor(skin_image, cv2.COLOR_BGR2HSV)
        skin_ycrcb_image = cv2.cvtColor(skin_image, cv2.COLOR_BGR2YCrCb)
        
        #normalizando HSV image
        skin_hsv_image[0] = skin_hsv_image[0]/179.0
        skin_hsv_image[1] = skin_hsv_image[1]/255.0
        skin_hsv_image[2] = skin_hsv_image[2]/255.0
        
        skin_image = skin_image/255
        skin_ycrcb_image = skin_ycrcb_image/255 
        
        #------------------------------------------------------------------
        #adjusting no_skin samples ----------------------------------------
        no_skin_hsv_image = cv2.cvtColor(no_skin_image, cv2.COLOR_BGR2HSV)
        no_skin_ycrcb_image = cv2.cvtColor(no_skin_image, cv2.COLOR_BGR2YCrCb)
        
        #normalizando HSV image
        no_skin_hsv_image[0] = no_skin_hsv_image[0]/179.0
        no_skin_hsv_image[1] = no_skin_hsv_image[1]/255.0
        no_skin_hsv_image[2] = no_skin_hsv_image[2]/255.0
        
        no_skin_image = no_skin_image/255
        no_skin_ycrcb_image = no_skin_ycrcb_image/255 
        #-----------------------------------------------------------------
        
        skin_rgb_sequence.append(skin_image)
        skin_hsv_sequence.append(skin_hsv_image)
        skin_ycrcb_sequence.append(skin_ycrcb_image)
        
        no_skin_rgb_sequence.append(no_skin_image)
        no_skin_hsv_sequence.append(no_skin_hsv_image)
        no_skin_ycrcb_sequence.append(no_skin_ycrcb_image)
        
    return skin_rgb_sequence, skin_hsv_sequence, skin_ycrcb_sequence, no_skin_rgb_sequence, no_skin_hsv_sequence, no_skin_ycrcb_sequence

