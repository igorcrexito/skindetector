import numpy as np
from keras.models import model_from_json
from keras import backend as K
import operator
from keras.models import Model
from keras.optimizers import Adam
from scipy.spatial import distance
import matplotlib.pyplot as plt
import itertools
import cv2

def computeIoU(image, groundtruth_mask, width, height):
    
    image = np.asarray(image)
    image = np.reshape(image, (width, height, 1))
    ret,image = cv2.threshold(image,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    groundtruth_mask = np.asarray(groundtruth_mask)
    groundtruth_mask = np.reshape(groundtruth_mask, (width, height, 1))
    ret,groundtruth_mask = cv2.threshold(groundtruth_mask,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    intersection = 0
    union = 0

    for i in range(0, width):
        for j in range(0, height):  
            if ((groundtruth_mask[i,j] == 1) and (image[i,j] == 1)):
                intersection = intersection + 1
            
            if ((groundtruth_mask[i,j] == 1) or (image[i,j] == 1)):
                union = union + 1
    
    #cv2.imshow('skin',image*255)
    #cv2.imshow('rgb_image',groundtruth_mask*255)
        
    return intersection/union