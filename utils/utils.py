import numpy as np
from keras.models import model_from_json
from keras import backend as K
import operator
from keras.models import Model
from keras.optimizers import Adam
from scipy.spatial import distance
import matplotlib.pyplot as plt
import itertools

def save_weights(model):
    model_json = model.to_json()
    with open("weights.json", "w") as json_file:
        json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights("weights.h5")
    print("Saved model to disk")

def load_weights():
    json_file = open("weights.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("weights.h5")
 
    return loaded_model
        