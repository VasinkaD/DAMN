import numpy as np
import tifffile

##########################################

#Subtract minimum and norm to unit sum
def Normalize_data(data):
    #Subtract minima
    data_subbed = data - np.expand_dims(data.min(axis=(-1,-2)), axis=(-1,-2))
    
    #Normalize to unit sum
    norms = data_subbed.sum(axis=(-1,-2))
    data_in = data_subbed / np.expand_dims(norms, axis=(-1,-2))
    
    #Normalize images of other sizes than the default 50x50 
    data_in_renormed = data_in * (data.shape[-2]*data.shape[-1]) / (50*50)
    
    return data_in_renormed, norms

##########################################

def Load_NPC_stack(path):
    frames = tifffile.imread(path).astype(float)
    frames_50k = frames[-50000:, 35:85, 35:85]

    frames_5k = frames_50k[:5000]
    
    frames_5k_normed, norms_5k = Normalize_data(frames_5k)
    
    return frames_5k_normed, norms_5k

##########################################
