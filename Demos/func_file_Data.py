import os
import numpy as np
np.random.seed(18)
import scipy
from zipfile import ZipFile
import tifffile

##########################################
##########################################

#Symmetric Point spread functions

def Gauss_function(x_, sigma_):
    Gauss_amplitude = np.exp(-(x_)**2/(2*sigma_**2))     #Amplitude is the Gaussian profile
    Gauss_intensity = Gauss_amplitude**2                 #Intesity is also Gaussian but of different width
    return Gauss_intensity

def Airy_function(r_, gamma_):
    q = 2 * r_ / gamma_
    J_1 = scipy.special.jv(1, q)                   #scipy.special.jv(v, z) calls Bessel function of the first kind of "v" for a complex "z" position
    
    safe_division = np.divide(J_1, q, out=0.5+np.zeros_like(J_1), where=q!=0)

    Airy_amplitude = 2*safe_division               #Amplitude is the Besinc function
    Airy_intensity = Airy_amplitude**2             #Intensity is the Airy disc
    return Airy_intensity

##########################################

#Symmetric Convolutions

def Convolve_Gauss(image_, c_):
    radius = int(np.ceil(3*c_))
    k = int(2*radius+1)
    #---------------------------------------------------------------------------------------------------------------------------------------------
    x_ = np.ones((k,k)) * np.linspace(-radius, radius, k)
    xx_ = np.sqrt(x_**2 + np.transpose(x_)**2)                                      #Field for kernel
    unnormed_psf_matrix = Gauss_function(xx_, c_)                                   #Gaussian kernel; size based on 3sigma rule of 99.7%
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()             #Correctly normed Gaussian kernel
    #---------------------------------------------------------------------------------------------------------------------------------------------
    convolved = scipy.signal.convolve(image_, normed_psf_matrix, mode="same")       #Convolution with the kernel
    
    return convolved

def Convolve_Airy(image_, c_):
    q_mins = np.array([3.8317,7.0156,10.1735,13.3237])       #Minima (in q space) of Airy intensity, i.e., dark rings of Airy disc -> these circles contain the power of 83.8%, 91.0%, 93.8%, x%
    n = 3                                                    #Chossing the n-th minimum to be the border of Airy kernel
    radius = int(np.ceil(c_/2 * q_mins[n-1]))                #Finding the radius (in r space) corresponding to including the n-th minima
    k = int(2*radius+1)                                      #Matrix
    #---------------------------------------------------------------------------------------------------------------------------------------------
    x_ = np.ones((k,k)) * np.linspace(-radius, radius, k)
    xx_ = np.sqrt(x_**2 + np.transpose(x_)**2)                                      #Field for kernel
    unnormed_psf_matrix = Airy_function(xx_, c_)                                    #Airy disc kernel
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()             #Correctly normed Airy disc kernel
    #---------------------------------------------------------------------------------------------------------------------------------------------
    convolved = scipy.signal.convolve(image_, normed_psf_matrix, mode="same")       #Convolution with the kernel
    
    return convolved

##########################################

#Functions to generate dataset

def Place_emitters(image_size, concentration, em_power, decrease_power=True):
    #Get the matrix field
    size = image_size[0] * image_size[1]
    field = np.zeros((size, size))
    
    #One by one, add emitters to the field
    for i in range(concentration):
        #Assign current emitter with Emitter power
        emitter_power = 0
        while emitter_power == 0:
            emitter_power = np.random.poisson(em_power)
        if decrease_power:
            decrease_factor = np.random.uniform(low = 0.1, high = 1)
            emitter_power = emitter_power * decrease_factor
        
        #Assign current emitter with its position
        field[np.random.randint(0, size), np.random.randint(0, size)] += emitter_power
    
    return field
    
def Normalize(matrix):
    #Normalize matrix to a unit sum
    if np.sum(matrix) != 0:
        normalized = matrix / np.sum(matrix)
    else:
        normalized = matrix
    
    return normalized

##########################################

def Generate_normed_dataset(num_data = 100,                                                    #Number of samples to generate
                            size = 50, upsampling_factor = 8,                                  #Image sizes
                            emit_power = 5000, noise_inten = 10,                               #Intensity and noise
                            PSF_width = 2, kernel_choice = 0,                                  #PSF
                            concentration = 50,                                                #Concentration
                            input_size_unit_norm=50, decrease_power=True, sub_min=True):       #Supporting
    data_w_blur = []
    data_clear = []
    
    #Iteration cycle generating one matrix field at the time
    for y in range(num_data):
        #Emitters are placed within the matrix field and associated with their intensities
        matrix = Place_emitters([size, upsampling_factor], concentration, emit_power, decrease_power=decrease_power)
        
        #Convolving the matrix field by Gaussian/Airy PSF
        c = PSF_width * upsampling_factor
        if kernel_choice == 0:
            convolved = Convolve_Gauss(matrix, c)
        elif kernel_choice == 1:
            convolved = Convolve_Airy(matrix, c)
            
        #The convolved upscaled ground truth image is to be downsampled to the base size
        convolved_down = np.sum(convolved.reshape(size, upsampling_factor, size, upsampling_factor), axis=(1,3))
        
        #Add not_convolved camera noise
        conv_blurred = convolved_down + np.random.poisson(noise_inten, (size, size))
        
        #Subtracting the minimum of image intensity
        if sub_min:
            conv_blurred_sub = conv_blurred - conv_blurred.min()
        else:
            conv_blurred_sub = conv_blurred
        
        #Normalization and saving
        data_w_blur.append(Normalize(conv_blurred_sub) * (size**2)/(input_size_unit_norm**2))
        data_clear.append(Normalize(matrix))
        #Note: The model was trained for normalization of 50x50 image to a unit sum. Therefore, differently size image are normalized accordingly.
        
    return np.array(data_w_blur), np.array(data_clear)

##########################################

def Load_sequence():
    if not os.path.exists("Small_demo_stack"):
        with ZipFile("Small_demo_stack.zip", 'r') as zip_ref:
            zip_ref.extractall("Small_demo_stack")
    
    allframes = [f for f in os.listdir("Small_demo_stack/") if os.path.isfile(os.path.join("Small_demo_stack/", f))]
    allframes.sort()
    
    frames = np.zeros([len(allframes), 128, 128])
    for i in range(len(allframes)):
        frames[i] = tifffile.imread("Small_demo_stack/" + allframes[i]).astype(float)
    
    return frames

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



