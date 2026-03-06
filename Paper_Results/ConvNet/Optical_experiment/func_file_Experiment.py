import numpy as np
import scipy

##############################################################################################################################
##############################################################################################################################

def Place_emitters(image_size_, concentration_, em_power_, decrease_power = False):
    #Get the matrix field
    size = image_size_[0] * image_size_[1]
    field = np.zeros((size, size))
    
    #One by one, add emitters to the field
    for i in range(concentration_):
        #Assign current emitter with Emitter power
        emitter_power = 0
        while emitter_power == 0:
            emitter_power = np.random.poisson(em_power_)
        if decrease_power:
            decrease_factor = np.random.uniform(low = 0.1, high = 1)
            emitter_power = emitter_power * decrease_factor
        
        #Assign current emitter with its position
        field[np.random.randint(0, size), np.random.randint(0, size)] += emitter_power
    
    return field

###############################################################################################################

def Airy_function(r_, sigma_):
    q = 2 * r_ / sigma_
    J_1 = scipy.special.jv(1, q)                   #scipy.special.jv(v, z) calls Bessel function of the first kind of "v" for a complex "z" position
    
    safe_division = np.divide(J_1, q, out=0.5+np.zeros_like(J_1), where=q!=0)

    Airy_amplitude = 2*safe_division               #Amplitude is the Besinc function
    Airy_intensity = Airy_amplitude**2             #Intensity is the Airy disc
    return Airy_intensity

def Airy_kernel(sigma_):
    q_mins = np.array([3.8317,7.0156,10.1735,13.3237])       #Minima (in q space) of Airy intensity, i.e., dark rings of Airy disc -> these circles contain the power of 83.8%, 91.0%, 93.8%, x%
    n = 3                                                    #Chossing the n-th minimum to be the border of Airy kernel
    radius = int(np.ceil(sigma_/2 * q_mins[n-1]))                #Finding the radius (in r space) corresponding to including the n-th minima
    k = int(2*radius+1)                                      #Matrix
    #---------------------------------------------------------------------------------------------------------------------------------------------
    x_ = np.ones((k,k)) * np.linspace(-radius, radius, k)
    xx_ = np.sqrt(x_**2 + np.transpose(x_)**2)                                      #Field for kernel
    unnormed_psf_matrix = Airy_function(xx_, sigma_)                                    #Airy disc kernel
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()             #Correctly normed Airy disc kernel
    
    return normed_psf_matrix

def Convolve_Airy(image_, sigma_):
    q_mins = np.array([3.8317,7.0156,10.1735,13.3237])       #Minima (in q space) of Airy intensity, i.e., dark rings of Airy disc -> these circles contain the power of 83.8%, 91.0%, 93.8%, x%
    n = 3                                                    #Chossing the n-th minimum to be the border of Airy kernel
    radius = int(np.ceil(sigma_/2 * q_mins[n-1]))                #Finding the radius (in r space) corresponding to including the n-th minima
    k = int(2*radius+1)                                      #Matrix
    #---------------------------------------------------------------------------------------------------------------------------------------------
    x_ = np.ones((k,k)) * np.linspace(-radius, radius, k)
    xx_ = np.sqrt(x_**2 + np.transpose(x_)**2)                                      #Field for kernel
    unnormed_psf_matrix = Airy_function(xx_, sigma_)                                    #Airy disc kernel
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()             #Correctly normed Airy disc kernel
    #---------------------------------------------------------------------------------------------------------------------------------------------
    convolved = scipy.signal.convolve(image_, normed_psf_matrix, mode="same")       #Convolution with the kernel
    
    return convolved

###############################################################################################################

def Normalize(matrix_):
    #Normalize matrix to a unit sum
    if np.sum(matrix_) != 0:
        normalized = matrix_ / np.sum(matrix_)
    else:
        normalized = matrix_
    
    return normalized

###############################################################################################################

def Generate_normed_dataset_for_experiment(num_data_ = 100, base_size_ = 50, upsampling_factor_ = 1, emit_power_ = 5000, noise_inten_ = 10, PSF_width_ = 2, conc_ = 50, decrease_power = False, sub_min = False):
    data_w_blur = []
    data_clear = []
    image_size = [base_size_, upsampling_factor_]
    
    #Iteration cycle generating one matrix field at the time
    for y in range(num_data_):
        #Emitters are placed within the matrix field and associated with their intensities
        matrix = Place_emitters(image_size, conc_, emit_power_, decrease_power = decrease_power)
        
        #Convolving the matrix field by Gaussian PSF
        convolved = Convolve_Airy(matrix, PSF_width_)
        
        #Downsample the convolved image to the base_size
        convolved_50x50 = np.sum(convolved.reshape(base_size_, upsampling_factor_, base_size_, upsampling_factor_), axis=(1,3))
        
        #Adding Poisson noise
        conv_blurred = convolved_50x50 + np.random.poisson(noise_inten_, (base_size_, base_size_))
        
        #Subtracting the minimum of image intensity (part of normalization for inputs)
        if sub_min:
            conv_blurred_sub = conv_blurred - conv_blurred.min()
        else:
            conv_blurred_sub = conv_blurred
        
        #Normalize and save
        data_w_blur.append(Normalize(conv_blurred_sub) * (base_size_**2)/(50**2))
        data_clear.append(Normalize(matrix))
        
    return np.array(data_w_blur), np.array(data_clear)















