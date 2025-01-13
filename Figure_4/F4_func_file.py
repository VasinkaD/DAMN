import numpy as np
import scipy

##############################################################################################################################
##############################################################################################################################

def Place_emitters(size_, concentration_, em_power_):
    #Get the matrix field
    field = np.zeros((size_, size_))
    
    #One by one, add emitters to the field
    for i in range(concentration_):
        #Assign current emitter with Emitter power
        emitter_power = 0
        while emitter_power == 0:
            emitter_power = np.random.poisson(em_power_)
        
        #Assign current emitter with its position
        field[np.random.randint(size_), np.random.randint(size_)] += emitter_power
    
    return field

###############################################################################################################

def Airy_function(r_, sigma_):
    q = 2 * r_ / sigma_
    J_1 = scipy.special.jv(1, q)                   #scipy.special.jv(v, z) calls Bessel function of the first kind of "v" for a complex "z" position
    
    safe_division = np.divide(J_1, q, out=0.5+np.zeros_like(J_1), where=q!=0)

    Airy_amplitude = 2*safe_division               #Amplitude is the Besinc function
    Airy_intensity = Airy_amplitude**2             #Intensity is the Airy disc
    return Airy_intensity

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

def Generate_normed_dataset_for_experiment(num_data_ = 100, image_size_ = 50, emit_power_ = 23000, noise_inten_ = 10, PSF_width_ = 2.05, conc_ = 50):
    data_w_blur = []
    data_clear = []
    
    #Iteration cycle generating one matrix field at the time
    for y in range(num_data_):
        #Emitters are placed within the matrix field and associated with their intensities
        matrix = Place_emitters(image_size_, conc_, emit_power_)
        
        #Convolving the matrix field by Gaussian PSF
        convolved = Convolve_Airy(matrix, PSF_width_)
            
        #Adding Poisson noise
        conv_blurred = convolved + np.random.poisson(noise_inten_, (image_size_, image_size_))
        
        #Normalize and save
        data_w_blur.append(Normalize(conv_blurred))
        data_clear.append(Normalize(matrix))
        
    return np.array(data_w_blur), np.array(data_clear)

###############################################################################################################

def custom_mse_func(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_sum(squared_difference, axis=(-1,-2,-3))

def custom_mae_func(y_true, y_pred):
    squared_difference = tf.math.abs(y_true - y_pred)
    return tf.reduce_sum(squared_difference, axis=(-1,-2,-3))

###############################################################################################################

def Evaluate_metric(target_, output_):
    absolute_error = np.abs(target_ - output_)
    summed_AE_per_image = np.sum(absolute_error, axis=(-2,-1))
    return summed_AE_per_image

###############################################################################################################

def Gauss_function(x_, sigma_):
    return np.exp(-(x_)**2/(sigma_**2))

def Gauss_kernel(sigma_):
    radius = int(np.ceil(3*sigma_))
    k = int(2*radius+1)
    #------------------------------------------------------------------------
    x = np.ones((k,k)) * np.linspace(-radius, radius, k)
    xx = np.sqrt(x**2 + np.transpose(x)**2)                                         #Field for kernel; size based on 3sigma rule
    #------------------------------------------------------------------------
    unnormed_psf_matrix = Gauss_function(xx, sigma_)                                #Gaussian kernel; size based on 3sigma rule
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()             #Correctly normed Gaussian kernel
    
    return normed_psf_matrix

def RL_iteration_for_concurrent(kernel_, input_sample_):    
    d = input_sample_ / np.sum(input_sample_)
    u_new = d
    u = np.zeros(u_new.shape)
    
    iteration = 0
    while (np.sum(np.abs(u - u_new)) / (d.shape[0]*d.shape[1])) > 10**(-10): #10**-10 for single pixel        
        u = u_new
        
        convolution = scipy.signal.convolve(u, kernel_, mode="same")
        division = np.divide(d, convolution, out=np.zeros_like(d), where=convolution!=0)
        
        u_new = u * scipy.signal.convolve(division, kernel_, mode="same")
        if iteration > 10**6:
            break
        else:
            iteration += 1
    
    return u_new * np.sum(input_sample_)









