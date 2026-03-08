import numpy as np
np.random.seed(18)
import scipy
import concurrent

##########################################
##########################################

#Parameter distributions

def Emit_num_dist(lower_bound_, upper_bound_, number_of_samples_):
    y = np.random.uniform(0, 1, number_of_samples_)
    x = upper_bound_ - (upper_bound_-lower_bound_)*np.sqrt(1-y)
    return x

def Emit_intensity_dist(lower_bound_, upper_bound_, number_of_samples_):
    y = np.random.uniform(0, 1, number_of_samples_)
    x = lower_bound_ + (upper_bound_-lower_bound_)*np.sqrt(y)
    return x

def Noise_intensity_dist(lower_bound_, upper_bound_, number_of_samples_):
    y = np.random.uniform(0, 1, number_of_samples_)
    x = upper_bound_ - (upper_bound_-lower_bound_)*np.sqrt(1-y)
    return x

def PSF_dist(lower_bound_, upper_bound_, number_of_samples_):
    y = np.random.uniform(0, 1, number_of_samples_)
    x = upper_bound_ - (upper_bound_-lower_bound_)*np.sqrt(1-y)
    return x

def Choose_kernel(number_of_samples_):
    coin_flip = np.random.randint(0, 2, number_of_samples_)       #Random 0 or 1, i.e., Gauss or Airy
    return coin_flip

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

#Asymmetric Point spread functions

def Axial_Gauss_function(x_, y_, sx_, sy_):
    Gauss_amplitude = np.exp((-x_**2) / (2*sx_**2) + (-y_**2) / (2*sy_**2))
    Gauss_intensity = Gauss_amplitude**2
    return Gauss_intensity

def Diagonal_Gauss_function(x_, y_, sd_, sa_):
    Gauss_amplitude = np.exp((-x_**2 / 2) * (1/(2*sd_**2) + 1/(2*sa_**2)) + (-y_**2 / 2) * (1/(2*sd_**2) + 1/(2*sa_**2)) - (2*x_*y_ / 2) * (1/(2*sd_**2) - 1/(2*sa_**2)))
    Gauss_intensity = Gauss_amplitude**2
    return Gauss_intensity

def Axial_Airy_function(x_, y_, sx_, sy_):
    q = 2 * np.sqrt((x_/sx_)**2 + (y_/sy_)**2)
    J_1 = scipy.special.jv(1, q)                   #scipy.special.jv(v, z) calls Bessel function of the first kind of "v" for a complex "z" position
    
    safe_division = np.divide(J_1, q, out=0.5+np.zeros_like(J_1), where=q!=0)

    Airy_amplitude = 2*safe_division               #Amplitude is the Besinc function
    Airy_intensity = Airy_amplitude**2             #Intensity is the Airy disc
    return Airy_intensity

def Diagonal_Airy_function(x_, y_, sd_, sa_):
    q = 2 * np.sqrt(x_**2 * (1/(2*sd_**2) + 1/(2*sa_**2)) + y_**2 * (1/(2*sd_**2) + 1/(2*sa_**2)) + 2*x_*y_ * (1/(2*sd_**2) - 1/(2*sa_**2)))
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

#Asymmetric Convolutions

def Convolve_Asym_Gauss(image_, c1_, c2_, ori_):        #ori_ = Orientation: Axial = 0, Diagonal = 1
    c = np.max([c1_, c2_])
    radius = int(np.ceil(3*c))
    k = int(2*radius+1)
    #---------------------------------------------------------------------------------------------------------------------------------------------
    x = np.ones((k,k)) * np.linspace(-radius, radius, k)                                                                          #Horizontal x axis increasing from left to right
    y = np.flip(np.transpose(x), axis=0)                                                                                          #Vertical y axis increasing from botoom to top
    #---------------------------------------------------------------------------------------------------------------------------------------------
    unnormed_psf_matrix = (1-ori_) * Axial_Gauss_function(x, y, c1_, c2_) + (ori_) * Diagonal_Gauss_function(x, y, c1_, c2_)      #Gaussian kernel; size based on 3sigma rule of 99.7%; ori_ and (1-ori_) serve as an IF function 
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()                                                           #Correctly normed Gaussian kernel
    #---------------------------------------------------------------------------------------------------------------------------------------------
    convolved = scipy.signal.convolve(image_, normed_psf_matrix, mode="same")                                                     #Convolution with the kernel
    
    return convolved

def Convolve_Asym_Airy(image_, c1_, c2_, ori_):        #ori_ = Orientation: Axial = 0, Diagonal = 1
    c = np.max([c1_, c2_])
    q_mins = np.array([3.8317,7.0156,10.1735,13.3237])                                                                            #Minima (in q space) of Airy intensity, i.e., dark rings of Airy disc -> these circles contain the power of 83.8%, 91.0%, 93.8%, x%
    n = 3                                                                                                                         #Chossing the n-th minimum to be the border of Airy kernel
    radius = int(np.ceil(c/2 * q_mins[n-1]))                                                                                      #Finding the radius (in r space) corresponding to including the n-th minima
    k = int(2*radius+1)                                                                                                           #Matrix size
    #---------------------------------------------------------------------------------------------------------------------------------------------
    x = np.ones((k,k)) * np.linspace(-radius, radius, k)                                                                          #Horizontal x axis increasing from left to right
    y = np.flip(np.transpose(x), axis=0)                                                                                          #Vertical y axis increasing from botoom to top
    #---------------------------------------------------------------------------------------------------------------------------------------------
    unnormed_psf_matrix = (1-ori_) * Axial_Airy_function(x, y, c1_, c2_) + (ori_) * Diagonal_Airy_function(x, y, c1_, c2_)        #Airy disc kernel; size based on 3sigma rule of 99.7%; ori_ and (1-ori_) serve as an IF function 
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()                                                           #Correctly normed Gaussian kernel
    #---------------------------------------------------------------------------------------------------------------------------------------------
    convolved = scipy.signal.convolve(image_, normed_psf_matrix, mode="same")                                                     #Convolution with the kernel
    
    return convolved

##########################################

#Functions to generate dataset

def Place_emitters(image_size_, em_number_, em_intensity_):
    size = image_size_[0] * image_size_[1]
    field = np.zeros((size, size))
    
    for i in range(em_number_):
            current_em_inten_average = Emit_intensity_dist(10**em_intensity_[0], 10**em_intensity_[1], 1)
            
            emitter_intensity = 0
            while emitter_intensity == 0:
                emitter_intensity = np.random.poisson(current_em_inten_average)
                decrease_factor = np.random.uniform(low = 0.1, high = 1)
                
            field[np.random.randint(0, size), np.random.randint(0, size)] += emitter_intensity * decrease_factor
    
    return field
    
def Normalize_input(tensor_):
    if np.sum(tensor_) != 0:
        normalized = tensor_ / np.sum(tensor_)
    else:
        normalized = tensor_
    
    return normalized

#Output has the same normalization as input, but keeping them separate
def Normalize_output(tensor_):
    if np.sum(tensor_) != 0:
        normalized = tensor_ / np.sum(tensor_)
    else:
        normalized = tensor_
    
    return normalized

def Generate_normed_dataset(num_data_, image_size_, PSF_width_, kernel_choice_, emit_inten_, camera_noise_inten_, data_number_, input_size_unit_norm_=50):
    data_w_blur = []
    data_clear = []
    
    size = image_size_[0]
    upscaling_factor = image_size_[1]
    upscaled_size = size * upscaling_factor
    
    #Iteration cycle generating one matrix field at the time
    for y in range(num_data_):
        current_PSF_width = PSF_width_[y]
        current_kernel_choice = kernel_choice_[y]
        current_camera_noise_intensity = camera_noise_inten_[y]
        current_number_of_emitters = int(data_number_[y])
        
        #Emitters are placed within the matrix field and associated with their intensities
        matrix = Place_emitters(image_size_, current_number_of_emitters, emit_inten_)
        
        #Randomly Asymmetric
        is_asym = np.random.randint(0, 2)          #Randomly choosing Symmetric or Assymetric (50:50)
        if is_asym == 0:
            ori = 0                                #Any value (0,1) suffices, orientation is redundant is symmetric PSF
            param_1 = 0                            #Neither axis is asymmetric
            param_2 = 0                            #Neither axis is asymmetric
        else:
            ori = np.random.randint(0, 2)          #Randomly choosing Axial vs Diagonal
            param_1 = np.random.randint(0, 2)      #Randomly choosing which axis is asymmetric
            param_2 = 1 - param_1                  #Complementary to param_1
        
        #Convolving the matrix field by Gaussian/Airy PSF with asymmetric width c and 0.95c
        c = np.copy(current_PSF_width) * upscaling_factor
        if current_kernel_choice == 0:
            convolved = Convolve_Asym_Gauss(matrix, c -0.05*c*param_1, c -0.05*c*param_2, ori)
        elif current_kernel_choice == 1:
            convolved = Convolve_Asym_Airy(matrix, c -0.05*c*param_1, c -0.05*c*param_2, ori)
            
        #The convolved upscaled ground truth image is to be transformed to the base size
        convolved_base = np.sum(convolved.reshape(size, upscaling_factor, size, upscaling_factor), axis=(1,3))
        
        #Add not_convolved camera noise
        conv_blurred = convolved_base + np.random.poisson(current_camera_noise_intensity, (size, size))
        
        #Subtracting the minimum of image intensity
        conv_blurred_sub = conv_blurred - conv_blurred.min()
        
        #Normalization and saving
        data_w_blur.append(Normalize_input(conv_blurred_sub) * (size**2)/(input_size_unit_norm_**2))
        data_clear.append(Normalize_output(matrix))
        
    return np.array(data_w_blur), np.array(data_clear)

##########################################

def Get_data(number_of_data_, params_, input_size_unit_norm_=50):
    size, upscale_factor, LB_concentration, UB_concentration, EI_lower_bound, EI_upper_bound, CNI_lower_bound, CNI_upper_bound, PSF_lower_bound, PSF_upper_bound = params_
    
    camera_noise_inten = 10**Noise_intensity_dist(CNI_lower_bound, CNI_upper_bound, number_of_data_)
    PSF_width = 10**PSF_dist(PSF_lower_bound, PSF_upper_bound, number_of_data_)
    kernel_choice = Choose_kernel(number_of_data_)
    data_number = Emit_num_dist(LB_concentration, UB_concentration, number_of_data_)
    
    data_blur, data_clear = Generate_normed_dataset(number_of_data_, [size, upscale_factor], PSF_width, kernel_choice, [EI_lower_bound, EI_upper_bound], camera_noise_inten, data_number, input_size_unit_norm_)
    
    return np.expand_dims(data_blur, axis=3), np.expand_dims(data_clear, axis=3)

##########################################

def Generate_one_sample(parameters):
    data_input, data_target = Get_data(1, parameters)
    return data_input, data_target

def Generate_epoch_data(epoch_size, parameters, workers=8):
    # Map tasks to processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(Generate_one_sample, [parameters] * epoch_size))
    
    # Concatenate
    X = np.concatenate([r[0] for r in results], axis=0).astype(np.float32)
    y = np.concatenate([r[1] for r in results], axis=0).astype(np.float32)
    
    return X, y

##########################################

