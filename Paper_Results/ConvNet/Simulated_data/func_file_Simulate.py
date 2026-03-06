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

def Convolve_Gauss(image_, sigma_):
    radius = int(np.ceil(3*sigma_))
    k = int(2*radius+1)
    #------------------------------------------------------------------------
    x = np.ones((k,k)) * np.linspace(-radius, radius, k)
    xx = np.sqrt(x**2 + np.transpose(x)**2)                                         #Field for kernel; size based on 3sigma rule
    #------------------------------------------------------------------------
    unnormed_psf_matrix = Gauss_function(xx, sigma_)                                #Gaussian kernel; size based on 3sigma rule
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()             #Correctly normed Gaussian kernel
    #------------------------------------------------------------------------
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

def Generate_normed_dataset(num_data_ = 100, base_size_ = 50, upsampling_factor_ = 1, emit_power_ = 5000, noise_inten_ = 10, PSF_width_ = 2, conc_ = 50, decrease_power = False, sub_min = False):
    data_w_blur = []
    data_clear = []
    image_size = [base_size_, upsampling_factor_]
    
    #Iteration cycle generating one matrix field at the time
    for y in range(num_data_):
        #Emitters are placed within the matrix field and associated with their intensities
        matrix = Place_emitters(image_size, conc_, emit_power_, decrease_power = decrease_power)
        
        #Convolving the matrix field by Gaussian PSF
        convolved = Convolve_Gauss(matrix, PSF_width_)
        
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

###############################################################################################################

def Gauss_for_transition(r, sigma):
    return np.exp(-r**2/sigma**2)/(np.pi*sigma**2)
    
def Besinc_for_transition(r, gamma, epsilon=1e-16):
    return (scipy.special.jv(1, 2*(r+epsilon)/gamma)/(2*(r+epsilon)/gamma))**2/(np.pi*gamma**2/4)

###############################################################################################################

def Get_rescaling_factor(width, gaussFraction):
    radiusMax = 1e2
    pointNumber = 1e6
    radiusList = np.linspace(0, radiusMax, int(pointNumber), endpoint=True)
    
    gaussList = Gauss_for_transition(radiusList, width*(1-gaussFraction)) / np.max(Gauss_for_transition(radiusList, width*gaussFraction))
    besincList = Besinc_for_transition(radiusList, width*gaussFraction) / np.max(Besinc_for_transition(radiusList, width*(1-gaussFraction)))
    
    psfList = gaussList * besincList
    psfList = psfList / np.max(psfList)
    
    psfHWHM = len(np.argwhere(psfList > np.max(psfList)/2))*radiusMax/pointNumber
    return width / psfHWHM

def PSF_function_for_transition_kernel(x_array, sigma, gaussFraction):
    width = sigma * 0.833 #renormalization
    rescaling_factor = Get_rescaling_factor(width, gaussFraction)
    x_array_rescaled = x_array / rescaling_factor
    
    gaussList = Gauss_for_transition(x_array_rescaled, width*(1-gaussFraction)) / np.max(Gauss_for_transition(x_array_rescaled, width*gaussFraction))
    besincList = Besinc_for_transition(x_array_rescaled, width*gaussFraction) / np.max(Besinc_for_transition(x_array_rescaled, width*(1-gaussFraction)))
    
    psfList = gaussList * besincList
    psfList = psfList / np.max(psfList)
    return psfList
    
###############################################################################################################

def Get_PSF_kernel(sigma_, gaussFraction_):
    radius = 10
    k = int(2*radius+1)
    #--------------------------------------------------------------------
    x_ = np.ones((k,k)) * np.linspace(-radius, radius, k)
    xx_ = np.sqrt(x_**2 + np.transpose(x_)**2)
    #--------------------------------------------------------------------
    unnormed_psf_matrix = PSF_function_for_transition_kernel(xx_, sigma_, gaussFraction_)
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()
    return normed_psf_matrix

def Convolve_PSF(image_, kernel_):
    convolved = scipy.signal.convolve(image_, kernel_, mode="same")
    return convolved

###############################################################################################################

def Generate_normed_dataset_for_transition(int_fac_, num_data_ = 100, base_size_ = 50, upsampling_factor_ = 1, emit_power_ = 5000, noise_inten_ = 10, PSF_width_ = 2, conc_ = 50, decrease_power = False):
    data_w_blur = []
    data_clear = []
    image_size = [base_size_, upsampling_factor_]
    
    conv_kernel = Get_PSF_kernel(PSF_width_, int_fac_)
    
    #Iteration cycle generating one matrix field at the time
    for y in range(num_data_):
        #Emitters are placed within the matrix field and associated with their intensities
        matrix = Place_emitters(image_size, conc_, emit_power_, decrease_power = decrease_power)
        
        #Convolving the matrix field by Gaussian PSF
        convolved = Convolve_PSF(matrix, conv_kernel)
        
        #Downsample the convolved image to the base_size
        convolved_50x50 = np.sum(convolved.reshape(base_size_, upsampling_factor_, base_size_, upsampling_factor_), axis=(1,3))
            
        #Adding Poisson noise
        conv_blurred = convolved_50x50 + np.random.poisson(noise_inten_, (base_size_, base_size_))
        
        #Subtracting the minimum of image intensity (part of normalization for inputs)
        conv_blurred_sub = conv_blurred - conv_blurred.min()
        
        #Normalize and save
        data_w_blur.append(Normalize(conv_blurred_sub) * (base_size_**2)/(50**2))
        data_clear.append(Normalize(matrix))
        
    return np.array(data_w_blur), np.array(data_clear)

###############################################################################################################















