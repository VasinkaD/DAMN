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

def Generate_normed_dataset(num_data_ = 100, image_size_ = 50, emit_power_ = 5000, noise_inten_ = 10, PSF_width_ = 2, conc_ = 50):
    data_w_blur = []
    data_clear = []
    
    #Iteration cycle generating one matrix field at the time
    for y in range(num_data_):
        #Emitters are placed within the matrix field and associated with their intensities
        matrix = Place_emitters(image_size_, conc_, emit_power_)
        
        #Convolving the matrix field by Gaussian PSF
        convolved = Convolve_Gauss(matrix, PSF_width_)
            
        #Adding Poisson noise
        conv_blurred = convolved + np.random.poisson(noise_inten_, (image_size_, image_size_))
        
        #Normalize and save
        data_w_blur.append(Normalize(conv_blurred))
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

def Generate_normed_dataset_for_transition(int_fac_, num_data_ = 100, image_size_ = 50, emit_power_ = 5000, noise_inten_ = 10, PSF_width_ = 2, conc_ = 50):
    data_w_blur = []
    data_clear = []
    
    conv_kernel = Get_PSF_kernel(PSF_width_, int_fac_)
    
    #Iteration cycle generating one matrix field at the time
    for y in range(num_data_):
        #Emitters are placed within the matrix field and associated with their intensities
        matrix = Place_emitters(image_size_, conc_, emit_power_)
        
        #Convolving the matrix field by Gaussian PSF
        convolved = Convolve_PSF(matrix, conv_kernel)
            
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

###############################################################################################################

def Get_rescaling_factor(width, gaussFraction):
    def gauss(r, sigma):
        return np.exp(-r**2/sigma**2)/(np.pi*sigma**2)
        
    def besinc(r, gamma, epsilon=1e-16):
        return (scipy.special.jv(1, 2*(r+epsilon)/gamma)/(2*(r+epsilon)/gamma))**2/(np.pi*gamma**2/4)
        
    radiusMax = 1e2
    pointNumber = 1e6
    radiusList = np.linspace(0, radiusMax, int(pointNumber), endpoint=True)
    
    gaussList = gauss(radiusList, width*(1-gaussFraction)) / np.max(gauss(radiusList, width*gaussFraction))
    besincList = besinc(radiusList, width*gaussFraction) / np.max(besinc(radiusList, width*(1-gaussFraction)))
    
    psfList = gaussList * besincList
    psfList = psfList / np.max(psfList)
    
    psfHWHM = len(np.argwhere(psfList > np.max(psfList)/2))*radiusMax/pointNumber
    return width / psfHWHM

def AtoG_function_for_kernel(x_array, sigma, gaussFraction):
    def gauss(r, sigma):
        return np.exp(-r**2/sigma**2)/(np.pi*sigma**2)
        
    def besinc(r, gamma, epsilon=1e-16):
        return (scipy.special.jv(1, 2*(r+epsilon)/gamma)/(2*(r+epsilon)/gamma))**2/(np.pi*gamma**2/4)
    
    width = sigma * 0.833 #renormalization
    rescaling_factor = Get_rescaling_factor(width, gaussFraction)
    x_array_rescaled = x_array / rescaling_factor
    
    gaussList = gauss(x_array_rescaled, width*(1-gaussFraction)) / np.max(gauss(x_array_rescaled, width*gaussFraction))
    besincList = besinc(x_array_rescaled, width*gaussFraction) / np.max(besinc(x_array_rescaled, width*(1-gaussFraction)))
    
    psfList = gaussList * besincList
    psfList = psfList / np.max(psfList)
    return psfList

def Get_AtoG_kernel(sigma_, gaussFraction_):
    radius = 10
    k = int(2*radius+1)
    #--------------------------------------------------------------------
    x_ = np.ones((k,k)) * np.linspace(-radius, radius, k)
    xx_ = np.sqrt(x_**2 + np.transpose(x_)**2)
    #--------------------------------------------------------------------
    unnormed_psf_matrix = AtoG_function_for_kernel(xx_, sigma_, gaussFraction_)
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()
    return normed_psf_matrix

###############################################################################################################

def Evaluate_metric(target_, output_):
    absolute_error = np.abs(target_ - output_)
    summed_AE_per_image = np.sum(absolute_error, axis=(-2,-1))
    return summed_AE_per_image

###############################################################################################################

def Forward(SNR_):
    Guass_kernel_peak = 0.0795779                 #Value of normed Gaussian kernel with width=2 at the center
    PNR = SNR_ * Guass_kernel_peak
    return PNR

def Inverse(PNR_):
    Guass_kernel_peak = 0.0795779                 #Value of normed Gaussian kernel with width=2 at the center
    SNR = PNR_ / Guass_kernel_peak
    return SNR

###############################################################################################################













