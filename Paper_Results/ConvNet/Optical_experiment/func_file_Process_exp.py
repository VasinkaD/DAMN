import numpy as np
import scipy
from keras.models import load_model

##############################################################################################################################
##############################################################################################################################

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

def Gradient(u):
    grad_x = np.zeros_like(u)
    grad_y = np.zeros_like(u)

    grad_x[:, :-1] = u[:, 1:] - u[:, :-1]   # x-gradient
    grad_y[:-1, :] = u[1:, :] - u[:-1, :]   # y-gradient

    return grad_x, grad_y

def Gradient_magnitude(grad_x, grad_y, eps=1e-8):
    return np.sqrt(grad_x**2 + grad_y**2 + eps)

def Divergence(px, py):
    div = np.zeros_like(px)

    div[:, :-1] += px[:, :-1]
    div[:, 1:]  -= px[:, :-1]

    div[:-1, :] += py[:-1, :]
    div[1:, :]  -= py[:-1, :]

    return div

def TV_term(image, eps=1e-8):
    grad_x, grad_y = Gradient(image)
    mag = Gradient_magnitude(grad_x, grad_y, eps)

    px = grad_x / mag
    py = grad_y / mag

    return Divergence(px, py)

###############################################################################################################

def RL_TV_iteration_for_concurrent(kernel_, input_sample_):
    reg_strength = 2e-3
    
    d = input_sample_ / np.sum(input_sample_)
    u_new = d
    u = np.zeros(u_new.shape)
    
    iteration = 0
    while (np.sum(np.abs(u - u_new)) / (d.shape[0]*d.shape[1])) > 10**(-10): #10**-10 for single pixel
        u = u_new
        
        #RL
        convolution = scipy.signal.convolve(u, kernel_, mode="same")
        division = np.divide(d, convolution, out=np.zeros_like(d), where=convolution!=0)
        
        #TV
        tv_term = TV_term(u)
        reg_term = np.maximum(1-reg_strength*tv_term, 1e-8)   #maximum to avoid divisition by small numbers
        
        #Update
        u_new = u * scipy.signal.convolve(division, kernel_, mode="same") / reg_term
        
        if iteration > 10**6:
            break
        else:
            iteration += 1
    
    return u_new * np.sum(input_sample_)

###############################################################################################################

def custom_mse_func(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_sum(squared_difference, axis=(-1,-2,-3))

def custom_mae_func(y_true, y_pred):
    squared_difference = tf.math.abs(y_true - y_pred)
    return tf.reduce_sum(squared_difference, axis=(-1,-2,-3))

def Load_ConvNet_model(path):
    DAMN_model = load_model(path, 
                            custom_objects={"custom_mse_func": custom_mse_func, "custom_mae_func": custom_mae_func})
    return DAMN_model

###############################################################################################################

def Evaluate_metric(target_, output_):
    absolute_error = np.abs(target_ - output_)
    summed_AE_per_image = np.sum(absolute_error, axis=(-2,-1))
    return summed_AE_per_image








