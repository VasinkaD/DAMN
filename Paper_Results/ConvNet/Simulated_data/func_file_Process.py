import numpy as np
import scipy
from keras.models import load_model

##############################################################################################################################
##############################################################################################################################

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
    #while np.sum(np.abs(u - u_new)) > 10**(-5): 
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




