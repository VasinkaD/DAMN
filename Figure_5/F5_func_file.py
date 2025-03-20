from PIL import Image
import numpy as np
import scipy

##############################################################################################################################
##############################################################################################################################

def Load_astrodata(path_):
    rgb_image = Image.open(path_)
    grayscale = rgb_image.convert('L')
    return np.array(grayscale)

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

def Custom_mse_conv_func(y_true_, y_pred_):
    #Kernel for a normed Gaussian filter with PSF_width = 2
    kernel_array = Gauss_kernel(2)[:,:,None,None]
    kernel_tensor = tf.constant(kernel_array, dtype=tf.float32)

    #Valid padding, not to lose information on the borders
    y_true_conv = tf.nn.conv2d(y_true_, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    y_pred_conv = tf.nn.conv2d(y_pred_, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    
    squared_difference = tf.square(y_true_conv - y_pred_conv)
    loss = tf.reduce_sum(squared_difference, axis=(-1,-2,-3))
    entropy = -tf.reduce_sum(y_pred_ * tf.math.log(y_pred_ + 1e-10), axis=(-1,-2,-3))
    return loss + entropy * 5e-5

def Custom_mae_conv_func(y_true_, y_pred_):
    #Kernel for a normed Gaussian filter with PSF_width = 2
    kernel_array = Gauss_kernel(2)[:,:,None,None]
    kernel_tensor = tf.constant(kernel_array, dtype=tf.float32)

    #Valid padding, not to lose information on the borders
    y_true_conv = tf.nn.conv2d(y_true_, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    y_pred_conv = tf.nn.conv2d(y_pred_, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    
    absolute_difference = tf.math.abs(y_true_conv - y_pred_conv)
    loss = tf.reduce_sum(absolute_difference, axis=(-1,-2,-3))
    return loss

##############################################################################################################################

def Reconstruct_data_with_model(data_, model_):
    data_subbed = data_ - data_.min(axis=(-1,-2))[:,None,None]
    norms = data_subbed.sum(axis=(-1,-2))
    data_in = data_subbed / norms[:,None,None]
    
    data_in_renormed = data_in * (data_.shape[-2]*data_.shape[-1]) / (50*50)
    
    predicted = np.squeeze(model_.predict(data_in_renormed[:,:,:,None]))
    return predicted, norms

##############################################################################################################################

def Gauss_duo_function(x_y_, x0_A, y0_A, I0_A, sigma_A, x0_B, y0_B, I0_B, sigma_B, offset_):
    #Two Gaussian functions
    x, y = x_y_
    Gauss_A = I0_A * np.exp(-((x - x0_A)**2 + (y - y0_A)**2)/(sigma_A**2))
    Gauss_B = I0_B * np.exp(-((x - x0_B)**2 + (y - y0_B)**2)/(sigma_B**2))
    return Gauss_A + Gauss_B + offset_

def Fit_Gauss_duo(data_):
    #Grids for fit function
    X_Y_grids = np.indices((data_.shape[0],data_.shape[1]))
    X, Y = X_Y_grids
    xy_for_fit = np.vstack((X.ravel(), Y.ravel()))
    z_for_fit = data_.ravel()
    
    #Estimated starting parameter values for fit
    x0_est_A, y0_est_A = np.unravel_index(np.argmax(data_), data_.shape)
    x0_est_B, y0_est_B = np.array([8, 5])
    I0_est_A = data_.max()
    I0_est_B = data_[x0_est_B, y0_est_B]
    radius_est_A = 0.2
    radius_est_B = 0.4
    offset_est = data_.min()
    
    #Fit Gaussian duo to our data
    p_optimal, p_covariance = scipy.optimize.curve_fit(Gauss_duo_function, xy_for_fit, z_for_fit, p0 = (x0_est_A, y0_est_A, I0_est_A, radius_est_A, x0_est_B, y0_est_B, I0_est_B, radius_est_B, offset_est), 
                                                       bounds=([0, 0, 0, 0.0001, 0, 0, 0, 0.0001, 0], 
                                                               [data_.shape[0], data_.shape[1], np.inf, data_.shape[0], data_.shape[0], data_.shape[1], np.inf, data_.shape[0], 1]))
    return p_optimal



