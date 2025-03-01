from PIL import Image
import numpy as np

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
    return tf.reduce_sum(squared_difference, axis=(-1,-2,-3))

def Custom_mae_conv_func(y_true_, y_pred_):
    #Kernel for a normed Gaussian filter with PSF_width = 2
    kernel_array = Gauss_kernel(2)[:,:,None,None]
    kernel_tensor = tf.constant(kernel_array, dtype=tf.float32)

    #Valid padding, not to lose information on the borders
    y_true_conv = tf.nn.conv2d(y_true_, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    y_pred_conv = tf.nn.conv2d(y_pred_, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    
    absolute_difference = tf.math.abs(y_true_conv - y_pred_conv)
    return tf.reduce_sum(absolute_difference, axis=(-1,-2,-3))

##############################################################################################################################

def Reconstruct_data_with_model(data_, model_):
    data_subbed = data_ - data_.min(axis=(-1,-2))[:,None,None]
    norms = data_subbed.sum(axis=(-1,-2))
    data_in = data_subbed / norms[:,None,None]
    
    data_in_renormed = data_in * (data_.shape[-2]*data_.shape[-1]) / (50*50)
    
    predicted = np.squeeze(model_.predict(data_in_renormed[:,:,:,None]))
    return predicted, norms





