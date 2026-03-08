import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

##########################################
##########################################

class ResNet_block(layers.Layer):
    def __init__(self, channels, kernel_size, 
                 LeakyReLU_slope=0.1, padding="same", kernel_initializer="he_uniform"):
        super().__init__()
        
        self.conv1 = layers.Conv2D(channels, kernel_size, padding=padding, kernel_initializer=kernel_initializer)
        self.bn1   = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(channels, kernel_size, padding=padding, kernel_initializer=kernel_initializer)
        self.bn2   = layers.BatchNormalization()
        
        self.add   = layers.Add()
        self.activ = layers.LeakyReLU(LeakyReLU_slope)
        
    def call(self, x, training=False):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.activ(out)
        
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        
        shortcut = x
        out = self.add([out, shortcut])
        return self.activ(out)

##########################################

class ResNet_segment(layers.Layer):
    def __init__(self, channels, num_blocks, kernel_size, 
                 LeakyReLU_slope=0.1, padding="same", kernel_initializer="he_uniform"):
        super().__init__()
        
        self.blocks = [
            ResNet_block(channels, kernel_size, 
                         padding=padding, LeakyReLU_slope=LeakyReLU_slope, kernel_initializer=kernel_initializer) 
            for _ in range(num_blocks)
        ]
        
    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        return x

##########################################

class Upsample_block(layers.Layer):
    def __init__(self, channels, kernel_size, 
                 LeakyReLU_slope=0.1, upsample_size=(2,2), 
                 interpolation="bilinear", padding="same", kernel_initializer="he_uniform"):
        super().__init__()
        
        self.upsample = layers.UpSampling2D(size=upsample_size, interpolation=interpolation)
        self.conv     = layers.Conv2D(channels, kernel_size, padding=padding, kernel_initializer=kernel_initializer)
        self.activ    = layers.LeakyReLU(LeakyReLU_slope)
        
    def call(self, x):
        out = self.upsample(x)
        out = self.conv(out)
        return self.activ(out)

##########################################

class ResNet_model_paper(tf.keras.Model):
    def __init__(self, channels, num_blocks_array,
                 kernel_sizes=[5,7,9,11], LeakyReLU_slope=0.1,
                 padding="same", interpolation="bilinear", kernel_initializer="he_uniform"
                 , **kwargs):
        super().__init__(**kwargs)
        
        self.conv_in = layers.Conv2D(channels, kernel_sizes[0], padding=padding, kernel_initializer=kernel_initializer)
        self.activ   = layers.LeakyReLU(LeakyReLU_slope)
        
        self.segment1  = ResNet_segment(channels, num_blocks_array[0], kernel_sizes[0], 
                                        padding=padding, LeakyReLU_slope=LeakyReLU_slope, kernel_initializer=kernel_initializer)
        self.upsample1 = Upsample_block(channels, kernel_sizes[0], 
                                        interpolation=interpolation, padding=padding, LeakyReLU_slope=LeakyReLU_slope, kernel_initializer=kernel_initializer)
        
        self.segment2  = ResNet_segment(channels, num_blocks_array[1], kernel_sizes[1], 
                                        padding=padding, LeakyReLU_slope=LeakyReLU_slope, kernel_initializer=kernel_initializer)
        self.upsample2 = Upsample_block(channels, kernel_sizes[1], 
                                        interpolation=interpolation, padding=padding, LeakyReLU_slope=LeakyReLU_slope, kernel_initializer=kernel_initializer)
        
        self.segment3  = ResNet_segment(channels, num_blocks_array[2], kernel_sizes[2], 
                                        padding=padding, LeakyReLU_slope=LeakyReLU_slope, kernel_initializer=kernel_initializer)
        self.upsample3 = Upsample_block(channels, kernel_sizes[2], 
                                        interpolation=interpolation, padding=padding, LeakyReLU_slope=LeakyReLU_slope, kernel_initializer=kernel_initializer)
        
        self.segment4  = ResNet_segment(channels, num_blocks_array[3], kernel_sizes[3], 
                                        padding=padding, LeakyReLU_slope=LeakyReLU_slope, kernel_initializer=kernel_initializer)
        
        self.conv_out  = layers.Conv2D(1, kernel_sizes[-1], padding=padding, kernel_initializer=kernel_initializer)
        self.activ_out = layers.Softmax(axis=(-2,-3))
    
    def call(self, x, training=False):
        x = self.conv_in(x)
        x = self.activ(x)
        
        x = self.segment1(x, training=training)
        x = self.upsample1(x)
        
        x = self.segment2(x, training=training)
        x = self.upsample2(x)
        
        x = self.segment3(x, training=training)
        x = self.upsample3(x)
        
        x = self.segment4(x, training=training)
        
        x = self.conv_out(x)
        return self.activ_out(x)

##########################################

def Gauss_function(x_, sigma_):
    Gauss_amplitude = np.exp(-(x_)**2/(2*sigma_**2))     #Amplitude is the Gaussian profile
    Gauss_intensity = Gauss_amplitude**2                 #Intesity is also Gaussian but of different width
    return Gauss_intensity

def Gauss_kernel(c_):
    radius = int(np.ceil(3*c_))
    k = int(2*radius+1)
    #---------------------------------------------------------------------------------------------------------------------------------------------
    x_ = np.ones((k,k)) * np.linspace(-radius, radius, k)
    xx_ = np.sqrt(x_**2 + np.transpose(x_)**2)                                      #Field for kernel
    unnormed_psf_matrix = Gauss_function(xx_, c_)                                   #Gaussian kernel; size based on 3sigma rule of 99.7%
    normed_psf_matrix = unnormed_psf_matrix / unnormed_psf_matrix.sum()             #Correctly normed Gaussian kernel
    
    return normed_psf_matrix

def Get_kernel_for_Loss():
    #Kernel for a normed Gaussian filter with PSF_width = 2
    kernel_array = Gauss_kernel(2)[:,:,None,None]
    return kernel_array

def Custom_CCE_conv_func_regularized(y_true_, y_pred_, reg_strength_):
    #Kernel for a normed Gaussian filter with PSF_width = 2
    kernel_array = Get_kernel_for_Loss()
    kernel_tensor = tf.constant(kernel_array, dtype=tf.float32)

    #Convolve
    y_true_conv = tf.nn.conv2d(y_true_, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    y_pred_conv = tf.nn.conv2d(y_pred_, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    
    #CCE
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=None)
    crossentropy = cce(tf.keras.layers.Flatten()(y_true_conv), tf.keras.layers.Flatten()(y_pred_conv))
    
    #Entropy regularization
    entropy = -tf.reduce_sum(y_pred_ * tf.math.log(y_pred_ + 1e-10), axis=(-1,-2,-3))
    
    return crossentropy + entropy * reg_strength_

def Custom_MSE_conv_func_regularized(y_true_, y_pred_, reg_strength_):
    #Kernel for a normed Gaussian filter with PSF_width = 2
    kernel_array = Get_kernel_for_Loss()
    kernel_tensor = tf.constant(kernel_array, dtype=tf.float32)

    #Convolve
    y_true_conv = tf.nn.conv2d(y_true_, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    y_pred_conv = tf.nn.conv2d(y_pred_, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    
    #MSE
    squared_difference = tf.square(y_true_conv - y_pred_conv)
    
    #Entropy regularization
    entropy = -tf.reduce_sum(y_pred_ * tf.math.log(y_pred_ + 1e-10), axis=(-1,-2,-3))
    
    return squared_difference + entropy * reg_strength_

def Custom_mae_conv_func(y_true_, y_pred_):
    #Kernel for a normed Gaussian filter with PSF_width = 2, not computation to save time
    kernel_array = Get_kernel_for_Loss()
    kernel_tensor = tf.constant(kernel_array, dtype=tf.float32)

    #Convolve
    y_true_conv = tf.nn.conv2d(y_true_, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    y_pred_conv = tf.nn.conv2d(y_pred_, kernel_tensor, strides=[1, 1, 1, 1], padding='VALID')
    
    #MAE
    absolute_difference = tf.math.abs(y_true_conv - y_pred_conv)
    return tf.reduce_sum(absolute_difference, axis=(-1,-2,-3))

##########################################

#Logger function
class EpochLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file, monitor, optimizer, mode="min"):
        super().__init__()
        self.log_file = log_file
        self.monitor = monitor
        self.mode = mode
        self.optimizer = optimizer
        self.best_value = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val = logs.get(self.monitor)
    
        if current_val is None:
            best_reached = False
        else:
            if self.best_value is None or \
               (self.mode == "min" and current_val < self.best_value) or \
               (self.mode == "max" and current_val > self.best_value):
                self.best_value = current_val
                best_reached = True
            else:
                best_reached = False
    
        # Current learning rate
        lr = float(tf.keras.backend.get_value(self.optimizer.learning_rate))
    
        # Build log line
        line = f"Epoch {epoch+1} finished.\n"
        if self.best_value is not None:
            line += (f"New best validation metric reached: {best_reached} "
                     f"(value: {self.best_value:.6f}).\n")
        else:
            line += "Validation metric not available yet.\n"
        if best_reached:
            line += (f">>> Model checkpoint saved.\n")
        line += f"Current LR: {lr:.6f}\n\n"
    
        with open(self.log_file, "a") as f:
            f.write(line)