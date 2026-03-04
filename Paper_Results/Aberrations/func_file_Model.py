import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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

