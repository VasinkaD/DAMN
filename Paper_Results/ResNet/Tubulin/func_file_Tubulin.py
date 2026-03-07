import os
from zipfile import ZipFile
import tifffile
import numpy as np
import cv2

##############################################################################################################################
##############################################################################################################################

def Load_sequence():
    if not os.path.exists("sequence"):
        with ZipFile("sequence.zip", 'r') as zip_ref:
            zip_ref.extractall()
    
    allframes = [f for f in os.listdir("sequence/") if os.path.isfile(os.path.join("sequence/", f))]
    allframes.sort()
    
    frames = np.zeros([len(allframes), 128, 128])
    for i in range(len(allframes)):
        frames[i] = tifffile.imread("sequence/" + allframes[i]).astype(float)
    
    return frames

##############################################################################################################################

#SOS Plugin: Render the emitter as Gaussian function at localized positions (loc. uncertainty gives its width)
def Render_SOS_Plugin(x, y, intensity, stdev_x, stdev_y):
    #Upsampling factor:
    orig_size = 128
    upsample = 10
    hr_size = orig_size * upsample
    
    #How large to draw each Gaussian kernel (in reconstruction pixels)
    gaussian_support = 3     #Extends +-3 sigma around the center

    #Allocate variable
    sos_reconstructed = np.zeros([hr_size, hr_size])
    
    for xi, yi, inten, sx, sy in zip(x, y, intensity, stdev_x, stdev_y):
        #Convert original pixel coordinates to high-res coordinates
        x_hr = xi * upsample
        y_hr = yi * upsample
    
        #Convert localization uncertainty from original pixels to high-res pixels
        sx_hr = sx * upsample
        sy_hr = sy * upsample
    
        #Define bounding box for Gaussian kernel
        x_min = int(max(0, np.floor(x_hr - gaussian_support * sx_hr)))
        x_max = int(min(hr_size - 1, np.ceil (x_hr + gaussian_support * sx_hr)))
        y_min = int(max(0, np.floor(y_hr - gaussian_support * sy_hr)))
        y_max = int(min(hr_size - 1, np.ceil (y_hr + gaussian_support * sy_hr)))
    
        #Make coordinate grid
        y_grid, x_grid = np.meshgrid(
            np.arange(y_min, y_max),
            np.arange(x_min, x_max),
            indexing="xy"
        )
    
        #2D Gaussian
        gauss = inten * np.exp(-0.5 * (
            ((x_grid - x_hr) / sx_hr)**2 +
            ((y_grid - y_hr) / sy_hr)**2
        ))
    
        #Add to reconstruction
        sos_reconstructed[x_min:x_max, y_min:y_max] += gauss
    
    return sos_reconstructed

##############################################################################################################################

#Subtract minimum and norm to unit sum
def Normalize_data(data):
    #Subtract minima
    data_subbed = data - np.expand_dims(data.min(axis=(-1,-2)), axis=(-1,-2))
    
    #Normalize to unit sum
    norms = data_subbed.sum(axis=(-1,-2))
    data_in = data_subbed / np.expand_dims(norms, axis=(-1,-2))
    
    #Normalize images of other sizes than the default 50x50 
    data_in_renormed = data_in * (data.shape[-2]*data.shape[-1]) / (50*50)
    
    return data_in_renormed, norms

##############################################################################################################################

def Get_correction(corners_, target_size_, remap_offset_):
    target_corners = remap_offset_ + np.array([(0, 0), (0, target_size_-1), (target_size_-1, 0), (target_size_-1, target_size_-1)])
    
    #cv2 has switched height and width axis notation
    corners_[:, [1, 0]] = corners_[:, [0, 1]]
    target_corners[:, [1, 0]] = target_corners[:, [0, 1]]
    
    homography, _ = cv2.findHomography(corners_, target_corners, params=None)
    
    return homography

def Apply_correction(image, parameters):
    target_size, out_size, remap_offset, drift, cut = parameters
    
    corners = np.array([[0,0], [0,target_size-1], [target_size-1,0+drift], [target_size-1,target_size-1+drift]])
    homography = Get_correction(corners, target_size, remap_offset)
    
    corrected_image = cv2.warpPerspective(image, homography, (out_size, out_size))
    
    return corrected_image[cut:-cut,cut:-cut]

##############################################################################################################################

def FWHM_interpolate(profile, px_to_nm, x=None):
    if x is None:
        x = np.arange(len(profile))

    half_max = 0.5
    sign = np.sign(profile - half_max)
    crossings = np.where(np.diff(sign))[0]

    if len(crossings) < 2:
        return 0  # no valid FWHM

    # interpolate left crossing
    i1 = crossings[0]
    x1 = x[i1] + (half_max - profile[i1]) * (x[i1+1] - x[i1]) / (profile[i1+1] - profile[i1])

    # interpolate right crossing
    i2 = crossings[-1]
    x2 = x[i2] + (half_max - profile[i2]) * (x[i2+1] - x[i2]) / (profile[i2+1] - profile[i2])

    return (x2 - x1) * px_to_nm

