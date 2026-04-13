import numpy as np
import tifffile

##########################################

#Load the tiff image of the double star
def Load_image(image_path = "Double_star_LR.tiff"):
    image = tifffile.imread(image_path).astype(float)[100:150,100:150]
    return image

#Subtract minimum and norm to unit sum
def Normalize_data(data):
    #Subtract minima
    data_subbed = data - np.expand_dims(data.min(axis=(-1,-2)), axis=(-1,-2))
    
    #Normalize to unit sum
    norms = data_subbed.sum(axis=(-1,-2))
    data_in = data_subbed / np.expand_dims(norms, axis=(-1,-2))
    
    #Normalize images of other sizes than the default 50x50 
    data_in_renormed = data_in * (data.shape[-2]*data.shape[-1]) / (50*50)
    
    return data_in_renormed

##########################################

#Get the Gaia-based information from the table
def Get_table_information():
    #Get information from the ground truth table
    GT_table = np.genfromtxt('GT_table.csv', delimiter=',')
    
    #Extract the coordinates [px]
    star_x1_pos = GT_table[1, 5]
    star_y1_pos = GT_table[1, 6]    
    star_x2_pos = GT_table[2, 5]
    star_y2_pos = GT_table[2, 6]
    
    #Extract the separation [arcsec] and [px]
    table_sep_arcsec = GT_table[1, 8]
    table_sep_pixels = np.sqrt((star_x1_pos - star_x2_pos)**2 + (star_y1_pos - star_y2_pos)**2)
    separations = [table_sep_arcsec, table_sep_pixels]
    
    #Calculate the [arcsec] to [px]
    arcsec_to_px = table_sep_pixels / table_sep_arcsec
    
    #Extract the uncertainties [px]
    star_x1_unc = (GT_table[1, 2] / 1000) * arcsec_to_px
    star_y1_unc = (GT_table[1, 4] / 1000) * arcsec_to_px
    star_x2_unc = (GT_table[2, 2] / 1000) * arcsec_to_px
    star_y2_unc = (GT_table[2, 4] / 1000) * arcsec_to_px
    uncertainties = [star_x1_unc, star_y1_unc, star_x2_unc, star_y2_unc]
    
    #Extract the intensities
    star_1_int = GT_table[1, 7]
    star_2_int = GT_table[2, 7]
    intensities = [star_1_int, star_2_int]
    
    #Correct the shift
    star_x1_pos_sh, star_y1_pos_sh = Correct_shift(star_x1_pos, star_y1_pos, arcsec_to_px)
    star_x2_pos_sh, star_y2_pos_sh = Correct_shift(star_x2_pos, star_y2_pos, arcsec_to_px)
    
    #Cut to center image region
    star_x1_pos_cut, star_y1_pos_cut = Cut_coordinates(star_x1_pos_sh, star_y1_pos_sh)
    star_x2_pos_cut, star_y2_pos_cut = Cut_coordinates(star_x2_pos_sh, star_y2_pos_sh)
    positions = [star_x1_pos_cut, star_y1_pos_cut, star_x2_pos_cut, star_y2_pos_cut]
    
    return positions, uncertainties, separations, intensities, arcsec_to_px

##########################################

#Correct the subpixel mismatch in low-resolution image coordinates to the reference star localizations
def Correct_shift(x, y, arcsec_to_px, ra_shift = 0.061, dec_shift = 0.003):
    #Extracted low-resolution image from Pan-STARSS1 coordinates differs at sub-pixel level from the Gaia reference positions
    #We estimated the shift to be 0.061 arcsec in ra and 0.003 arcsec in dec
    
    x_shifted = x + ra_shift * arcsec_to_px
    y_shifted = y + dec_shift * arcsec_to_px
    
    return x_shifted, y_shifted

#Adjust the star positions for the center region cut of the image
def Cut_coordinates(x, y, x_cut = 100, y_cut = 100):
    #The full image is 256x256 px but we only use the center region
    
    x_cut = x - x_cut
    y_cut = y - y_cut
    
    return x_cut, y_cut

#Turn original image localizations to the upsampled image localizations
def Upsample_coordinates(x, y, upsampling = 8):
    #Note: 8 times finer grid has a different (0,0) image origin; correct the offset (8-1)/2
    offset = (upsampling-1)/2
    
    x_up = upsampling * x + offset
    y_up = upsampling * y + offset
    
    return x_up, y_up

#Adjust the star positions for the recontruction zoom-in
def Zoom_coordinates(x, y, zoom_x, zoom_y):
    #The visualization zooms to the area of the double star - adjusting the localization coordinates accordingly
    
    x_zoomed = x - zoom_x
    y_zoomed = y - zoom_y
    
    return x_zoomed, y_zoomed




