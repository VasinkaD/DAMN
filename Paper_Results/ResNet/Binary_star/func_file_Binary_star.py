import numpy as np

##########################################

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

#Get the star separation based on the ground truth information in the table
def Get_star_separation():
    #Get information from the ground truth table
    GT_table = np.genfromtxt('GT_table.csv', delimiter=',')
    
    #Arcsec from the table
    table_row = 21
    table_column = 9
    table_sep_arcsec = GT_table[table_row, table_column]
    
    #Pixels calculated from the table star positions
    (star_x1_pos, star_y1_pos), (star_x2_pos, star_y2_pos) = Localization_table()
    table_sep_pixels = np.sqrt((star_x1_pos - star_x2_pos)**2 + (star_y1_pos - star_y2_pos)**2)
    
    return table_sep_arcsec, table_sep_pixels

##########################################

#Get the star positions based on the ground truth information in the table
def Localization_table():
    #Get information from the ground truth table
    GT_table = np.genfromtxt('GT_table.csv', delimiter=',')
    
    #Specify the star positions from the table
    table_row = 21
    star_x1_column = 12
    star_y1_column = 13
    star_x2_column = 15
    star_y2_column = 16
    
    #Extract the coordinates
    star_x1_pos = GT_table[table_row, star_x1_column]
    star_y1_pos = GT_table[table_row, star_y1_column]
    star_x2_pos = GT_table[table_row, star_x2_column]
    star_y2_pos = GT_table[table_row, star_y2_column]
    
    return (star_x1_pos, star_y1_pos), (star_x2_pos, star_y2_pos)

##########################################

#Get the star positions for the cut-image and its reconstruction based on the ground truth information in the table
def Localization_cut():
    #Extract from table
    (star_x1_pos, star_y1_pos), (star_x2_pos, star_y2_pos) = Localization_table()
    
    cut_vt = cut_hl = 100
    
    #Coordinates in 50x50 region
    star_x1_pos_cut = star_x1_pos - cut_hl
    star_y1_pos_cut = star_y1_pos - cut_vt
    star_x2_pos_cut = star_x2_pos - cut_hl
    star_y2_pos_cut = star_y2_pos - cut_vt
    
    #Coordinates in reconstructed upsampled region
    star_x1_pos_cut_up = 8*star_x1_pos_cut + 3.5
    star_y1_pos_cut_up = 8*star_y1_pos_cut + 3.5
    star_x2_pos_cut_up = 8*star_x2_pos_cut + 3.5
    star_y2_pos_cut_up = 8*star_y2_pos_cut + 3.5
    #Note: 8 times finer grid has different origin; correct the offset (8-1)/2
    
    return (star_x1_pos_cut_up, star_y1_pos_cut_up), (star_x2_pos_cut_up, star_y2_pos_cut_up), (star_x1_pos_cut, star_y1_pos_cut), (star_x2_pos_cut, star_y2_pos_cut)

##########################################

#Get the star positions for the zoomed reconstruction based on the ground truth information in the table
def Localization_zoom(zoom):
    (star_x1_pos, star_y1_pos), (star_x2_pos, star_y2_pos), _, _ = Localization_cut()
    zoom_vt, zoom_hl = zoom
    
    #Coordinates in zoomed region with correction
    star_x1_pos_zoom = star_x1_pos - zoom_hl + 3.5
    star_y1_pos_zoom = star_y1_pos - zoom_vt + 3.5
    star_x2_pos_zoom = star_x2_pos - zoom_hl + 3.5
    star_y2_pos_zoom = star_y2_pos - zoom_vt + 3.5
    
    return (star_x1_pos_zoom, star_y1_pos_zoom), (star_x2_pos_zoom, star_y2_pos_zoom)
    


