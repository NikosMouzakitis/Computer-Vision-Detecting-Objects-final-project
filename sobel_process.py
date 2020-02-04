import numpy as np
import scipy
import General_function as tools

def sobel_process(Grayscale_Image):
    Gradient_Mask_x=[[-1,-2,-1],[0,0,0],[1,2,1]]
    Gradient_Mask_x= np.asarray(Gradient_Mask_x, dtype = np.int32 )
    
    Gradient_Mask_y=np.transpose(Gradient_Mask_x)
    
    Image_Gradient_x=scipy.ndimage.convolve(Grayscale_Image,Gradient_Mask_x)
    Image_Gradient_y=scipy.ndimage.convolve(Grayscale_Image,Gradient_Mask_y)
    
    Image_Gradient_x_y_abs=np.abs(Image_Gradient_x)+np.abs(Image_Gradient_y)
    Image_Gradient_x_y_abs=tools.float_to_uint8(Image_Gradient_x_y_abs)
    return Image_Gradient_x_y_abs




#gia to main programma :
    
#Breite thn entasi tou background.Auti i entasi einai to threshold_vlaue