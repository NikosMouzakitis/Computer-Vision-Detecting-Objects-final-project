import numpy as np
import scipy
import General_function as tools

def Lapplacian_Sharpenning(Grayscale_Image,threshold_value):
    mask_manual=np.ones((3,3))
    mask_manual[1,1]=-8
    
    rows=len(Grayscale_Image)
    columns=len(Grayscale_Image[0])
    
    Image_Lapplaced=np.zeros((rows,columns))
    
    Image_Lapplaced=scipy.ndimage.convolve(Grayscale_Image,mask_manual)
    
    
    Sharpenned_image=Grayscale_Image-Image_Lapplaced
    Sharpenned_image=tools.float_to_uint8(Sharpenned_image)
    
    Sharpenned_image=np.where(Sharpenned_image==threshold_value,0,Sharpenned_image)
    return Sharpenned_image




#gia to main programma :
    
#Breite thn entasi tou background.Auti i entasi einai to threshold_vlaue
