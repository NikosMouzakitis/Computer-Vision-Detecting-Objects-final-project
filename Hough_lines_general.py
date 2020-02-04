import numpy as np
from skimage import feature
from skimage.transform import probabilistic_hough_line

def Hough_Transform(Grayscale_Image):
    Grayscale_Image= np.asarray(Grayscale_Image, dtype = np.float64 )
    #------------------------------------------------------------------------------------------------------------
    # Compute the Canny filter for two values of sigma
    Image_edge_sigma_original_image = feature.canny(Grayscale_Image, sigma=2.5,high_threshold =9)
    Image_edge_sigma_original_image= np.asarray(Image_edge_sigma_original_image, dtype = np.int32 )
    
    #-----------------------------------------------------------------------------------------------------------
    lines = probabilistic_hough_line(Image_edge_sigma_original_image)
    
    lines=list(lines)
    return lines




#gia to main programma :
    
#for line in lines:
#    p0, p1 = line
#    plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
#    
#plt.show()

