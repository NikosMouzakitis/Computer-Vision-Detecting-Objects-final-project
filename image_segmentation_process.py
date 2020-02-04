import numpy as np
import General_function as tools
#megalitero_apo,mikrotero_apo=timi gia threshold,timi[0,1],opou 1 kanei diplo threshold, 0 kanei mono threshold
def image_segmentation(Grayscale_Image,megalitero_apo,mikrotero_apo,timi):
    if(timi==1):
        list_cords=([d for d in Grayscale_Image >megalitero_apo ])
        
        
        x_y_cords2=[]
        cords_x=[]  
        cords_y=[]
            
        for i in range(len(list_cords)):
            x_y_cords2.append([k for k, e in enumerate(list_cords[i]) if e == True ])
        
        
        for i in range(len(x_y_cords2)):
            temp=len(x_y_cords2[i])
            if(temp>0):
                for k in range(temp):
                    cords_x.append(i)
                    cords_y.append(x_y_cords2[i][k])
        
        cords_x= np.array(cords_x)
        cords_y= np.array(cords_y)
            
        #---------------------------------------------------------------------------------------------------------------------
            
        object1=np.ones((len(Grayscale_Image),len(Grayscale_Image[0])))
        object1[cords_x,cords_y]=0
        object1=tools.type_conversion_to_int32(object1)
        
        list_cords=([d for d in Grayscale_Image <mikrotero_apo ])
        
        
        x_y_cords2=[]
        cords_x=[]  
        cords_y=[]
            
        for i in range(len(list_cords)):
            x_y_cords2.append([k for k, e in enumerate(list_cords[i]) if e == True ])
        
        
        for i in range(len(x_y_cords2)):
            temp=len(x_y_cords2[i])
            if(temp>0):
                for k in range(temp):
                    cords_x.append(i)
                    cords_y.append(x_y_cords2[i][k])
        
        cords_x= np.array(cords_x)
        cords_y= np.array(cords_y)
        
        object2=np.ones((len(Grayscale_Image),len(Grayscale_Image[0])))
        object2[cords_x,cords_y]=0
        object2=tools.type_conversion_to_int32(object2)  
        
        object3=object1-object2
        return object3
    
    if(timi==0):
        
        list_cords=([d for d in Grayscale_Image <mikrotero_apo ])
        
        
        x_y_cords2=[]
        cords_x=[]  
        cords_y=[]
            
        for i in range(len(list_cords)):
            x_y_cords2.append([k for k, e in enumerate(list_cords[i]) if e == True ])
        
        
        for i in range(len(x_y_cords2)):
            temp=len(x_y_cords2[i])
            if(temp>0):
                for k in range(temp):
                    cords_x.append(i)
                    cords_y.append(x_y_cords2[i][k])
        
        cords_x= np.array(cords_x)
        cords_y= np.array(cords_y)
        
        object2=np.ones((len(Grayscale_Image),len(Grayscale_Image[0])))
        object2[cords_x,cords_y]=0
        object2=tools.type_conversion_to_int32(object2)  
        
        object3=object2
        return object3
