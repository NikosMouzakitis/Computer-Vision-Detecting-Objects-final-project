''' Mouzakitis Nikolaos TP4460 
    Computer Vision Project.'''
import numpy as np
import General_function as tools
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import scipy.ndimage as nimg
import skimage as image_tool
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.transform
from skimage import measure
import skimage.morphology
import Hough_lines_general as hl
import sobel_process as sp
import Lapplace_Sharp as lap

# function that gets coordinates where a binary image has '1's.Returns 2 lists for this purpose.
def getmecoords(x):
    x_list = []
    y_list = []
    
    for i in range(0,1110):
        for j in range(0,1390):
            if(x[i][j] == 1):
                x_list.append(i)
                y_list.append(j)
    return x_list, y_list


PIXEL_CUT = 1035
Grayscale_Image = mpimg.imread('Art.png')
Grayscale_Image= np.asarray(Grayscale_Image, dtype = np.float64)
Grayscale_Image=(Grayscale_Image-np.min(Grayscale_Image))*255/(np.max(Grayscale_Image)-np.min(Grayscale_Image))
clr_img = clr_img2 = mpimg.imread('Art_rgb_light.png')
ttvg = testttvg = clr_img[:,:,0]
ttvg= np.asarray(ttvg, dtype = np.float64)
ttvg=(ttvg-np.min(ttvg))*255/(np.max(ttvg)-np.min(ttvg))
#agalma : >199
agalma = Grayscale_Image.copy()
agalma = np.where( Grayscale_Image < 240, 1 ,0)
agalma = np.asarray(agalma, dtype=np.uint8)
Grayscale_Image[PIXEL_CUT:,:] = 0
plt.imshow(Grayscale_Image, cmap='gray')
plt.title('initial image cropped on the bottom')
plt.show()
agalmacutted = Grayscale_Image.copy()
agalmacutted = np.where( Grayscale_Image < 240, 0 ,1)
agalmacutted = np.asarray(agalmacutted, dtype=np.uint8)
#cup
# commented section since we got better accuracy after doing twice segmentation, substracting and manually changin values to get the full object.
#test = Grayscale_Image.copy()
#test= np.where(Grayscale_Image < 192, 0, 1)
#test = np.asarray(test, dtype = np.bool)
#test=test.astype(int)
#Overall_connected_list=image_tool.measure.regionprops(test) #Find the conected pixels
#Coords_of_connected_components=Overall_connected_list[0].coords #Save coords in a variable
#[Labeled_image,num_of_neighbors] = measure.label(test,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
#Group_connected_list=image_tool.measure.regionprops(Labeled_image) #Find the conected pixels
#Labeled_image = np.where(Labeled_image== 3, 1, 0)
#cup = Labeled_image - agalmacutted
#cup = np.where(cup == 0, 0,1)

test2c = Grayscale_Image.copy()
test2c= np.where(Grayscale_Image < 183, 0, 1)
test2c = np.asarray(test2c, dtype = np.bool)
test2c=test2c.astype(int)
Overall_connected_list=image_tool.measure.regionprops(test2c) #Find the conected pixels
Coords_of_connected_components=Overall_connected_list[0].coords #Save coords in a variable
[Labeled_image,num_of_neighbors] = measure.label(test2c,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
Group_connected_list=image_tool.measure.regionprops(Labeled_image) #Find the conected pixels

argument1 = Labeled_image  
argument1 = argument1 + 100
test2c2 = Grayscale_Image.copy()
test2c2= np.where(Grayscale_Image < 195, 0, 1)
test2c2 = np.asarray(test2c2, dtype = np.bool)
test2c2=test2c2.astype(int)
Overall_connected_list=image_tool.measure.regionprops(test2c2) #Find the conected pixels
Coords_of_connected_components=Overall_connected_list[0].coords #Save coords in a variable
[Labeled_image,num_of_neighbors] = measure.label(test2c2,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
Group_connected_list=image_tool.measure.regionprops(Labeled_image) #Find the conected pixels

argument2 = Labeled_image 

arr_to_process = argument1 - argument2 - agalmacutted
for y in range(688, 1057):
    for x in range(449,848):
        if(arr_to_process[x][y] == 101):
            arr_to_process[x][y] = 98 # we intend to do that to get the perfect segmentation of the shape.
            
arr_to_process = np.where(arr_to_process== 98, 1, 0)

cup2 = arr_to_process
cup = cup2

#front_cup
t2 = Grayscale_Image.copy()
t2= np.where(t2 < 200, 0, 1)
t2 = np.asarray(t2, dtype = np.bool)
t2=t2.astype(int)
Overall_connected_list1=image_tool.measure.regionprops(t2) #Find the conected pixels
Coords_of_connected_components=Overall_connected_list1[0].coords #Save coords in a variable
[Labeled_image,num_of_neighbors] = measure.label(t2,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
Group_connected_list=image_tool.measure.regionprops(Labeled_image) #Find the conected pixels
Labeled_image = np.where(Labeled_image== 2, 1, 0)
front_cup = Labeled_image.copy()
#head
T1 = 118
T2 = 174
s1 = Grayscale_Image.copy()
s1= np.where(Grayscale_Image < T1, 0, 1)
s2 = Grayscale_Image.copy()
s2= np.where(Grayscale_Image < T2, 0, 1)
s3 = s1-s2
Overall_connected_list1=image_tool.measure.regionprops(s3) #Find the conected pixels
Coords_of_connected_components=Overall_connected_list1[0].coords #Save coords in a variable
[Labeled_image,num_of_neighbors] = measure.label(s3,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
Group_connected_list=image_tool.measure.regionprops(Labeled_image) #Find the conected pixels
head1 = np.where( (Labeled_image== 3), 1, 0)
head1_absolute = head1.copy() # this is to be used with the cone detection since, the real(not 2 at the worst case moved pxl) needed to find the cone coordinates.
#fixing a little bit of the holes here.
for i in range(0,1110):
    flag = 0
    countj = 0
    for j in range(0,1390):
        if(head1[i][j] == 1):
            flag+=1
        if(flag and head1[i][j] == 0):
            head1[i][j] = 1
            countj+=1
        ## essential or we will completely destroy the image !
        if(countj > 2):
            break            
head2 = np.where( (Labeled_image== 14), 1, 0)
head3 = np.where( (Labeled_image== 34), 1, 0)
head4 = np.where( (Labeled_image== 37), 1, 0)
head = head1+head2+head3+head4  #get full head at some point!
head_absolute = head1_absolute+head2+head3+head4 
head_absolute = np.asarray(head_absolute, dtype = np.bool)
head_absolute = head_absolute.astype(int)
head = np.asarray(head, dtype = np.bool)
head = head.astype(int)
#cone
T1 = 150
T2 = 190
s1 = Grayscale_Image.copy()
s1= np.where(Grayscale_Image < T1, 0, 1)
s2 = Grayscale_Image.copy()
s2= np.where(Grayscale_Image < T2, 0, 1)
s3 = s1-s2
Overall_connected_list1=image_tool.measure.regionprops(s3) #Find the conected pixels
Coords_of_connected_components=Overall_connected_list1[0].coords #Save coords in a variable
[Labeled_image,num_of_neighbors] = measure.label(s3,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
Group_connected_list=image_tool.measure.regionprops(Labeled_image) #Find the conected pixels
cone_with_head = np.where( (Labeled_image== 3), 1, 0)
cone_with_head = np.asarray(cone_with_head, dtype = np.bool)
cone_with_head = cone_with_head.astype(int)
pre_cone = cone_with_head - head_absolute
Overall_connected_list1=image_tool.measure.regionprops(pre_cone) #Find the conected pixels
Coords_of_connected_components=Overall_connected_list1[0].coords #Save coords in a variable
[Labeled_image,num_of_neighbors] = measure.label(pre_cone,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
Group_connected_list=image_tool.measure.regionprops(Labeled_image)
cone = np.where(Labeled_image == 29, 1, 0)
#crayon
T1 = 180
T2 = 202
s1 = Grayscale_Image.copy()
s1= np.where(Grayscale_Image < T1, 0, 1)
s2 = Grayscale_Image.copy()
s2= np.where(Grayscale_Image < T2, 0, 1)
s3 = s1-s2
Overall_connected_list1=image_tool.measure.regionprops(s3) #Find the conected pixels
Coords_of_connected_components=Overall_connected_list1[0].coords #Save coords in a variable
[Labeled_image,num_of_neighbors] = measure.label(s3,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
Group_connected_list=image_tool.measure.regionprops(Labeled_image) #Find the conected pixels
cray1 = np.where( (Labeled_image== 21), 1, 0)
cray2 = np.where( (Labeled_image== 16), 1, 0)
cray_part1 = cray1 + cray2
cray_part1 = np.asarray(cray_part1, dtype = np.bool)
cray_part1 = cray_part1.astype(int)
cray_part1 -= front_cup
cray_part1 -= cone
cray_part1 = np.where(cray_part1 ==1, 1, 0)
T1 = 200
T2 = 230
s1 = Grayscale_Image.copy()
s1= np.where(Grayscale_Image < T1, 0, 1)
s2 = Grayscale_Image.copy()
s2= np.where(Grayscale_Image < T2, 0, 1)
s3 = s1-s2
s3 = np.asarray(s3, dtype=np.bool)
s3 = s3.astype(int)
Overall_connected_list1=image_tool.measure.regionprops(s3) #Find the conected pixels
Coords_of_connected_components=Overall_connected_list[0].coords #Save coords in a variable
[Labeled_image,num_of_neighbors] = measure.label(s3,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
Group_connected_list=image_tool.measure.regionprops(Labeled_image)
cray_part2 = np.where(Labeled_image == 4 , 1, 0)
crayon = cray_part1+cray_part2
crayon = np.where(crayon==2, 1, crayon)
#circ1
T1 = 185
T2 = 206
s1 = Grayscale_Image.copy()
s1= np.where(Grayscale_Image < T1, 0, 1)
s2 = Grayscale_Image.copy()
s2= np.where(Grayscale_Image < T2, 0, 1)
s3 = s1-s2
Overall_connected_list1=image_tool.measure.regionprops(s3) #Find the conected pixels
Coords_of_connected_components=Overall_connected_list1[0].coords #Save coords in a variable
[Labeled_image,num_of_neighbors] = measure.label(s3,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
Group_connected_list=image_tool.measure.regionprops(Labeled_image)
circ1 = np.where(Labeled_image == 1, 1, 0)
#circ2
T1 = 158
T2 = 179
s1 = Grayscale_Image.copy()
s1= np.where(Grayscale_Image < T1, 0, 1)
s2 = Grayscale_Image.copy()
s2= np.where(Grayscale_Image < T2, 0, 1)
s3 = s1-s2
Overall_connected_list1=image_tool.measure.regionprops(s3) #Find the conected pixels
Coords_of_connected_components=Overall_connected_list1[0].coords #Save coords in a variable
[Labeled_image,num_of_neighbors] = measure.label(s3,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
Group_connected_list=image_tool.measure.regionprops(Labeled_image)
tmp1 = np.where(Labeled_image==2, 1, 0)
tmp2 = np.where(Labeled_image==3, 1, 0)
tmp3 = np.where(Labeled_image==8, 1, 0)
tmp4 = np.where(Labeled_image==11, 1, 0)
tmp5 = np.where(Labeled_image==6, 1, 0)
circ2 = tmp1+tmp2+tmp3+tmp4+tmp5
#circle3
T1 = 133
T2 = 155
s1 = Grayscale_Image.copy()
s1= np.where(Grayscale_Image < T1, 0, 1)
s2 = Grayscale_Image.copy()
s2= np.where(Grayscale_Image < T2, 0, 1)
s3 = s1-s2
s3[665:,:] = 0
Overall_connected_list1=image_tool.measure.regionprops(s3) #Find the conected pixels
Coords_of_connected_components=Overall_connected_list1[0].coords #Save coords in a variable
[Labeled_image,num_of_neighbors] = measure.label(s3,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
Group_connected_list=image_tool.measure.regionprops(Labeled_image)
tmp1 = np.where(Labeled_image==2, 1, 0)
tmp2 = np.where(Labeled_image==5, 1, 0)
tmp3 = np.where(Labeled_image==6, 1, 0)
circ3 = tmp1+tmp2+tmp3
#circle4
T1 = 115
T2 = 146
s1 = Grayscale_Image.copy()
s1= np.where(Grayscale_Image < T1, 0, 1)
s2 = Grayscale_Image.copy()
s2= np.where(Grayscale_Image < T2, 0, 1)
s3 = s1-s2
#adjusting
s3[564:,:] = 0
s3 = s3 - circ3
Overall_connected_list1=image_tool.measure.regionprops(s3) #Find the conected pixels
Coords_of_connected_components=Overall_connected_list1[0].coords #Save coords in a variable
[Labeled_image,num_of_neighbors] = measure.label(s3,  neighbors=8,background=False,return_num=True) #Get labeeled image on 8 neighbors
Group_connected_list=image_tool.measure.regionprops(Labeled_image)
tmp1 = np.where(Labeled_image==1, 1, 0)
tmp2 = np.where(Labeled_image==3, 1, 0)
tmp3 = np.where(Labeled_image==5, 1, 0)
tmp4 = np.where(Labeled_image==6, 1, 0)
tmp5 = np.where(Labeled_image==9, 1, 0)
tmp6 = np.where(Labeled_image==16, 1, 0)
tmp7 = np.where(Labeled_image==18, 1, 0)
tmp8 = np.where(Labeled_image==22, 1, 0)
circ4 = tmp1 +tmp2 +tmp3 +tmp4 +tmp5 +tmp6 +tmp7 +tmp8 


#### Laplacian for each figure and then combined into a single image.
clr_img2 = mpimg.imread('Art_rgb_light.png')
clr_img2 = clr_img2[:,:,1]  
clr_img2=(clr_img2-np.min(clr_img2))*255/(np.max(clr_img2)-np.min(clr_img2))   
clr_img2 = clr_img2.astype(int) 

# got the laplace on the whole image. Now the strategy is to cut off all 
# the pixels that ain't belong into the extracted structures we got.
test_lap_whole_img = lap.Lapplacian_Sharpenning(clr_img2, 30)

x_agalma, y_agalma = getmecoords(agalmacutted)
x_cup, y_cup = getmecoords(cup)
x_crayon, y_crayon = getmecoords(crayon)
x_front_cup, y_front_cup = getmecoords(front_cup)
x_head, y_head = getmecoords(head)
x_cone, y_cone = getmecoords(cone)
x_circ1, y_circ1 = getmecoords(circ1)
x_circ2, y_circ2 = getmecoords(circ2)
x_circ3, y_circ3 = getmecoords(circ3)
x_circ4, y_circ4 = getmecoords(circ4)

laplace_img = np.zeros([1110,1390])
laplace_img = laplace_img.astype(int) 

for i in range(0, len(x_agalma)):
    laplace_img[x_agalma[i]][y_agalma[i]] = test_lap_whole_img[x_agalma[i]][y_agalma[i]]
for i in range(0, len(x_cup)):
    laplace_img[x_cup[i]][y_cup[i]] = test_lap_whole_img[x_cup[i]][y_cup[i]]

for i in range(0, len(x_head)):
    laplace_img[x_head[i]][y_head[i]] = test_lap_whole_img[x_head[i]][y_head[i]]
for i in range(0, len(x_crayon)):
    laplace_img[x_crayon[i]][y_crayon[i]] = test_lap_whole_img[x_crayon[i]][y_crayon[i]]

for i in range(0, len(x_front_cup)):
    laplace_img[x_front_cup[i]][y_front_cup[i]] = test_lap_whole_img[x_front_cup[i]][y_front_cup[i]]
for i in range(0, len(x_cone)):
    laplace_img[x_cone[i]][y_cone[i]] = test_lap_whole_img[x_cone[i]][y_cone[i]]

for i in range(0, len(x_circ1)):
    laplace_img[x_circ1[i]][y_circ1[i]] = test_lap_whole_img[x_circ1[i]][y_circ1[i]]
for i in range(0, len(x_circ2)):
    laplace_img[x_circ2[i]][y_circ2[i]] = test_lap_whole_img[x_circ2[i]][y_circ2[i]]

for i in range(0, len(x_circ3)):
    laplace_img[x_circ3[i]][y_circ3[i]] = test_lap_whole_img[x_circ3[i]][y_circ3[i]]
for i in range(0, len(x_circ4)):
    laplace_img[x_circ4[i]][y_circ4[i]] = test_lap_whole_img[x_circ4[i]][y_circ4[i]]

plt.figure('Laplace only on the pixels of the figures.')
plt.imshow(laplace_img, cmap='gray')


## SUBPLOT containing each one of the detected structures

figure2 = plt.figure('Extracted Structures') 
#---------------little statue
subplot1=figure2.add_subplot(2,5,1)
agalma = np.where(agalma == 0, 1, 0)
plt.imshow(agalmacutted,cmap="binary")
subplot1.set_title('little statue')
#---------------cup
subplot1=figure2.add_subplot(2,5,2)
plt.imshow(cup,cmap="binary")
subplot1.set_title('Cup')
#---------------front cup
subplot1=figure2.add_subplot(2,5,3)
plt.imshow(front_cup,cmap="binary")
subplot1.set_title('Front Cup')
#---------------head
subplot1=figure2.add_subplot(2,5,4)
plt.imshow(head,cmap="binary")
subplot1.set_title('Head')
#---------------cone
subplot1=figure2.add_subplot(2,5,5)
plt.imshow(cone,cmap="binary")
subplot1.set_title('Cone')
#---------------crayon
subplot1=figure2.add_subplot(2,5,6)
plt.imshow(crayon,cmap="binary")
subplot1.set_title('Crayon')
#---------------circle1
subplot1=figure2.add_subplot(2,5,7)
plt.imshow(circ1,cmap="binary")
subplot1.set_title('Circle1')
#---------------circle2
subplot1=figure2.add_subplot(2,5,8)
plt.imshow(circ2,cmap="binary")
subplot1.set_title('Circle2')
#---------------circle3
subplot1=figure2.add_subplot(2,5,9)
plt.imshow(circ3,cmap="binary")
subplot1.set_title('Circle3')
#---------------circle4
subplot1=figure2.add_subplot(2,5,10)
plt.imshow(circ4,cmap="binary")
subplot1.set_title('Circle4')

perigramma_head = tools.gitniasi_8_perigramma(head)
perigramma_agalma = tools.gitniasi_8_perigramma(agalmacutted)
perigramma_cup = tools.gitniasi_8_perigramma(cup)
perigramma_crayon = tools.gitniasi_8_perigramma(crayon)
perigramma_cone = tools.gitniasi_8_perigramma(cone)
perigramma_front_cup = tools.gitniasi_8_perigramma(front_cup)
perigramma_circ1 = tools.gitniasi_8_perigramma(circ1)
perigramma_circ2 = tools.gitniasi_8_perigramma(circ2)
perigramma_circ3 = tools.gitniasi_8_perigramma(circ3)
perigramma_circ4 = tools.gitniasi_8_perigramma(circ4)
print('Perigramata done')

plt.figure('Perigrammata in Colored Image')

clr_img = mpimg.imread('Art_rgb_light.png')

plt.imshow(clr_img, cmap='binary')

for i in range(0, len(perigramma_head)):
    plt.scatter(perigramma_head[i][1], perigramma_head[i][0], c="#bcbd22", s=1, linewidths=0.01)
print('D1')
for i in range(0, len(perigramma_agalma)):
    plt.scatter(perigramma_agalma[i][1], perigramma_agalma[i][0], c= "#ff7f0e", s=1, linewidths=0.01)
print('D2')    
for i in range(0, len(perigramma_cup)):
    plt.scatter(perigramma_cup[i][1], perigramma_cup[i][0], c='b',  s=1,linewidths=0.01)
print('D3')    
for i in range(0, len(perigramma_front_cup)):
    plt.scatter(perigramma_front_cup[i][1], perigramma_front_cup[i][0], c='c', s=1, linewidths=0.01)
print('D4')    
for i in range(0, len(perigramma_circ1)):
    plt.scatter(perigramma_circ1[i][1], perigramma_circ1[i][0], c='m', s=1, linewidths=0.01)
print('D5')    
for i in range(0, len(perigramma_circ2)):
    plt.scatter(perigramma_circ2[i][1], perigramma_circ2[i][0], c='y', s=1, linewidths=0.01)
print('D6')    
for i in range(0, len(perigramma_circ3)):
    plt.scatter(perigramma_circ3[i][1], perigramma_circ3[i][0], c='k', s=1, linewidths=0.01)
print('D7')    
for i in range(0, len(perigramma_circ4)):
    plt.scatter(perigramma_circ4[i][1], perigramma_circ4[i][0], c='w', s=1, linewidths=0.01)
print('D8')  
for i in range(0, len(perigramma_crayon)):
    plt.scatter(perigramma_crayon[i][1], perigramma_crayon[i][0], c='g', s=1, linewidths=0.01)
print('D9')    
for i in range(0, len(perigramma_cone)):
    plt.scatter(perigramma_cone[i][1], perigramma_cone[i][0], c='r', s=1, linewidths=0.01)
print('D10')        
      
#sobel in the grayscale image.
plt.figure('After Sobel process')
sob = sp.sobel_process(ttvg)


sobel_img = np.zeros([1110,1390])
sobel_img = sobel_img.astype(int) 

for i in range(0, len(x_agalma)):
    sobel_img[x_agalma[i]][y_agalma[i]] = sob[x_agalma[i]][y_agalma[i]]
for i in range(0, len(x_cup)):
    sobel_img[x_cup[i]][y_cup[i]] = sob[x_cup[i]][y_cup[i]]

for i in range(0, len(x_head)):
    sobel_img[x_head[i]][y_head[i]] = sob[x_head[i]][y_head[i]]
for i in range(0, len(x_crayon)):
    sobel_img[x_crayon[i]][y_crayon[i]] = sob[x_crayon[i]][y_crayon[i]]

for i in range(0, len(x_front_cup)):
    sobel_img[x_front_cup[i]][y_front_cup[i]] = sob[x_front_cup[i]][y_front_cup[i]]
for i in range(0, len(x_cone)):
    sobel_img[x_cone[i]][y_cone[i]] = sob[x_cone[i]][y_cone[i]]

for i in range(0, len(x_circ1)):
    sobel_img[x_circ1[i]][y_circ1[i]] = sob[x_circ1[i]][y_circ1[i]]
for i in range(0, len(x_circ2)):
    sobel_img[x_circ2[i]][y_circ2[i]] = sob[x_circ2[i]][y_circ2[i]]

for i in range(0, len(x_circ3)):
    sobel_img[x_circ3[i]][y_circ3[i]] = sob[x_circ3[i]][y_circ3[i]]
for i in range(0, len(x_circ4)):
    sobel_img[x_circ4[i]][y_circ4[i]] = sob[x_circ4[i]][y_circ4[i]]
    
plt.imshow(sobel_img, cmap='gray')
plt.title('Sobel')

hough_input = np.zeros([1110,1390])
hough_input = hough_input.astype(int) 
#get values of the GrayScale image, to the hough input array, where the detected objects exist.
for i in range(0, len(x_agalma)):
    hough_input[x_agalma[i]][y_agalma[i]] = Grayscale_Image[x_agalma[i]][y_agalma[i]]
for i in range(0, len(x_cup)):
    hough_input[x_cup[i]][y_cup[i]] = Grayscale_Image[x_cup[i]][y_cup[i]]

for i in range(0, len(x_head)):
    hough_input[x_head[i]][y_head[i]] = Grayscale_Image[x_head[i]][y_head[i]]
for i in range(0, len(x_crayon)):
    hough_input[x_crayon[i]][y_crayon[i]] = Grayscale_Image[x_crayon[i]][y_crayon[i]]

for i in range(0, len(x_front_cup)):
    hough_input[x_front_cup[i]][y_front_cup[i]] = Grayscale_Image[x_front_cup[i]][y_front_cup[i]]
for i in range(0, len(x_cone)):
    hough_input[x_cone[i]][y_cone[i]] = Grayscale_Image[x_cone[i]][y_cone[i]]

for i in range(0, len(x_circ1)):
    hough_input[x_circ1[i]][y_circ1[i]] = Grayscale_Image[x_circ1[i]][y_circ1[i]]
for i in range(0, len(x_circ2)):
    hough_input[x_circ2[i]][y_circ2[i]] = Grayscale_Image[x_circ2[i]][y_circ2[i]]

for i in range(0, len(x_circ3)):
    hough_input[x_circ3[i]][y_circ3[i]] = Grayscale_Image[x_circ3[i]][y_circ3[i]]
for i in range(0, len(x_circ4)):
    hough_input[x_circ4[i]][y_circ4[i]] = Grayscale_Image[x_circ4[i]][y_circ4[i]]
    
#Hough lines
plt.figure('After Hough Transform')
lines = hl.Hough_Transform(hough_input)
plt.imshow(sobel_img, cmap='gray') 
for line in lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
plt.title('Hough Lines')
plt.show()