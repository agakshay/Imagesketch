# -*- coding: utf-8 -*-
"""

@author: akshay
"""


import cv2
# import scipy.ndimage
import matplotlib.pyplot as plt
image = cv2.imread("D:\\My_project\\Image segmentation model\\Main image folde\\sampleimage.jpg")

RGBimage= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.title('RGB _image')
plt.imshow(RGBimage, vmin=0, vmax=255)

gr_img= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.subplot(221)
plt.title('Gray_image')
plt.imshow(gr_img,cmap='gray', vmin=0, vmax=255)
inverted_img = 255-gr_img
plt.subplot(222)
plt.title('Inverted_image')
plt.imshow(inverted_img,cmap='gray', vmin=0, vmax=255)

# blurring = scipy.ndimage.filters.gaussian_filter(inverted_img,sigma=5)
blurring= cv2.GaussianBlur(inverted_img,(21,21),0)    
plt.subplot(223)
plt.title('Blurred_image')
plt.imshow(blurring,cmap='gray', vmin=0, vmax=255)


def sketch(img1,img2): 
    img=img1*255/(255-img2)  
    # The the pixel values has to be adjusted and re-assigned
    img[img>1]=255
    img[img2==255]=255
    return img.astype('uint8')

final_img= sketch(blurring,gr_img)
plt.subplot(224)
plt.title('Sketch_image')
plt.imshow(final_img,cmap='gray', vmin=0, vmax=255)

plt.imsave('Image.png', final_img, cmap='gray', vmin=0, vmax=255)




