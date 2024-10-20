from otsu_binarization import otsu_binarization,RGB2GRAY
import matplotlib.pyplot as plt
import numpy as np
import cv2
def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Error")
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_list = img_rgb.tolist()
    return img_list
path = "./test.png"
img = read_image(path)
gray_img = RGB2GRAY(img)
converted_img = otsu_binarization(img)
bin_array = np.array(converted_img,dtype=np.uint8)
gray_array = np.array(gray_img,dtype=np.uint8)
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(gray_array,cmap='gray')
ax[0].set_title('Gray img')
ax[0].axis('off')
ax[1].imshow(bin_array,cmap = 'gray')
ax[1].set_title('Binarized img')
ax[1].axis('off')
plt.show()  