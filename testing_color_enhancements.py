# we need to investigate whether we can remove gaussian noise for image, 
# conduct histogram equalization,
# edge detection

import matplotlib.pyplot as plt
import cv2
from functions_all import plotImagesSideBySide, getColourInfo

fig = plt.figure(num="Enhancement", figsize=(15, 6))
plt.clf() # Should clear last plot but keep window open? 

file1 = "MessedUp_Notes_DataSet\\MessedUp_Resized_010_back_current_1.jpg"
img1 = plt.imread(file1)

# # parameters (src, dts, h, hForColorComponents, templateWindowSize, searchWindowSize)
# noiseless_image_colored = cv2.fastNlMeansDenoisingColored(img1,None,10,10,7,21) 
# print(getColourInfo(noiseless_image_colored))

# # plt.imshow(img1)
# plt.imshow(noiseless_image_colored)
# plt.axis("off")

# plt.show()
img1 = cv2.imread(file1)

r = cv2.selectROI(img1)

# plt.imshow(img1)
# # plt.imshow(noiseless_image_colored)
# plt.axis("off")

# plt.show()