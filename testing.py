from scipy import ndimage
import cv2

# imgName = "Notes_DataSet\\050_front_old_1.jpg"

# image = cv2.imread(imgName)
# cv2.imshow(imgName, image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #rotation angle in degree
# rotated = ndimage.rotate(image, 15)
# cv2.imshow("Rotated", rotated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#---------------------------------------------------------------------------------------

# image = cv2.imread(imgName, 0)
# (x, y) = image.shape

# resizedImage = cv2.resize(image, (1024, 512), interpolation = cv2.INTER_AREA)
# cv2.imshow("Resized", resizedImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#---------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import cv2

fig = plt.figure(num="Enhancement", figsize=(15, 8))
plt.clf() # Should clear last plot but keep window open? 

file1 = "Notes_DataSet\\010_back_current_1.png"
img1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
file2 = "Resized_Notes_DataSet\\Resized_010_back_current_1.jpg"
img2 = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)
file3 = "Enhanced_Individual_Images\\Histogram_Equalized_Resized_010_back_current_1.jpg"
img3 = cv2.imread(file3, cv2.IMREAD_GRAYSCALE)
file4 = "Enhanced_Individual_Images\\Negative_Resized_010_back_current_1.jpg"
img4 = cv2.imread(file4, cv2.IMREAD_GRAYSCALE)
file5 = "Smoothed_Individual_Images\\Gaussian_Smooth_Resized_010_back_current_1.jpg"
img5 = cv2.imread(file5, cv2.IMREAD_GRAYSCALE)
file6 = "Sharpened_Individual_Images\\Sharpened_Gaussian_Smooth_Resized_010_back_current_1.jpg"
img6 = cv2.imread(file6, cv2.IMREAD_GRAYSCALE)
file7 = "Smoothed_Individual_Images\\Gaussian_Smooth_Histogram_Equalized_Resized_010_back_current_1.jpg"
img7 = cv2.imread(file7, cv2.IMREAD_GRAYSCALE)
file8 = "Sharpened_Individual_Images\\Sharpened_Gaussian_Smooth_Histogram_Equalized_Resized_010_back_current_1.jpg"
img8 = cv2.imread(file8, cv2.IMREAD_GRAYSCALE)
file9 = "Smoothed_Individual_Images\\Gaussian_Smooth_Negative_Resized_010_back_current_1.jpg"
img9 = cv2.imread(file9, cv2.IMREAD_GRAYSCALE)
file10 = "Sharpened_Individual_Images\\Sharpened_Gaussian_Smooth_Negative_Resized_010_back_current_1.jpg"
img10 = cv2.imread(file10, cv2.IMREAD_GRAYSCALE)


# cv2.imshow("THING", cv2.imread(file2, cv2.IMREAD_GRAYSCALE))
# cv2.waitKey()

numRows = 4; numColumns = 3

fig.add_subplot(numRows, numColumns, 1)
plt.imshow(img1, cmap='gray')
plt.title("Original", wrap=True)
plt.axis('off') #Removes axes

fig.add_subplot(numRows, numColumns, 2)
plt.imshow(img2, cmap='gray')
plt.title("Resized", wrap=True)
plt.axis('off') #Removes axes

fig.add_subplot(numRows, numColumns, 3)
plt.imshow(img5, cmap='gray')
plt.title("Gaussian Smooth on Resized", wrap=True)
plt.axis('off') #Removes axes

fig.add_subplot(numRows, numColumns, 4)
plt.imshow(img6, cmap='gray')
plt.title("Sharpen on Gaussian Smooth on Resized", wrap=True)
plt.axis('off') #Removes axes

fig.add_subplot(numRows, numColumns, 5)
plt.imshow(img3, cmap='gray')
plt.title("Histogram Equalized on Resized", wrap=True)
plt.axis('off') #Removes axes

fig.add_subplot(numRows, numColumns, 6)
plt.imshow(img7, cmap='gray')
plt.title("Gaussian Smooth on Histogram Equalized on Resized", wrap=True)
plt.axis('off') #Removes axes

fig.add_subplot(numRows, numColumns, 7)
plt.imshow(img8, cmap='gray')
plt.title("Sharpen on Gaussian Smooth on Histogram Equalized on resized", wrap=True)
plt.axis('off') #Removes axes

fig.add_subplot(numRows, numColumns, 8)
plt.imshow(img4, cmap='gray')
plt.title("Negative on Resized", wrap=True)
plt.axis('off') #Removes axes

fig.add_subplot(numRows, numColumns, 9)
plt.imshow(img9, cmap='gray')
plt.title("Gaussian Smooth on Negative on resized", wrap=True)
plt.axis('off') #Removes axes

fig.add_subplot(numRows, numColumns, 10)
plt.imshow(img10, cmap='gray')
plt.title("Sharpen on Gaussian Smooth on Negative on resized", wrap=True)
plt.axis('off') #Removes axes

plt.tight_layout() # Prevents title overlap in display
plt.show()