import matplotlib.pyplot as plt
import cv2


def plotImagesSideBySide(fig, imgArray, labelArray, numRows, numColumns):
    for i in range(len(imgArray)):
        fig.add_subplot(numRows, numColumns, i+1)
        plt.imshow(imgArray[i], cmap='gray')
        plt.title(labelArray[i], wrap=True)
        plt.axis('off') #Removes axes

    plt.tight_layout()
    plt.show()

###
fig = plt.figure(num="Processing", figsize=(8, 4))
plt.clf() # Should clear last plot but keep window open?

file1 = "MessedUp_Notes_DataSet\\MessedUp_Resized_100_back_old_1.jpg"
file2 = "Aligned_Individual_Images\\Realigned_MessedUp_Resized_100_back_old_1.jpg"
file3 = "Enhanced_Individual_Images\\HistogramEqualized_Realigned_MessedUp_Resized_100_back_old_1.jpg"
file4 = "Masked_Individual_Images\\VerticalSobelMask_HistogramEqualized_Realigned_MessedUp_Resized_100_back_old_1.jpg"
file5 = "Masked_Individual_Images\\HorizontalSobelMask_HistogramEqualized_Realigned_MessedUp_Resized_100_back_old_1.jpg"
file6 = "Smoothed_Individual_Images\\GaussianSmooth_HistogramEqualized_Realigned_MessedUp_Resized_100_back_old_1.jpg"


file7 = "Sharpened_Individual_Images\\Sharpened_HistogramEqualized_Realigned_MessedUp_Resized_100_back_old_1.jpg"
file8 = "Converted_Notes_DataSet\\Binary_HistogramEqualized_Realigned_MessedUp_Resized_100_back_old_1.jpg"

file9 = "Morphological_Changed_Individual_Images\\BoundaryBinary_HistogramEqualized_Realigned_MessedUp_Resized_100_back_old_1.jpg"
file10 = "Morphological_Changed_Individual_Images\\ClosingBinary_HistogramEqualized_Realigned_MessedUp_Resized_100_back_old_1.jpg"
file11 = "Morphological_Changed_Individual_Images\\OpenedBinary_HistogramEqualized_Realigned_MessedUp_Resized_100_back_old_1.jpg"
file12 = "Transformed_Individual_Images\\HighPassFilter_HistogramEqualized_Realigned_MessedUp_Resized_100_back_old_1.jpg"
file13 = "Transformed_Individual_Images\\LowPassFilter_HistogramEqualized_Realigned_MessedUp_Resized_100_back_old_1.jpg"

img1 = plt.imread(file1)
img2 = plt.imread(file2)
img3 = plt.imread(file3)
img4 = plt.imread(file4)
img5 = plt.imread(file5)
img6 = plt.imread(file6)
img7 = plt.imread(file7)
img8 = plt.imread(file8)
img9 = plt.imread(file9)
img10 = plt.imread(file10)
img11 = plt.imread(file11)
img12 = plt.imread(file12)
img13 = plt.imread(file13)

numRows = 5
numColumns = 3
imgArray = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12, img13]
labelArray = ["Original Image", "Re-aligned Image", "Histogram Equalized Image",
                "Vertical Sobel Mask", "Horizontal Sobel Mask", "Gaussian Smoothed Image",  
                "Sharpened Image", "Binary Image", "Boundary Image", "Closed Image", "Opened",
                "High Pass Transform", "Low Pass Transform"

]

plotImagesSideBySide(fig, imgArray, labelArray, numRows, numColumns)