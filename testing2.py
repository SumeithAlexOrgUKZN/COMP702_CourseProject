

from cProfile import label
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import ones

# allows for any number of images to be placed in a grid
def plotColourImagesSideBySide(fig, imgArray, labelArray, colourArray, numRows, numColumns):
    for i in range(len(imgArray)):
        fig.add_subplot(numRows, numColumns, i+1)
        plt.imshow(imgArray[i], cmap=colourArray[i])
        plt.title(labelArray[i], wrap=True)
        plt.axis('off') #Removes axes

    plt.tight_layout()
    plt.show()
    # plt.savefig("temp.jpg")
###
# allows for any number of images to be placed in a grid
def saveColourImagesSideBySide(fig, imgArray, labelArray, colourArray, numRows, numColumns):
    for i in range(len(imgArray)):
        fig.add_subplot(numRows, numColumns, i+1)
        plt.imshow(imgArray[i], cmap=colourArray[i])
        plt.title(labelArray[i], wrap=True)
        plt.axis('off') #Removes axes

    plt.tight_layout()
    # plt.show()
    plt.savefig("temp.jpg")
###

def printColourInfo(img):
    red, green, blue = img[ : ,  : , 0], img[ : ,  : , 1], img[ : ,  : , 2]
    globalAverages = [np.average(red), np.average(blue), np.average(green)]

    print("Averages:")
    print("Red:", np.average(red))
    print("Blue:", np.average(green))
    print("Green:", np.average(blue))
    print()

    # print(red, "Average", np.average(red))
    # print(green, "Average", np.average(green))
    # print(blue, "Average", np.average(blue))
    # print()
    # print(img)
    # print("Yay")

    return globalAverages
###

# file = "Resized_Notes_DataSet\\Resized_010_back_current_1.jpg"
file1 = "Notes_DataSet\\010_back_current_1.png"
file2 = "Notes_DataSet\\020_back_current_1.png"
file3 = "Notes_DataSet\\050_back_current_1.png"
file4 = "Notes_DataSet\\100_back_current_1.png"
file5 = "Notes_DataSet\\200_back_current_1.png"

# matplotlib seems to be RGB
img1 = plt.imread(file1)
img2 = plt.imread(file2)
img3 = plt.imread(file3)
img4=  plt.imread(file4)
img5 = plt.imread(file5)

arr1 = printColourInfo(img1)
arr2 = printColourInfo(img2)
arr3 = printColourInfo(img3)
arr4 = printColourInfo(img4)
arr5 = printColourInfo(img5)

print("Averages")
labelArray = ["R", "B", "G"]
for i in range(3):
    print(labelArray[i], ": ", sep="", end="")
    print(arr1[i], "; ", sep="", end="")
    print(arr2[i], "; ", sep="", end="")
    print(arr3[i], "; ", sep="", end="")
    print(arr4[i], "; ", sep="", end="")
    print(arr4[i], "; ", sep="")
# print(img.shape)

# fig = plt.figure(num="Enhancement", figsize=(15, 6))
# plt.clf() # Should clear last plot but keep window open? 

# numRows = 2; numColumns = 4

# imgArray = [img, img[ : ,  : , 0], img[ : ,  : , 1], img[ : ,  : , 2], img, img[ : ,  : , 0], img[ : ,  : , 1], img[ : ,  : , 2]]
# labelArray = ["Original", "Red Channel as red", "Green Channel as green", "Blue Channel as blue", 
#               "Original", "Red Channel as gray", "Green Channel as gray", "Blue Channel as gray"]
# colourArray = ["gray", "Reds", "Greens", "Blues", "gray", "gray", "gray", "gray"]

# plotColourImagesSideBySide(fig, imgArray, labelArray, colourArray, numRows, numColumns)
# saveColourImagesSideBySide(fig, imgArray, labelArray, colourArray, numRows, numColumns)


# import os.path

# if (os.path.exists("Resized_Notes_DataSet\\Resized_010_back_current_1.jpg") ):
#     print("Yay")