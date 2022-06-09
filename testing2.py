

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

    # print("Averages:")
    # print("Red:", np.average(red))
    # print("Blue:", np.average(green))
    # print("Green:", np.average(blue))
    # print()

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

file6 = "Notes_DataSet\\010_front_current_1.png"
file7 = "Notes_DataSet\\020_front_current_1.png"
file8 = "Notes_DataSet\\050_front_current_1.png"
file9 = "Notes_DataSet\\100_front_current_1.png"
file10 = "Notes_DataSet\\200_front_current_1.png"

# matplotlib seems to be RGB
img1 = plt.imread(file1)
img2 = plt.imread(file2)
img3 = plt.imread(file3)
img4=  plt.imread(file4)
img5 = plt.imread(file5)

img6 = plt.imread(file6)
img7 = plt.imread(file7)
img8 = plt.imread(file8)
img9 =  plt.imread(file9)
img10 = plt.imread(file10)

arr1 = printColourInfo(img1)
arr2 = printColourInfo(img2)
arr3 = printColourInfo(img3)
arr4 = printColourInfo(img4)
arr5 = printColourInfo(img5)

arr6 = printColourInfo(img6)
arr7 = printColourInfo(img7)
arr8 = printColourInfo(img8)
arr9 = printColourInfo(img9)
arr10 = printColourInfo(img10)

# print("Averages")
# labelArray = ["R", "B", "G"]
# for i in range(3):
#     print(labelArray[i], ": ", sep="", end="")
#     print("(", arr1[i], ", ",  arr6[i], "); ", sep="", end="")
#     print("(", arr2[i], ", ", arr7[i],  "); ", sep="", end="")
#     print("(", arr3[i], ", ", arr8[i],  "); ", sep="", end="")
#     print("(", arr4[i], ", ", arr9[i],  "); ", sep="", end="")
#     print("(", arr5[i], ", ", arr10[i],  "); ", sep="")
# print(img.shape)

globalColourFeatures = [[0.0 for i in range(5)] for j in range(4)]
labels = ["(R10-Back, R10-Front)", "(R20-Back, R20-Front)", "(R50-Back, R50-Front)", 
            "(R100-Back, R100-Front)", "(R200-Back, R200-Front)"]

globalColourFeatures[0] = labels
temp = ""

for i in range(3):
    temp = "(" + str(arr1[i]) + ", " + str(arr6[i]) + ")"
    globalColourFeatures[i+1][0] = temp

    temp = "(" + str(arr2[i]) + ", " + str(arr7[i]) + ")"
    globalColourFeatures[i+1][1] = temp

    temp = "(" + str(arr3[i]) + ", " + str(arr8[i]) + ")"
    globalColourFeatures[i+1][2] = temp

    temp = "(" + str(arr4[i]) + ", " + str(arr9[i]) + ")"
    globalColourFeatures[i+1][3] = temp

    temp = "(" + str(arr5[i]) + ", " + str(arr10[i]) + ")"
    globalColourFeatures[i+1][4] = temp
 
# print(globalColourFeatures)

fig = plt.figure(num="Enhancement", figsize=(15, 6))
plt.clf() # Should clear last plot but keep window open? 

plt.table(cellText=globalColourFeatures, loc='center')
plt.axis('off') #Removes axes
plt.show()


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