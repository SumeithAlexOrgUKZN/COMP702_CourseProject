

# from cProfile import label
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# from numpy import ones

# # allows for any number of images to be placed in a grid
# def plotColourImagesSideBySide(fig, imgArray, labelArray, colourArray, numRows, numColumns):
#     for i in range(len(imgArray)):
#         fig.add_subplot(numRows, numColumns, i+1)
#         plt.imshow(imgArray[i], cmap=colourArray[i])
#         plt.title(labelArray[i], wrap=True)
#         plt.axis('off') #Removes axes

#     plt.tight_layout()
#     plt.show()
#     # plt.savefig("temp.jpg")
# ###
# # allows for any number of images to be placed in a grid
# def saveColourImagesSideBySide(fig, imgArray, labelArray, colourArray, numRows, numColumns):
#     for i in range(len(imgArray)):
#         fig.add_subplot(numRows, numColumns, i+1)
#         plt.imshow(imgArray[i], cmap=colourArray[i])
#         plt.title(labelArray[i], wrap=True)
#         plt.axis('off') #Removes axes

#     plt.tight_layout()
#     # plt.show()
#     plt.savefig("temp.jpg")
# ###

# # file = "Resized_Notes_DataSet\\Resized_010_back_current_1.jpg"
# file = "Notes_DataSet\\010_back_current_1.png"

# # matplotlib seems to be RGB
# img = plt.imread(file)
# # print(img.shape)

# fig = plt.figure(num="Enhancement", figsize=(15, 6))
# plt.clf() # Should clear last plot but keep window open? 

# numRows = 2; numColumns = 4

# imgArray = [img, img[ : ,  : , 0], img[ : ,  : , 1], img[ : ,  : , 2], img, img[ : ,  : , 0], img[ : ,  : , 1], img[ : ,  : , 2]]
# labelArray = ["Original", "Red Channel as red", "Green Channel as green", "Blue Channel as blue", 
#               "Original", "Red Channel as gray", "Green Channel as gray", "Blue Channel as gray"]
# colourArray = ["gray", "Reds", "Greens", "Blues", "gray", "gray", "gray", "gray"]

# # plotColourImagesSideBySide(fig, imgArray, labelArray, colourArray, numRows, numColumns)
# # saveColourImagesSideBySide(fig, imgArray, labelArray, colourArray, numRows, numColumns)
# import os.path

# if (os.path.exists("Resized_Notes_DataSet\\Resized_010_back_current_1.jpg") ):
#     print("Yay")