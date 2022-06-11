'''
Authors:    Sumeith Ishwanthlal (219006284)
            Alexander Goudemond (219030365)

Program name: functions_all.py

Goal: Holds most of the code used in the project

Summary:

'''

#----------------------------------------------------------------------------------------------------------------Packages Below

# Tkinter is the GUI 
import tkinter as tk
from tkinter import filedialog, Toplevel, Radiobutton, IntVar, Button, W, Label

# library for image manipulation
import cv2

from matplotlib import pyplot as plt

import numpy as np
from numpy import r_
from scipy.ndimage.interpolation import rotate # used in DCT compression

from skimage.segmentation import felzenszwalb # type of segmentation method
from skimage.feature import canny #region filling
from skimage.util import random_noise # several noise options

from mahotas import haar # used for haar transform

from scipy import fft # used for dct transform
from scipy.fftpack import dct, idct # used for Compression
from scipy.ndimage import binary_fill_holes #region filling
from scipy.ndimage import rotate #used to rotate an image
from skimage.util import random_noise # used for to inject random noise

# getcwd == Get Current Working Directory, walk = traverses a directory
from os import getcwd, walk, mkdir, remove
from types import NoneType
 
import random

#--------------------------------------------------------------------------------------------------------------Global Variables

# Global Vars below
global window 
global updateFrame
global labelUpdates

scale = 1.5
resizedHeight = (int) (1024 / scale) 
resizedWidth = (int) (512 / scale)

window = tk.Tk()
window.title("COMP702 Bank Note Recognition")
window.geometry("780x510+0+0")

updateFrame = tk.Frame()

labelUpdates = tk.Label(
    master = updateFrame, 
    text = "Updates will be placed in this box, when necessary",
    font = ("Helvetica", 12),
    compound = 'top',
    width = resizedHeight // 10,
    bg = "gray"
)

#----------------------------------------------------------------------------------------------------------------Functions Below
#------------------------------------------------------------------------------------Experiment Section Functions---------------

# opens another window for user to select additional options
def chooseExperimentMethod():
    # print("Inside chooseExperimentMethod()")

    experimentWindow = Toplevel(window)
    experimentWindow.title("Choose further options below")
    # experimentWindow.geometry("500x500")

    experimentFrame = tk.Frame(experimentWindow)
    buttonFrameTop = tk.Frame(experimentWindow)
    buttonFrameMiddle1 = tk.Frame(experimentWindow)
    buttonFrameMiddle2 = tk.Frame(experimentWindow)
    buttonFrameBottom1 = tk.Frame(experimentWindow)
    buttonFrameBottom2 = tk.Frame(experimentWindow)

    button1 = tk.Button(
        master = buttonFrameTop,
        text = "Get DataSet Information",
        width = 40,
        height = 5, 
        bg = "silver",
        command = printDataSetInfo
    )
    button2= tk.Button(
        master = buttonFrameTop,
        text = "Bulk Changes (Resize)",
        width = 40,
        height = 5, 
        bg = "silver",
        command = conductBulkResize
    )
    button3 = tk.Button(
        master = buttonFrameTop,
        text = "Open an Image",
        width = 40,
        height = 5, 
        bg = "silver",
        command = openTheImage
    )
    button4 = tk.Button(
        master = buttonFrameTop,
        text = "Resize an Image",
        width = 40,
        height = 5, 
        bg = "silver",
        command = resizeTheImage
    )
    button5 = tk.Button(
        master = buttonFrameMiddle1,
        text = "Convert an Image",
        width = 40,
        height = 5, 
        bg = "silver",
        command = convertTheImage
    )
    button6 = tk.Button(
        master = buttonFrameMiddle1,
        text = "Enhance an Image",
        width = 40,
        height = 5, 
        bg = "silver",
        command = chooseEnhancement
    )
    button7 = tk.Button(
        master = buttonFrameMiddle1,
        text = "Smooth an Image",
        width = 40,
        height = 5, 
        bg = "silver",
        comman = chooseSmoothing
    )
    button8 = tk.Button(
        master = buttonFrameMiddle1,
        text = "Sharpen an Image",
        width = 40,
        height = 5, 
        bg = "silver",
        command = chooseSharpening
    )
    button9 = tk.Button(
        master = buttonFrameMiddle2,
        text = "Morphologically Change an Image",
        width = 40,
        height = 5, 
        bg = "silver",
        command = chooseMorphology
    )
    button10 = tk.Button(
        master = buttonFrameMiddle2,
        text = "Apply a Mask",
        width = 40,
        height = 5, 
        bg = "silver",
        command = chooseMask
    )
    button11 = tk.Button(
        master = buttonFrameMiddle2,
        text = "Segment an Image",
        width = 40,
        height = 5, 
        bg = "silver",
        command = chooseSegment
    )
    button12 = tk.Button(
        master = buttonFrameMiddle2,
        text = "Transform an Image",
        width = 40,
        height = 5, 
        bg = "silver",
        command = chooseImageTransformation
    )
    button13 = tk.Button(
        master = buttonFrameBottom1,
        text = "Compress an Image",
        width = 40,
        height = 5, 
        bg = "silver",
        command = chooseCompression
    )
    button14 = tk.Button(
        master = buttonFrameBottom1,
        text = "Bulk change (Messup images)",
        width = 40,
        height = 5, 
        bg = "silver",
        command = conductBulkMessUP
    )
    button15 = tk.Button(
        master = buttonFrameBottom1,
        text = "15",
        width = 40,
        height = 5, 
        bg = "silver",
    )
    button16 = tk.Button(
        master = buttonFrameBottom1,
        text = "16",
        width = 40,
        height = 5, 
        bg = "silver",
    )
    button17 = tk.Button(
        master = buttonFrameBottom2,
        text = "17",
        width = 40,
        height = 5, 
        bg = "silver",
    )
    button18 = tk.Button(
        master = buttonFrameBottom2,
        text = "18",
        width = 40,
        height = 5, 
        bg = "silver",
    )
    buttonClose = tk.Button(
        master = buttonFrameBottom2,
        text = "Exit the Program",
        width = 40,
        height = 5, 
        bg = "silver",
        command = window.quit
    )

    experimentFrame.pack()
    buttonFrameTop.pack(); buttonFrameMiddle1.pack(); buttonFrameMiddle2.pack(); buttonFrameBottom1.pack(); buttonFrameBottom2.pack()

    button1.pack(side = tk.LEFT); button2.pack(side = tk.LEFT); button3.pack(side = tk.LEFT); button4.pack(side = tk.RIGHT)
    button5.pack(side = tk.LEFT); button6.pack(side = tk.LEFT); button7.pack(side = tk.LEFT); button8.pack(side = tk.RIGHT)
    button9.pack(side = tk.LEFT); button10.pack(side = tk.LEFT); button11.pack(side = tk.LEFT); button12.pack(side = tk.RIGHT)
    button13.pack(side = tk.LEFT); button14.pack(side = tk.LEFT); button15.pack(side = tk.LEFT); button16.pack(side = tk.RIGHT)
    button17.pack(side = tk.LEFT); button18.pack(side = tk.LEFT); buttonClose.pack(side = tk.LEFT)
###

#------------------------------------------------------------------------------------DataSet Exploration Functions--------------

# here, we look at the original dataset and place results in a matplotlib plot
def printDataSetInfo():
    # print("inside getDataSetInfo()")

    currentDir = getcwd()
    photoPath = currentDir + "\\Notes_DataSet"
    path = walk(photoPath)

    # 4 X 2D Array: Image Sizes, Min/Max Values, Num Pics
    dataSetInfo = getDataSetInfo(path)

    spacers = "-" * 60

    print(spacers, "List of Image Sizes:", spacers, sep="\n")
    for item in dataSetInfo[0]:
        print(item[1])

    print("", spacers, "Global minimum and maximum dimensions:", spacers, sep="\n")
    for item in dataSetInfo[1]:
        print(item[0], item[1])

    print("", spacers, "Average X and Y Values:", spacers, sep="\n")
    for item in dataSetInfo[2]:
        print(item[0], item[1])

    print("", spacers, "Total Number of Pictures in DataSet:", spacers, sep="\n")
    print(dataSetInfo[3][0][0], dataSetInfo[3][0][1])

    tellUser("Printed in Terminal!", labelUpdates)
###

def getDataSetInfo(path):
    dataSetSizes = [["Example File Name", "Distinct Image Size"]]
    absoluteDimensions = [[]]
    totalPics = [[]]

    temp = ""
    template = []
    numPics = 0
    minX, maxX, minY, maxY = -1, -1, -1, -1

    # get average values
    averageX, averageY = 0, 0

    # path comes from os.path() --> enables traversal through directory
    for root, directories, files in path:
        for file in files:
            # get image size, by reading in as grayscale
            image = cv2.imread("Notes_DataSet" + "\\" + file, 0)
            (x, y) = image.shape
            temp = "(" + str(x) + "," + str(y) + ")"
            averageX += x
            averageY += y

            # instantiate min and max vars
            if (numPics == 0):
                minX, maxX = x, x; minY, maxY = y, y
            elif (x < minX): minX = x
            elif (y < minY): minY = y
            elif (x > maxX): maxX = x
            elif (y > maxY): maxY = y

            # only place unique dimensions
            if (temp not in template):
                template.append(temp)
                dataSetSizes.append( [ file, temp ] )
            
            numPics += 1
    
    averageX = averageX / len(dataSetSizes)
    averageY = averageY / len(dataSetSizes)
    
    absoluteDimensions = [["MinX", str(minX)], ["MaxX", str(maxX)], ["MinY", str(minY)], ["MaxY", str(maxY)]]
    totalPics = [["Total Pics", str(numPics)]]
    averageArray = [["Average X Value", averageX], [averageY, "Average Y Value"]]

    # notice 4 X 2D shape
    return [dataSetSizes, absoluteDimensions, averageArray, totalPics]
###

#------------------------------------------------------------------------------------Open Any Image Functions-------------------

def openTheImage():
    window.filename = openGUI("Select an Image to Open")
    success = False
    img = [[]]

    success, img = getImage(window.filename)

    if (success):
        tellUser("Image opened successfully", labelUpdates)

        cv2.imshow(window.filename, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() #Upon Keypress, close window
    else:
        tellUser("Something went wrong... Unable to open", labelUpdates)
###

def getImage(name):
    if name.endswith(".gif"):
        success, img = getGIF(window.filename)
    elif name.endswith(".raw"):
        success, img = getRAW(window.filename)
    # elif ("binary" in name):
    #     success = displayBinary(name)
    else:
        success, img = getPicture(name)

    if (success):
        return True, img
    else:
        return False, NoneType
###

def getRAW(imgName):
    tellUser("We will try do this for 512 X 512", labelUpdates)
    print("We will try do this for 512 X 512")
    try:
        # open file in text format, 'rb' == binary format for reading
        fd = open(imgName, 'rb') # BufferedReader Stream

        # uint8 --> 8 bit image
        # originalFile = np.fromfile(imgName, dtype=np.uint8)
        # print("Shape:", originalFile.shape) # shows us shape

        # TODO how flexibly discover this? For now, only worry about lena_gray.raw...
        rows = 512
        cols = 512

        # construct array from data in binary file
        f = np.fromfile(fd, dtype = np.uint8, count = rows * cols)
        image = f.reshape((rows, cols)) # (rows, cols) == 1st parameter

        # print(image) #Array containing values

        fd.close()
        return True, image

    except Exception as uhoh:
        print("New Error:", uhoh)
        return False, NoneType
###

def getPicture(imgName):
    try:
        image = cv2.imread(imgName)

        return True, image
    except Exception as uhoh:
        print("New Error:", uhoh)
        return False, NoneType
###

def getGIF(imgName):
    vid_capture = cv2.VideoCapture(imgName)

    if (vid_capture.isOpened() == False):
        tellUser("Error opening the video file", labelUpdates)
        return False, NoneType # need to return tuple
    else:
        fps = vid_capture.get(5)
        # print('Frames per second : ', fps, 'FPS')

        frameCount = vid_capture.get(7)
        # print('Frame count : ', frame_count)

    while(vid_capture.isOpened()):
        readyToRead, frame = vid_capture.read()

        if readyToRead:
            # cv2.imshow(imgName, frame)
            vid_capture.release()
            cv2.destroyAllWindows()

            return True, frame
            
    return False, NoneType # need to return tuple
###

#------------------------------------------------------------------------------------Resize an Image Functions------------------

def resizeTheImage():
    window.filename = openGUI("Select an Image to Resize")
    success = False
    img = [[]]

    success, img = getImage(window.filename)

    if (success):
        conductIndividualResize(img, window.filename)
    else:
        tellUser("Something went wrong... Unable to get the Image", labelUpdates)
###

def conductIndividualResize(image, imgName):
    resizeWindow = Toplevel(window)
    resizeWindow.title("Please enter some values")
    resizeWindow.geometry("300x300")

    xLabel = Label(resizeWindow, text="x = ...").pack() #used for reading instructions
    xValue = tk.Entry(resizeWindow)
    xValue.insert(0, "512")
    xValue.pack() #must be seperate for some reason...

    yLabel = Label(resizeWindow, text="y = ...").pack() #used for reading instructions
    yValue = tk.Entry(resizeWindow)
    yValue.insert(0, "1024")
    yValue.pack() #must be seperate for some reason...

    Button(resizeWindow, text="Do Individual Resize", width=50, bg='gray',
        command=lambda: individualResize(x=int( xValue.get() ), y=int( yValue.get() ), img = image, imgName=imgName, savePic = False)
    ).pack(anchor=W, side="top")
    Button(resizeWindow, text="Save Individual Resize", width=50, bg='gray',
        command=lambda: individualResize(x=int( xValue.get() ), y=int( yValue.get() ), img = image, imgName=imgName, savePic = True)
    ).pack(anchor=W, side="top")
###

def individualResize(x, y, img, imgName, savePic):
    # print("Inside individualResize()")

    resizedImage = cv2.resize(img, (y, x)) # note order

    if (not savePic):
        aString = "Resized to (" + str(x) + "," + str(y) + ")"
        cv2.imshow(aString, resizedImage)
    else:
        folder = "Resized_Individual_Pictures"

        success = saveFile(folder, imgName, "Resized_", resizedImage)
        if (success):
            tellUser("Image Saved successfully", labelUpdates)
        else:
            tellUser("Unable to Save File...", labelUpdates)
###

def conductBulkResize():
    resizeWindow = Toplevel(window)
    resizeWindow.title("Please enter some values")
    resizeWindow.geometry("300x300")

    xLabel = Label(resizeWindow, text="x = ...").pack() #used for reading instructions
    xValue = tk.Entry(resizeWindow)
    xValue.insert(0, "512")
    xValue.pack() #must be seperate for some reason...

    yLabel = Label(resizeWindow, text="y = ...").pack() #used for reading instructions
    yValue = tk.Entry(resizeWindow)
    yValue.insert(0, "1024")
    yValue.pack() #must be seperate for some reason...

    Button(resizeWindow, text="Do Bulk Resize", width=50, bg='gray',
        command=lambda: bulkResize(x=int( xValue.get() ), y=int( yValue.get() ))
    ).pack(anchor=W, side="top")
###

# conducts bulk resize based 
def bulkResize(x, y):
    # print("Inside bulkResize()")

    currentDir = getcwd()
    folder = "Notes_DataSet"
    path = walk(currentDir + "\\" + folder)
    destinationFolder = currentDir + "\\Resized_Notes_DataSet"

    count1 = 0
    for root, directories, files in path:
        for file in files:
            count1 += 1

            temp = currentDir + "\\" + folder + "\\" + file
            image = cv2.imread(temp, cv2.IMREAD_UNCHANGED)

            resizedImage = cv2.resize(image, (y, x)) # note order
            # cv2.imwrite(destinationFolder + "\\" + file, resizedImage)
            success = saveFile(folder="Resized_Notes_DataSet", imgPath=currentDir + "\\" + folder + "\\" + file, imgNameToAppend="Resized_", image=resizedImage)

    path = walk(destinationFolder)
    count2 = 0
    for root, directories, files in path:
        for file in files:
            count2 += 1
    
    if (count1 == count2):
        tellUser("Pictures Resized Successfully", labelUpdates)
    else:
        tellUser("Not all pictures resized...", labelUpdates)
###

#------------------------------------------------------------------------------------Converting Functions Below-----------------

def convertTheImage():
    # open new window - choose Grayscale or Binary
    convertWindow = Toplevel(window)
    convertWindow.title("Convert to...")
    convertWindow.geometry("300x300")

    enhanceOption = IntVar()
    enhanceOption.set(0)

    Radiobutton(convertWindow, text="Grayscale Conversion", variable=enhanceOption, value=1).pack(anchor=W)
    Radiobutton(convertWindow, text="Binary Conversion", variable=enhanceOption, value=2).pack(anchor=W)

    Button(convertWindow, text="Convert and Show", width=35, bg='silver',
            command=lambda: executeConversion(intVal=enhanceOption.get(), show=True) 
        ).pack()
    Button(convertWindow, text="Convert and Save", width=35, bg='silver',
            command=lambda: executeConversion(intVal=enhanceOption.get(), show=False) 
        ).pack()
###

def executeConversion(intVal, show):
    window.filename = openGUI("Select an Image to convert to Grayscale")

    if (intVal == 1):
        success, img = imgToGrayscale(window.filename)
    else:
        success, img = imgToBinary(window.filename)

    if (success):
        if (show):
            tellUser("Image shown!", labelUpdates)
            cv2.imshow("Converted Image", img)
        else:
            if (intVal == 1):
                aString = "Grayscale_"
            else:
                aString = "Binary_"
            
            # aString += getFileName(window.filename)

            # create directory
            destinationFolder = "Converted_Notes_DataSet"
            try:
                mkdir(destinationFolder)
            except FileExistsError as uhoh:
                pass
            except Exception as uhoh:
                print("New Error:", uhoh)
                pass
            
            # writtenSuccessfully = cv2.imwrite(destinationFolder + "\\" + aString, img)
            print(window.filename)
            writtenSuccessfully = saveFile(folder=destinationFolder , imgPath=window.filename, imgNameToAppend=aString, image=img)
            
            if (writtenSuccessfully):
                tellUser("Conversion successful!", labelUpdates)
            else:
                tellUser("Unable to write the converted image...", labelUpdates)
    else:
        tellUser("Conversion unsuccessful...", labelUpdates)
###

def imgToBinary(name):
    success = False
    img = [[]]

    success, img = imgToGrayscale(name)

    if (success):
        success, binaryImage = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)

        if (success):
            return True, binaryImage
        else:
            tellUser("Unable to get the Binary Image", labelUpdates)
            return False, NoneType
    else:
        tellUser("Unable to get the Grayscale Image", labelUpdates)
        return False, NoneType
    
###

def imgToGrayscale(name):
    success = False
    img = [[]]

    success, img = getImage(name)

    if (success):
        # temporarily write it, then read it as grayscale, then delete temp file
        cv2.imwrite("temp_pic.jpg", img)
        pic = cv2.imread("temp_pic.jpg", cv2.IMREAD_GRAYSCALE)
        remove("temp_pic.jpg")

        return True, pic
    else:
        tellUser("Unable to get the Grayscale Image", labelUpdates)
        return False, NoneType
        
###

#------------------------------------------------------------------------------------Enhancement Functions Below----------------

def chooseEnhancement():
    window.filename = openGUI("Select an Image to Enhance")
    success, imgGrayscale = imgToGrayscale(window.filename)

    if (success):
        # Open new window to choose enhancement
        choicesWindow = Toplevel(window)
        choicesWindow.title("Image Enhancements Below")
        choicesWindow.geometry("300x300")

        enhanceOption = IntVar()
        enhanceOption.set(0)
        
        Radiobutton(choicesWindow, text="Histogram Equalisation", variable=enhanceOption, value=1).pack(anchor=W)
        Radiobutton(choicesWindow, text="Point Processing: Negative Image", variable=enhanceOption, value=2).pack(anchor=W)
        Radiobutton(choicesWindow, text="Point Processing: Thresholding", variable=enhanceOption, value=3).pack(anchor=W)
        Radiobutton(choicesWindow, text="Logarithmic Transformations", variable=enhanceOption, value=4).pack(anchor=W)
        Radiobutton(choicesWindow, text="Power Law (Gamma) Transformations", variable=enhanceOption, value=5).pack(anchor=W)

        Button(
            choicesWindow, text="Enhance and Show", width=35, bg='silver',
            command=lambda: executeEnhancement(
                                intVal=enhanceOption.get(), img=imgGrayscale, 
                                imgName=window.filename, show = True
                            ) 
        ).pack()
        Button(
            choicesWindow, text="Enhance and Save", width=35, bg='silver',
            command=lambda: executeEnhancement(
                                intVal=enhanceOption.get(), img=imgGrayscale, 
                                imgName=window.filename, show=False
                            ) 
        ).pack()
        # Button above sends the user elsewhere

        Button(choicesWindow, text="Close All Plots", bg="gray", command=lambda: (plt.close('all')) ).pack()
        
    else:
        tellUser("Unable to Get Grayscale Image for Enhancement Window...", labelUpdates)
###

def executeEnhancement(intVal, img, imgName, show):
    newImg = [[]]
    newMessage = ""

    # Lets us stick 5 plots in 1 window
    fig = plt.figure(num="Enhancement", figsize=(15, 8))
    plt.clf() # Should clear last plot but keep window open? 

    # get the changed image, later on plot it
    if (intVal == 1):
        # 2 variables here used after this loop
        newImg = histEqualization(img)
        newMessage = "HistogramEqualized_"

    elif (intVal == 2):
        # 2 variables here used after this loop
        newImg = negImage(img)
        newMessage = "Negative_"

    elif (intVal == 3):
        # 2 variables here used after this loop
        newImg = thresholding(img)
        newMessage = "Thresholded_"

    elif (intVal == 4):
        # 2 variables here used after this loop
        newImg = logTransform(img)
        newMessage = "LogarithmicTransformation_"

    else:
        textBoxWindow = Toplevel(window)
        textBoxWindow.title("Image Enhancements Below")
        textBoxWindow.geometry("300x300")

        cLabel = Label(textBoxWindow, text="c = ...").pack() #used for reading instructions
        cValue = tk.Entry(textBoxWindow)
        cValue.insert(0, "1.0")
        cValue.pack() #must be seperate for some reason...
        gammaLabel = Label(textBoxWindow, text="gamma = ...").pack() #used for reading instructions
        gammaValue = tk.Entry(textBoxWindow)
        gammaValue.insert(0, "0.5")
        gammaValue.pack() #must be seperate for some reason...

        Button(textBoxWindow, text="Power Law (Gamma) Transformation", 
                bg="silver", command=lambda: gammaTransform(img, imgName,
                                                            float(cValue.get()), float(gammaValue.get()), fig, show = show)
                                            ).pack()
        Button(textBoxWindow, text="Close All Plots", bg="gray", command=lambda: (plt.close('all')) ).pack()

    # Because second panel needed for Gamma Transform, plt.show() appears in gammaTransformation()
    if (intVal != 5):
        if (show):
            tellUser("Opening now...", labelUpdates)

            fig.add_subplot(2, 3, 1)
            message = "B/W JPG Image of: " + getFileName(imgName)
            plt.imshow(img, cmap='gray')
            plt.title(message, wrap=True)
            plt.axis('off') #Removes axes
            
            fig.add_subplot(2, 3, 2)
            message = "Histogram of B/W JPG of: " + getFileName(imgName)
            displayHist(img, message)

            message = "Transformation Function: "
            fig.add_subplot(2, 3, 3)
            TransformationFunction(message, img, newImg)

            message = "Histogram of **Enhanced** B/W JPG of: " + getFileName(imgName)
            fig.add_subplot(2, 3, 4)
            displayHist(newImg, message)
            
            # updated higher up
            message = newMessage +  "of_" + getFileName(imgName)
            fig.add_subplot(2, 3, 5)
            plt.imshow(newImg, cmap='gray')
            plt.title(message, wrap=True)
            plt.axis('off') #Removes axes

            plt.tight_layout() # Prevents title overlap in display
            plt.show()
        else:
            # save image
            destinationFolder = "Enhanced_Individual_Images"
            success = saveFile(destinationFolder, imgName, newMessage, newImg)  
            if (success):
                tellUser("Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)
             
###


def TransformationFunction(message, input, output):
    plt.plot(input, output)
    plt.title(message, wrap=True)
    plt.xlabel('Input Intensity Values')
    plt.ylabel('Output Intensity Values')
###

def gammaTransform(img, imgName, cValue, gammaValue, fig, show):
    imageEnhanced = np.array(cValue*np.power(img,gammaValue))
    newMessage = "GammaTransformation_"

    if (show):
        fig.add_subplot(2, 3, 1)
        message = "B/W JPG Image of: " + getFileName(imgName)
        plt.imshow(img, cmap='gray')
        plt.title(message, wrap=True)
        plt.axis('off') #Removes axes
        
        fig.add_subplot(2, 3, 2)
        message = "Histogram of B/W JPG of: " + getFileName(imgName)
        displayHist(img, message)

        fig.add_subplot(2, 3, 3)
        message = "Transformation Function: "
        TransformationFunction(message, img, imageEnhanced)

        fig.add_subplot(2, 3, 4)
        message = "Histogram of **Enhanced** B/W JPG of: " + getFileName(imgName)
        displayHist(imageEnhanced, message)

        fig.add_subplot(2, 3, 5)
        message = newMessage + "of_" + getFileName(imgName)
        plt.imshow(imageEnhanced, cmap='gray') 
        plt.title(message, wrap=True)
        plt.axis('off') #Removes axes

        plt.tight_layout() # Prevents title overlap in display
        plt.show()
    else:
            # save image
            destinationFolder = "Enhanced_Individual_Images"
            success = saveFile(destinationFolder, imgName, newMessage, imageEnhanced)   
            if (success):
                tellUser("Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)
###

def logTransform(img):
    cValue = 255 / np.log(1 + np.max(img))
    imageEnhanced = cValue * np.log(1 + img) 
    
    return imageEnhanced
###

def thresholding(img):
    imageEnhanced = cv2.adaptiveThreshold(src=img, maxValue=255, adaptiveMethod=cv2.BORDER_REPLICATE, thresholdType=cv2.THRESH_BINARY,blockSize=3, C=10)

    return imageEnhanced
###

def negImage(img):
    imageEnhanced = cv2.bitwise_not(img)

    return imageEnhanced
###

def histEqualization(img):    
    imgEnhanced = cv2.equalizeHist(img)

    return imgEnhanced
###

# bins are qty of histogram pieces, range is for width of graph
def displayHist(img, str):
    plt.title(str, wrap=True)
    plt.hist(img.ravel(), bins=256, range=[0,256])
    plt.xlabel('Gray Levels')
    plt.ylabel('Frequencies')
###

#------------------------------------------------------------------------------------Smoothing Functions Below------------------

def chooseSmoothing():
    window.filename = openGUI("Select an Image to Smooth")
    success, imgGrayscale = imgToGrayscale(window.filename)

    if (success):
        # Open new window to choose enhancement
        smoothingWindow = Toplevel(window)
        smoothingWindow.title("Image Enhancements Below")
        smoothingWindow.geometry("300x300")

        enhanceOption = IntVar()
        enhanceOption.set(0)
        
        Radiobutton(smoothingWindow, text="Simple Smoothing", variable=enhanceOption, value=1).pack(anchor=W)
        Radiobutton(smoothingWindow, text="Moving Average Smoothing", variable=enhanceOption, value=2).pack(anchor=W)
        Radiobutton(smoothingWindow, text="Gaussian Smoothing", variable=enhanceOption, value=3).pack(anchor=W)
        Radiobutton(smoothingWindow, text="Median Smoothing", variable=enhanceOption, value=4).pack(anchor=W)
        # Radiobutton(smoothingWindow, text="Power Law (Gamma) Transformations", variable=enhanceOption, value=5).pack(anchor=W)

        arrayLabel = Label(smoothingWindow, text="Array Square Size = ...").pack() #used for reading instructions
        arrayValue = tk.Entry(smoothingWindow)
        arrayValue.insert(0, "3")
        arrayValue.pack() #must be seperate for some reason...

        Button(
            smoothingWindow, text="Smooth and Show", width=35, bg='silver',
            command=lambda: executeSmoothing(
                                intVal=enhanceOption.get(), 
                                arraySize=int(arrayValue.get()),
                                img=imgGrayscale, 
                                imgName=window.filename,
                                show = True
                            ) 
        ).pack()
        Button(
            smoothingWindow, text="Smooth and Save", width=35, bg='silver',
            command=lambda: executeSmoothing(
                                intVal=enhanceOption.get(), 
                                arraySize=int(arrayValue.get()),
                                img=imgGrayscale, 
                                imgName=window.filename,
                                show = False
                            ) 
        ).pack()

        Button(smoothingWindow, text="Close All Plots", bg="gray", command=lambda: (plt.close('all')) ).pack()
        
    else:
        tellUser("Unable to Get Grayscale Image for Smoothing Window...", labelUpdates)
###

def executeSmoothing(intVal, arraySize, img, imgName, show):
    fig = plt.figure(num="Smoothing", figsize=(8, 5))
    plt.clf() # Should clear last plot but keep window open? 

    newImg = [[]]
    newMessage = ""

    if (intVal == 1):
        newImg = simpleSmooth(img, arraySize)
        newMessage = 'SimpleSmooth_'

    elif (intVal == 2):
        newImg = movingAverageSmooth(img, arraySize)
        newMessage = 'MovingAverageSmooth_'

    elif (intVal == 3):
        newImg = gaussianSmooth(img, arraySize)
        newMessage = 'GaussianSmooth_'

    else:
        newImg = medianSmooth(img, arraySize)
        newMessage = 'MedianSmooth_'
    if (show):
        tellUser("Opening now...", labelUpdates)

        fig.add_subplot(1, 2, 1)
        message = "B\W JPG Image of: " + getFileName(imgName)
        plt.imshow(img, cmap='gray')
        plt.title(message, wrap=True)
        plt.axis('off') #Removes axes

        fig.add_subplot(1, 2, 2)
        plt.subplot(122)
        plt.imshow(newImg, cmap='gray')
        plt.title(newMessage, wrap=True)
        plt.axis('off') #Removes axes

        plt.tight_layout() # Prevents title overlap in display
        plt.show()  
    else:
        # save image
        destinationFolder = "Smoothed_Individual_Images"
        success = saveFile(destinationFolder, imgName, newMessage, newImg)   
        if (success):
            tellUser("Image Saved successfully", labelUpdates)
        else:
            tellUser("Unable to Save File...", labelUpdates)
###

def medianSmooth(img, arraySize):
    median = cv2.medianBlur(img,arraySize)
    
    return median
###

def gaussianSmooth(img, arraySize):
    blur = cv2.GaussianBlur(img,(arraySize,arraySize),0)
    
    return blur
###

def movingAverageSmooth(img, arraySize):
    kernel = np.ones((arraySize,arraySize), np.float32)/(arraySize * arraySize) # fills with all 1s
    dst = cv2.filter2D(img,-1,kernel)
    
    return dst
###

def simpleSmooth(img, arraySize):
    kernel = np.full((arraySize,arraySize), 1/(arraySize * arraySize)) # fills with numbers in array
    dst = cv2.filter2D(img,-1,kernel)

    return dst
###

#------------------------------------------------------------------------------------Sharpening Functions Below-----------------

def chooseSharpening():
    window.filename = openGUI("Select an Image to Sharpen")
    success, imgGrayscale = imgToGrayscale(window.filename)

    if (success):
        figure = plt.figure(num="Sharpening", figsize=(10, 5))

        sharpeningWindow = Toplevel(window)
        sharpeningWindow.title("Image Enhancements Below")
        sharpeningWindow.geometry("300x300")

        Button(sharpeningWindow, text="Sharpen and Show", width=35, 
                bg="silver", command=lambda: executeSharpening(imgGrayscale, imgName=window.filename, fig=figure, show = True) 
        ).pack()
        Button(sharpeningWindow, text="Sharpen and Save", width=35,
                bg="silver", command=lambda: executeSharpening(imgGrayscale, imgName=window.filename, fig=figure, show = False) 
        ).pack()  
        Button(sharpeningWindow, text="Close All Plots", bg="gray", command=lambda: (plt.close('all')) ).pack()      
    else:
        tellUser("Unable to Get Grayscale Image for Sharpening Window...", labelUpdates)
###

def executeSharpening(imgGrayscale, imgName, fig, show):
    # This filter is enough!
    # kernel = np.array([ [0, -1, 0], 
    #                     [-1, 5, -1], 
    #                     [0, -1, 0] ])
    # blur = cv2.filter2D(imgGrayscale,-1,kernel)
    
    blur = cv2.medianBlur(imgGrayscale, 3)
    edgesOnly = imgGrayscale - blur
    sharpenedImage = imgGrayscale + edgesOnly

    if (show):
        fig.add_subplot(1, 3, 1)
        plt.imshow(imgGrayscale, cmap='gray')
        plt.title('B\W Image of: '+ getFileName(imgName), wrap=True)
        plt.axis('off')

        fig.add_subplot(1, 3, 2)
        plt.imshow(edgesOnly, cmap='gray')
        plt.title('Edges of: '+ getFileName(imgName), wrap=True)
        plt.axis('off')

        fig.add_subplot(1, 3, 3)
        plt.imshow(sharpenedImage, cmap='gray')
        plt.title('Sharpened Image of: '+ getFileName(imgName), wrap=True)
        plt.axis('off')

        plt.tight_layout() # Prevents title overlap in display
        plt.show()
    else:
        # save image
        destinationFolder = "Sharpened_Individual_Images"
        success = saveFile(destinationFolder, imgName, "Sharpened_", sharpenedImage)   
        if (success):
            tellUser("Image Saved successfully", labelUpdates)
        else:
            tellUser("Unable to Save File...", labelUpdates)
###

#------------------------------------------------------------------------------------Morphological Functions Below--------------

def chooseMorphology():
    window.filename = openGUI("Select an Image to Morphologically Change")
    success, imgBinary = imgToBinary(window.filename)

    if (success):
        # Open new window to choose enhancement
        morphWindow = Toplevel(window)
        morphWindow.title("Choose an option...")
        morphWindow.geometry("300x300")

        enhanceOption = IntVar()
        enhanceOption.set(0)
        
        Radiobutton(morphWindow, text="Dilation", variable=enhanceOption, value=1).pack(anchor=W)
        Radiobutton(morphWindow, text="Erosion", variable=enhanceOption, value=2).pack(anchor=W)
        Radiobutton(morphWindow, text="Opening", variable=enhanceOption, value=3).pack(anchor=W)
        Radiobutton(morphWindow, text="Closing", variable=enhanceOption, value=4).pack(anchor=W)
        Radiobutton(morphWindow, text="Boundary Extraction", variable=enhanceOption, value=5).pack(anchor=W)
        # Radiobutton(smoothingWindow, text="Power Law (Gamma) Transformations", variable=enhanceOption, value=5).pack(anchor=W)

        Button(morphWindow, text="Morph and Show", width=35, bg='gray',
            command=lambda: executeMorphOption(intVal=enhanceOption.get(), binaryArray=imgBinary, imgName=window.filename, show = True) 
        ).pack()
        Button(morphWindow, text="Morph and Save", width=35, bg='gray',
            command=lambda: executeMorphOption(intVal=enhanceOption.get(), binaryArray=imgBinary, imgName=window.filename, show = False) 
        ).pack()
        Button(morphWindow, text="Close All Plots", bg="gray", command=lambda: (plt.close('all')) ).pack()
        
    else:
        tellUser("Unable to Get Grayscale Image for Morphological Window...", labelUpdates)
    return True
###

def executeMorphOption(intVal, binaryArray, imgName, show):
    fig = plt.figure(num="Morphological Changes", figsize=(8, 4))
    plt.clf() # Should clear last plot but keep window open? 

    newImg = [[]]
    newMessage = ""

    if (intVal == 1):
        newImg = executeDilation(array=binaryArray)
        newMessage = "DilatedBinary_"
        
    elif (intVal == 2):
        newImg = executeErosion(array=binaryArray)
        newMessage = "ErodedBinary_"
        
    elif (intVal == 3):
        newImg = executeOpening(array=binaryArray)
        newMessage = "OpenedBinary_"

    elif (intVal == 4):
        newImg = executeClosing(array=binaryArray)
        newMessage = "ClosingBinary_"
        
    else:
        newImg = executeBoundaryExtraction(array=binaryArray)
        newMessage = "BoundaryBinary_"

    if (show):
        fig.add_subplot(1, 2, 1)

        plt.imshow(binaryArray, cmap='gray')
        plt.title('Binary Image of '+ getFileName(imgName), wrap=True)

        fig.add_subplot(1, 2, 2)
        plt.imshow(newImg, cmap='gray')
        plt.title(newMessage + 'of_'+ getFileName(imgName), wrap=True)

        plt.show()
    else:
        # save image
        destinationFolder = "Morphological_Changed_Individual_Images"
        success = saveFile(destinationFolder, imgName, newMessage, newImg)   
        if (success):
            tellUser("Image Saved successfully", labelUpdates)
        else:
            tellUser("Unable to Save File...", labelUpdates)
###

# here, we get boundary of an image
def executeBoundaryExtraction(array):
    erodedArray = executeErosion(array)
    
    (x, y) = (array.shape)

    newArray = np.array( [[0 for i in range(y)] for j in range(x)] )
    
    for i in range(x):
        for j in range(y):
            temp = array[i][j] - erodedArray[i][j]

            if (temp >= 0):
                newArray[i][j] = temp

    return newArray
###

# here, we close an image
def executeClosing(array):
    dilatedArray = executeDilation(array)
    closedArray = executeErosion(dilatedArray)
    return closedArray
###

# here, we open an image
def executeOpening(array):
    erodedArray = executeErosion(array)
    openedArray = executeDilation(erodedArray)
    return openedArray
###

# here, we erode an image
def executeErosion(array):
    # pattern used in for loop, this is here for reference
    # structuringElement =   [ [0, 1, 0],
    #                          [1, 1, 1],
    #                          [0, 1, 0] ]

    paddedArray = np.pad(array, (1, 1), 'constant', constant_values=(0, 0))
    (x, y) = (paddedArray.shape)

    # This will be dilated - slice later
    newArray = np.array( [[0 for i in range(y)] for j in range(x)] )

    for i in range(1, x-1):
        for j in range(1, y-1):
            if (paddedArray[i-1][j] == 255) and (paddedArray[i][j-1] == 255) and (paddedArray[i][j+1] == 255) and (paddedArray[i+1][j] == 255):
                newArray[i][j] = 255

    return newArray[ 1 : x , 1 : y ] # slice and return
###

# here, we dilate an image
def executeDilation(array):
    # pattern used in for loop, this is here for reference
    # structuringElement =   [ [0, 1, 0],
    #                          [1, 1, 1],
    #                          [0, 1, 0] ]

    paddedArray = np.pad(array, (1, 1), 'constant', constant_values=(0, 0))
    (x, y) = (paddedArray.shape)

    # This will be dilated - slice later
    newArray = np.array( [[0 for i in range(y)] for j in range(x)] )

    for i in range(1, x-1):
        for j in range(1, y-1):
            if (paddedArray[i-1][j] == 255) or (paddedArray[i][j-1] == 255) or (paddedArray[i][j+1] == 255) or (paddedArray[i+1][j] == 255):
                newArray[i][j] = 255

    return newArray[ 1 : x , 1 : y ] # slice and return
###

#------------------------------------------------------------------------------------Mask Functions Below-----------------------

def chooseMask():
    window.filename = openGUI("Select an Image to apply a Mask to")
    success, imgGrayscale= imgToGrayscale(window.filename)

    if (success):
        # Open new window to choose enhancement
        maskWindow = Toplevel(window)
        maskWindow.title("Choose a mask...")
        maskWindow.geometry("300x400")

        maskOption1 = IntVar()
        maskOption1.set(0)

        R1 = Radiobutton(maskWindow, text="Laplacian 3x3", variable=maskOption1, value=1)
        R1.pack(anchor=W, side="top")
        R2 = Radiobutton(maskWindow, text="\'Standard\' Horizontal", variable=maskOption1, value=2)
        R2.pack(anchor=W, side="top")
        R3 = Radiobutton(maskWindow, text="\'Standard\' Vertical", variable=maskOption1, value=3)
        R3.pack(anchor=W, side="top")
        R4 = Radiobutton(maskWindow, text="\'Standard\' +45 degrees", variable=maskOption1, value=4)
        R4.pack(anchor=W, side="top")
        R5 = Radiobutton(maskWindow, text="\'Standard\' -45 degrees", variable=maskOption1, value=5)
        R5.pack(anchor=W, side="top")
        R6 = Radiobutton(maskWindow, text="\'Prewitt\' Horizontal", variable=maskOption1, value=6)
        R6.pack(anchor=W, side="top")
        R7 = Radiobutton(maskWindow, text="\'Prewitt\' Vertical", variable=maskOption1, value=7)
        R7.pack(anchor=W, side="top")
        R8 = Radiobutton(maskWindow, text="\'Prewitt\' +45 degrees", variable=maskOption1, value=8)
        R8.pack(anchor=W, side="top")
        R9 = Radiobutton(maskWindow, text="\'Prewitt\' -45 degrees", variable=maskOption1, value=9)
        R9.pack(anchor=W, side="top")
        R10 = Radiobutton(maskWindow, text="\'Sobel\' Horizontal", variable=maskOption1, value=10)
        R10.pack(anchor=W, side="top")
        R11 = Radiobutton(maskWindow, text="\'Sobel\' Vertical", variable=maskOption1, value=11)
        R11.pack(anchor=W, side="top")
        R12 = Radiobutton(maskWindow, text="\'Sobel\' +45 degrees", variable=maskOption1, value=12)
        R12.pack(anchor=W, side="top")
        R13 = Radiobutton(maskWindow, text="\'Sobel\' -45 degrees", variable=maskOption1, value=13)
        R13.pack(anchor=W, side="top")

        Button(maskWindow, text="Apply Mask and Show", width=35, bg='gray',
            command=lambda: executeMaskOption(intVal=maskOption1.get(), img=imgGrayscale, imgName=window.filename, show=True) 
        ).pack()
        Button(maskWindow, text="Apply Mask and Save", width=35, bg='gray',
            command=lambda: executeMaskOption(intVal=maskOption1.get(), img=imgGrayscale, imgName=window.filename, show=False) 
        ).pack()
        Button(maskWindow, text="Close Plots", width=35, bg='gray',
            command=lambda: (plt.close("Mask Changes"))
        ).pack()

    else:
        tellUser("Unable to Get Grayscale Image for Sharpening Window...", labelUpdates)
    
    return True
###

def executeMaskOption(intVal, img, imgName, show):

    fig = plt.figure(num="Mask Changes", figsize=(8, 4))
    plt.clf() # Should clear last plot but keep window open? 

    newImg = [[]]
    newMessage = ""

    # 7 options
    if (intVal == 1):
        # Laplacian Mask
        newImg, mask = applyLaplacianMask(img)
        newMessage = "LaplacianMask_"

    elif (intVal == 2):
        # Horizontal Mask
        newImg, mask = applyStandardHorizontalMask(img)
        newMessage = "HorizontalStandardMask_"

    elif (intVal == 3):
        # Vertical Mask
        newImg, mask = applyStandardVerticalMask(img)
        newMessage = "VerticalStandardMask_"

    elif (intVal == 4):
        # +45 degree Mask
        newImg, mask = applyStandardPositive45Mask(img)
        newMessage = "Positive45StandardMask_"

    elif (intVal == 5):
        # -45 degree Mask
        newImg, mask = applyStandardNegative45Mask(img)
        newMessage = "Negative45StandardMask_"

    elif (intVal == 6):
        # Horizontal Mask
        newImg, mask = applyPrewittHorizontalMask(img)
        newMessage = "HorizontalPrewittMask_"

    elif (intVal == 7):
        # Vertical Mask
        newImg, mask = applyPrewittVerticalMask(img)
        newMessage = "VerticalPrewittMask_"

    elif (intVal == 8):
        # +45 degree Mask
        newImg, mask = applyPrewittPositive45Mask(img)
        newMessage = "Positive45PrewittMask_"

    elif (intVal == 9):
        # -45 degree Mask
        newImg, mask = applyPrewittNegative45Mask(img)
        newMessage = "Negative45PrewittMask_"

    elif (intVal == 10):
        # Horizontal Mask
        newImg, mask = applySobelHorizontalMask(img)
        newMessage = "HorizontalSobelMask_"

    elif (intVal == 11):
        # Vertical Mask
        newImg, mask = applySobelVerticalMask(img)
        newMessage = "VerticalSobelMask_"

    elif (intVal == 12):
        # +45 degree Mask
        newImg, mask = applySobelPositive45Mask(img)
        newMessage = "Positive45SobelMask_"

    elif (intVal == 13):
        # -45 degree Mask
        newImg, mask = applySobelNegative45Mask(img)
        newMessage = "Negative45SobelMask_"
        
    else:
        tellUser("Select an option...", labelUpdates)
 
    if (show):
        fig.add_subplot(1, 3, 1)

        plt.imshow(img, cmap='gray')
        plt.title('Original Image of '+ getFileName(imgName), wrap=True)

        plotMask(fig, newImg, mask, imgName, newMessage)

        plt.tight_layout() # Prevents title overlap in display
        plt.show()  
    else:
        # save image
        destinationFolder = "Masked_Individual_Images"
        success = saveFile(destinationFolder, imgName, newMessage, newImg)   
        if (success):
            tellUser("Image Saved successfully", labelUpdates)
        else:
            tellUser("Unable to Save File...", labelUpdates)
###

def plotMask(fig, newImg, mask, imgName, newMessage):
    fig.add_subplot(1, 3, 2)
    plt.imshow(newImg, cmap='gray')
    plt.title(newMessage + "of_" + getFileName(imgName), wrap=True)
    plt.axis('off') #Removes axes

    fig.add_subplot(1, 3, 3)
    plt.text(0.3, 0.7, "Mask")
    plt.table(cellText=mask, loc='center')
    plt.axis('off') #Removes axes
###

def applySobelNegative45Mask(img):
    mask = np.array(    [[ 2, -1,  0],
                         [-1,  0,  1],
                         [ 0,  1,  2]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask 
###

def applySobelPositive45Mask(img):
    mask = np.array(    [[ 0,  1,  2],
                         [-1,  0,  1],
                         [-2, -1,  0]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask 
###

def applySobelVerticalMask(img):
    mask = np.array(    [[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask 
###

def applySobelHorizontalMask(img):
    mask = np.array(    [[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask 
###

def applyPrewittNegative45Mask(img):
    mask = np.array(    [[-1, -1,  0],
                         [-1,  0,  1],
                         [ 0,  1,  1]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask 
###

def applyPrewittPositive45Mask(img):
    mask = np.array(    [[ 0,  1,  1],
                         [-1,  0,  1],
                         [-1, -1,  0]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask 
###

def applyPrewittVerticalMask(img):
    mask = np.array(    [[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask 
###

def applyPrewittHorizontalMask(img):
    mask = np.array(    [[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask 
###

def applyStandardNegative45Mask(img):
    mask = np.array(    [[-1, -1,  2],
                         [-1,  2, -1],
                         [ 2, -1, -1]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask 
###

def applyStandardPositive45Mask(img):
    mask = np.array(    [[ 2, -1, -1],
                         [-1,  2, -1],
                         [-1, -1,  2]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask 
###

def applyStandardVerticalMask(img):
    mask = np.array(    [[-1, 2, -1],
                         [-1, 2, -1],
                         [-1, 2, -1]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask 
###

def applyStandardHorizontalMask(img):
    mask = np.array(    [[-1, -1, -1],
                         [ 2,  2,  2],
                         [-1, -1, -1]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask 
###

def applyLaplacianMask(img):
    mask = np.array(    [[1,  1, 1],
                         [1, -8, 1],
                         [1,  1, 1]] 
            )

    newImg = cv2.filter2D(img, -1, mask)

    return newImg, mask
###
#------------------------------------------------------------------------------------Segmentation Functions Below---------------

def chooseSegment():
    # print("Inside segmentation option")
    window.filename = openGUI("Select an Image...")
    success, imgGrayscale= imgToGrayscale(window.filename)

    if (success):

        # Open a window and show all the options to choose from

        # segmentWindow = tk.Tk(baseName = "segment_Window")
        segmentWindow = Toplevel(window)
        segmentWindow.title("Choose a kind of segmentation...")
        segmentWindow.geometry("500x200")

        segmentOption = IntVar()
        segmentOption.set(1)

        # left side buttons
        R1 = Radiobutton(segmentWindow, text="Edge Detection", variable=segmentOption, value=1, width=30)
        R3 = Radiobutton(segmentWindow, text="Watershed Method", variable=segmentOption, value=3, width=30)

        # right side buttons
        R4 = Radiobutton(segmentWindow, text="Thresholding", variable=segmentOption, value=4, width=30)
        R5 = Radiobutton(segmentWindow, text="Region Splitting and Merging", variable=segmentOption, value=5, width=30)
        R6 = Radiobutton(segmentWindow, text="Clustering", variable=segmentOption, value=6, width=30)

        # top labels
        L1 = Label(segmentWindow, text="Discontinuity Options", width=30)
        L2 = Label(segmentWindow, text="Continuity Options", width=30)

        B1 = Button(segmentWindow, text="Choose Segmentation Option", width=50, bg='gray',
            command=lambda: executeSegmentOption(intVal=segmentOption.get(), img=imgGrayscale, imgName=window.filename)
        )
        B2 = Button(segmentWindow, text="Close Plots", width=50, bg='gray',
            command=lambda: ( plt.close("Watershed Changes") )
        )

        # grid layout
        L1.grid(row=0, column=0)
        L2.grid(row=0, column=2)
        R1.grid(row=1, column=0)
        R3.grid(row=2, column=0)
        R4.grid(row=1, column=2)
        R5.grid(row=2, column=2)
        R6.grid(row=3, column=2)
        B1.grid(columnspan=3)
        B2.grid(columnspan=3)

    else:
        tellUser("Unable to Get Grayscale Image for Segmentation Window...", labelUpdates)
    
    return True
###


def executeSegmentOption(intVal, img, imgName):
    # give the user more options based on their choice:
    if (intVal == 1):
        # Edge Detection
        chooseEdgeDetectionMethod(intVal, img, imgName)

    elif (intVal == 3):
        # Watershed Method
        chooseWatershedMethod(intVal, img, imgName)

    elif (intVal == 4):
        # Thresholding
        chooseThresholdingMethod(intVal, img, imgName)

    elif (intVal == 5):
        # Region Splitting and Merging
        chooseRegionBasedMethod(intVal, img, imgName)

    elif (intVal == 6):
        # Clustering
        chooseClusteringMethod(intVal, img, imgName)

    else:
        # should never execute
        tellUser("Select an option...", labelUpdates)

    # print("Inside executeSegmentOption()")  
###

def chooseEdgeDetectionMethod(intVal, img, imgName):
    # print("inside ChooseEdgeDetectionMethod")

    '''
    Canny Edge Detection
    Simple Contour
    Complete Contour Detection
    Felzenszwalb's Segmentation
    '''
    
    edgeDetectionWindow = Toplevel(window)
    edgeDetectionWindow.title("Choose a kind of edgeDetection...")
    edgeDetectionWindow.geometry("300x300")

    threshOption = IntVar()
    threshOption.set(0)

    Radiobutton(edgeDetectionWindow, text="Canny Edge Detection", variable=threshOption, value=1, width=30).pack(anchor=W, side="top")
    Radiobutton(edgeDetectionWindow, text="Simple Contour Detection", variable=threshOption, value=2, width=30).pack(anchor=W, side="top")
    Radiobutton(edgeDetectionWindow, text="Complete Contour Detection", variable=threshOption, value=3, width=30).pack(anchor=W, side="top")
    Radiobutton(edgeDetectionWindow, text="Felzenswalbs Contour Detection", variable=threshOption, value=4, width=30).pack(anchor=W, side="top")

    Button(edgeDetectionWindow, text="Choose Segmentation Option and Show", width=50, bg='gray',
        command=lambda: executeEdgeDetectionChoice(intVal=threshOption.get(), img=img, imgName=imgName, show=True)
    ).pack(anchor=W, side="top")
    Button(edgeDetectionWindow, text="Choose Segmentation Option and Save", width=50, bg='gray',
        command=lambda: executeEdgeDetectionChoice(intVal=threshOption.get(), img=img, imgName=imgName, show=False)
    ).pack(anchor=W, side="top")
    Button(edgeDetectionWindow, text="Close Plots", width=50, bg='gray',
        command=lambda: ( plt.close("Edge Detection Changes") )
    ).pack(anchor=W, side="top")
###

def chooseWatershedMethod(intVal, img, imgName):
    # print("inside ChooseWatershedMethod")

    watershedWindow = Toplevel(window)
    watershedWindow.title("Choose a watershed option...")
    watershedWindow.geometry("300x300")

    Button(watershedWindow, text="Apply Watershed and Show", width=50, bg='gray',
        command=lambda: executeWatershed(img=img, imgName=imgName, show=True)
    ).pack(anchor=W, side="top")
    Button(watershedWindow, text="Apply Watershed and Save", width=50, bg='gray',
        command=lambda: executeWatershed(img=img, imgName=imgName, show=False)
    ).pack(anchor=W, side="top")
    Button(watershedWindow, text="Close Plots", width=50, bg='gray',
        command=lambda: ( plt.close("Watershed Changes") )
    ).pack(anchor=W, side="top")
###

def chooseClusteringMethod(intVal, img, imgName):
    # print("inside ChooseClusteringMethod")

    '''
    K-means

    more can be implemented if we discover them
    '''
    clusteringWindow = Toplevel(window)
    clusteringWindow.title("Choose a kind of clustering...")
    clusteringWindow.geometry("300x300")

    threshOption = IntVar()
    threshOption.set(0)

    Radiobutton(clusteringWindow, text="Iterative K-Means clustering", variable=threshOption, value=1, width=30).pack(anchor=W, side="top")
    # Radiobutton(clusteringWindow, text="Fuzzy-C clustering", variable=threshOption, value=2, width=30).pack(anchor=W, side="top")
    # Radiobutton(clusteringWindow, text="Linear Iterative clustering", variable=threshOption, value=3, width=30).pack(anchor=W, side="top")

    Button(clusteringWindow, text="Choose Segmentation Option and Show", width=50, bg='gray',
        command=lambda: executeClusteringChoice(intVal=threshOption.get(), img=img, imgName=imgName, show=True)
    ).pack(anchor=W, side="top")
    Button(clusteringWindow, text="Choose Segmentation Option and Save", width=50, bg='gray',
        command=lambda: executeClusteringChoice(intVal=threshOption.get(), img=img, imgName=imgName, show=False)
    ).pack(anchor=W, side="top")
    Button(clusteringWindow, text="Close Plots", width=50, bg='gray',
        command=lambda: ( plt.close("Clustering Changes") )
    ).pack(anchor=W, side="top")

###

def chooseRegionBasedMethod(intVal, img, imgName):
    print("inside ChooseRegionBasedMethod")
    '''
    Region Based
    Region growing - implement later, big algorithm
    Region splitting and merging - implement later, big algorithm
    '''

    regionWindow = Toplevel(window)
    regionWindow.title("Choose a kind of region...")
    regionWindow.geometry("300x300")

    option = IntVar()
    option.set(0)

    Radiobutton(regionWindow, text="Region Filling", variable=option, value=1, width=30).pack(anchor=W, side="top")
    # Radiobutton(regionWindow, text="Region Growing", variable=option, value=2, width=30).pack(anchor=W, side="top")
    # Radiobutton(regionWindow, text="Region Splitting and Merging", variable=option, value=3, width=30).pack(anchor=W, side="top")

    Button(regionWindow, text="Choose Segmentation Option and Show", width=50, bg='gray',
        command=lambda: executeRegionChoice(intVal=option.get(), img=img, imgName=imgName, show=True)
    ).pack(anchor=W, side="top")
    Button(regionWindow, text="Choose Segmentation Option and Save", width=50, bg='gray',
        command=lambda: executeRegionChoice(intVal=option.get(), img=img, imgName=imgName, show=False)
    ).pack(anchor=W, side="top")
    Button(regionWindow, text="Close Plots", width=50, bg='gray',
        command=lambda: ( plt.close("Region Based Changes") )
    ).pack(anchor=W, side="top")
###

def chooseThresholdingMethod(intVal, img, imgName):
    print("inside ChooseThresholdingMethod")
    '''
    Simple
    Manual / Iterative Thresholding
    Adaptive
    Otsus method
    '''

    thresholdingWindow = Toplevel(window)
    thresholdingWindow.title("Choose a kind of Thresholding...")
    thresholdingWindow.geometry("300x300")

    threshOption = IntVar()
    threshOption.set(0)

    Radiobutton(thresholdingWindow, text="Simple Thresholding", variable=threshOption, value=1, width=30).pack(anchor=W, side="top")
    Radiobutton(thresholdingWindow, text="Iterative Thresholding", variable=threshOption, value=2, width=30).pack(anchor=W, side="top")
    Radiobutton(thresholdingWindow, text="Adaptive Thresholding", variable=threshOption, value=3, width=30).pack(anchor=W, side="top")
    Radiobutton(thresholdingWindow, text="Otsu's Method", variable=threshOption, value=4, width=30).pack(anchor=W, side="top")

    Button(thresholdingWindow, text="Choose Segmentation Option and Show", width=50, bg='gray',
        command=lambda: executeThresholdingChoice(intVal=threshOption.get(), img=img, imgName=imgName, show=True)
    ).pack(anchor=W, side="top")
    Button(thresholdingWindow, text="Choose Segmentation Option and Save", width=50, bg='gray',
        command=lambda: executeThresholdingChoice(intVal=threshOption.get(), img=img, imgName=imgName, show=False)
    ).pack(anchor=W, side="top")
    Button(thresholdingWindow, text="Close Plots", width=50, bg='gray',
        command=lambda: ( plt.close("Segmentation Changes") )
    ).pack(anchor=W, side="top")
###

def executeWatershed(img, imgName, show):
    # threshold
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    # print(kernel)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    unsure_pic = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, unsure_pic)

    # Marker labelling
    ret, markers = cv2.connectedComponents(unsure_pic)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    # NOW - Watershed method
    watershedImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    finalMarkers = cv2.watershed(watershedImage, markers) #! needs colour image

    watershedImage[finalMarkers == -1] = [255, 0, 0]
    watershedImage = cv2.cvtColor(watershedImage, cv2.COLOR_BGR2GRAY) #! convert back to grayscale
    if (show):
        fig = plt.figure(num="Watershed Changes", figsize=(10, 6))
        plt.clf() # Should clear last plot but keep window open? 
        numRows = 3
        numColumns = 3

        modifiedImageArray = [img, thresh, opening, sure_bg, sure_fg, unknown, markers, watershedImage]
        labelArray = ["Original Image", "Thresholded Image", "Morphological Opened Image", "Known Background (black)", "Known Foreground (white)", 
                        "Unknown Aspects", "Unknown Aspects after Connecting Components (gray)", "Watershed Image"]

        tellUser("Images Shown...", labelUpdates)

        plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
    else:
        # save image
        destinationFolder = "Segmented_Individual_Images"
        newMessage = "Watershed_"
        success = saveFile(destinationFolder, imgName, newMessage, watershedImage)   
        if (success):
            tellUser("Image Saved successfully", labelUpdates)
        else:
            tellUser("Unable to Save File...", labelUpdates)
###

def executeRegionChoice(intVal, img, imgName, show): 
    # print("Inside executeRegionChoice")

    fig = plt.figure(num="Region Based Changes", figsize=(10, 6))
    plt.clf() # Should clear last plot but keep window open? 
    numRows = 1
    numColumns = 2

    if (intVal == 1):
        # region filling
        numRows = 2
        numColumns = 2

        cannyImage = canny(img)
        regionFilled = binary_fill_holes(cannyImage)

        if (show):
            modifiedImageArray = [img, cannyImage, regionFilled]
            labelArray = ["Original Image", "Canny Edge Detection", "Region Filled Image"]

            tellUser("Images Shown...", labelUpdates)

            plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
        else:
            # save image
            destinationFolder = "Segmented_Individual_Images"
            newMessage = "RegionFilled_"
            success = saveFile(destinationFolder, imgName, newMessage, regionFilled)   
            if (success):
                tellUser("Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

    # elif (intVal == 2):
    #     # region growing
    #     # seperate algorithm to implement
        
    # elif (intVal == 3):
    #     # Region Splitting and Merging
    #     # seperate algorithm to implement

    else:
        # should never execute
        tellUser("Select an option...", labelUpdates)
###

def executeEdgeDetectionChoice(intVal, img, imgName, show):

    fig = plt.figure(num="Edge Detection Changes", figsize=(10, 6))
    plt.clf() # Should clear last plot but keep window open? 
    numRows = 1
    numColumns = 2

    if (intVal == 1):
        # Canny Edge Detection
        edge = cv2.Canny(img,100,200)

        if (show):
            modifiedImageArray = [img, edge]
            labelArray = ["Original Image", "Canny Edge Detection"]
            
            tellUser("Images Shown...", labelUpdates)

            plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
        else:
            # save image
            destinationFolder = "Segmented_Individual_Images"
            newMessage = "CannyEdgeDetection_"
            success = saveFile(destinationFolder, imgName, newMessage, edge)   
            if (success):
                tellUser("Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

    elif (intVal == 2):
        # Simple Contour
        numRows = 3
        numColumns = 2

        (x, y) = img.shape
        resizedImg = cv2.resize(img,(256,256))
        # Compute the threshold of the grayscale image
        value1 , threshImg = cv2.threshold(resizedImg, np.mean(resizedImg), 255, cv2.THRESH_BINARY_INV)
        # canny edge detection
        cannyImg = cv2.Canny(threshImg, 0,255)
        # dilate edges detected.
        edges = cv2.dilate(cannyImg, None)

        modifiedImage = cv2.resize(edges, (y, x)) # resize

        if(show):
            modifiedImageArray = [img, resizedImg, threshImg, cannyImg, edges, modifiedImage]
            labelArray = ["Original Image", "Resized Image", "Thresholded Image", "Canny Edges of Thresholded Image", "Dilated Edges", "Restore Sizes"]

            tellUser("Images Shown...", labelUpdates)

            plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
        else:
            # save image
            destinationFolder = "Segmented_Individual_Images"
            newMessage = "SimpleContour_"
            success = saveFile(destinationFolder, imgName, newMessage, modifiedImage)   
            if (success):
                tellUser("Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

    elif (intVal == 3):
        # Complete Contour
        numRows = 4
        numColumns = 2

        (x, y) = img.shape
        resizedImg = cv2.resize(img,(256,256))

        # Compute the threshold of the grayscale image
        value1 , threshImg = cv2.threshold(resizedImg, np.mean(resizedImg), 255, cv2.THRESH_BINARY_INV)

        # canny edge detection
        cannyImg = cv2.Canny(threshImg, 0,255)

        # dilate edges detected.
        edges = cv2.dilate(cannyImg, None)

        # find all the open/closed regions in the image and store (cnt). (-1 subscript since the function returns a two-element tuple)
        # The - pass them through the sorted function to access the largest contours first.
        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]

        # Create a zero pixel mask that has equal shape and size to the original image.
        mask = np.zeros((256,256), np.uint8)

        # Draw the detected contours on the created mask.
        masked = cv2.drawContours(mask, [cnt], -1, 255, -1)

        # bitwise AND operation on the original image (img) and the mask
        dst = cv2.bitwise_and(resizedImg, resizedImg, mask=mask)
        
        modifiedImage = cv2.resize(dst, (y, x)) # resize

        if(show):
            modifiedImageArray = [img, resizedImg, threshImg, cannyImg, edges, masked, dst, modifiedImage]
            labelArray = ["Original Image", "Resized Image", "Thresholded Image", "Canny Edges of Thresholded Image", 
                            "Dilated Edges", "Image after mask", "Image Contours Detected", "resized Image"]

            tellUser("Images Shown...", labelUpdates)

            plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
        else:
            # save image
            destinationFolder = "Segmented_Individual_Images"
            newMessage = "CompleteContour_"
            success = saveFile(destinationFolder, imgName, newMessage, modifiedImage)   
            if (success):
                tellUser("Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)


    elif (intVal == 4):
        # Felzenszwalb's Segmentation
        numRows = 2
        numColumns = 2

        # from skimage.segmentation import felzenszwalb

        res1 = felzenszwalb(img, scale=50)
        res2 = felzenszwalb(img, scale=100)

        if(show):
            modifiedImageArray = [img, res1, res2]
            labelArray = ["Original Image", "Felzenswalb Image, Scale=50", "Felzenswalb Image, Scale=100"]

            tellUser("Images Shown...", labelUpdates)

            plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
        else:
            # save image
            destinationFolder = "Segmented_Individual_Images"
            
            newMessage = "FelzenswalbScale50_"
            success = saveFile(destinationFolder, imgName, newMessage, res1)   
            if (success):
                tellUser("First Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

            newMessage = "FelzenswalbScale100_"
            success = saveFile(destinationFolder, imgName, newMessage, res2)   
            if (success):
                tellUser("Second Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

    else:
        # should never execute
        tellUser("Select an option...", labelUpdates)
###

def executeClusteringChoice(intVal, img, imgName, show):

    fig = plt.figure(num="Clustering Changes", figsize=(10, 6))
    plt.clf() # Should clear last plot but keep window open? 
    numRows = 1 # used in matplotlib function below
    numColumns = 2 # used in matplotlib function below

    if (intVal == 1):
        # K-Means Clustering
        twoDimage = img.reshape((-1,1)) # Transform into a 1D matrix
        # print(twoDimage.shape)
        # print(twoDimage[0])
        twoDimage = np.float32(twoDimage) # float32 data type needed for this function
        # print(twoDimage[0])

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts=10

        modifiedImageArray = [img]
        labelArray = ["Original Image"]
        numRows = 2
        numColumns = 2

        for i in range(2, 5):
            K = i

            ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            result_image = res.reshape((img.shape))

            modifiedImageArray.append(result_image)
            labelArray.append("K-Means Clustering, Size=" + str(i))

        if (show):
            tellUser("Images Shown...", labelUpdates)

            plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
            
        else:
            # save image
            destinationFolder = "Segmented_Individual_Images"
            newMessage = "KMeansClusteringSize"

            for i in range(2, 5):
                tempMessage = newMessage + str(i) + "_"
                success = saveFile(destinationFolder, imgName, tempMessage, modifiedImageArray[i-1])   
                if (success):
                    tellUser("Image Saved successfully", labelUpdates)
                else:
                    tellUser("Unable to Save File...", labelUpdates)

    # elif (intVal == 2):
    #     #

    # elif (intVal == 3):
    #     #
    else:
        # should never execute
        tellUser("Select an option...", labelUpdates)

###

def executeThresholdingChoice(intVal, img, imgName, show):
    # print("Inside executeThresholdingChoice()")

    fig = plt.figure(num="Segmentation Changes", figsize=(8, 4))
    plt.clf() # Should clear last plot but keep window open? 
    numRows = 1 # used in matplotlib function below
    numColumns = 2 # used in matplotlib function below

    # 7 choices
    if (intVal == 1):
        # Simple 
        returnValue, modifiedImage = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        if (show):
            modifiedImageArray = [img, modifiedImage]
            labelArray = ["Original Image", "Simple Thresholding"]

            tellUser("Images Shown...", labelUpdates)

            plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
        else:
            # save image
            destinationFolder = "Segmented_Individual_Images"
            newMessage = "SimpleThresholding_"
            success = saveFile(destinationFolder, imgName, newMessage, modifiedImage)   
            if (success):
                tellUser("Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

    elif (intVal == 2):
        # Iterative Thresholding
        modifiedImageArray = [img]
        labelArray = ["Original Image"]
        numRows = 3
        numColumns = 2

        for i in range(2, 6):
            returnValue, modifiedImage = cv2.threshold(img, (255 // i), 255, cv2.THRESH_BINARY)
            modifiedImageArray.append(modifiedImage)
            labelArray.append("Simple Thresholding using \'" + str(255 // i) + "\'")

        if (show):
            tellUser("Images Shown...", labelUpdates)

            plotImagesSideBySide(fig, modifiedImageArray,  labelArray, numRows, numColumns)
        else:
            # save image
            destinationFolder = "Segmented_Individual_Images"
            newMessage = "IterativeThresholdingSize"

            for i in range(2, 6):
                tempMessage = newMessage + str(255 // i) + "_"

                success = saveFile(destinationFolder, imgName, tempMessage, modifiedImageArray[i-1])   
                if (success):
                    tellUser("Image Saved successfully", labelUpdates)
                else:
                    tellUser("Unable to Save File...", labelUpdates)

    elif (intVal == 3):
        # Adaptive Thresholding
        returnValue1, modifiedImage1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # last 2 parameters below: block size of neighbourhood and constant used
        modifiedImage2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) 
        modifiedImage3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        numRows = 2
        numColumns = 2
        if (show):
            modifiedImageArray = [img, modifiedImage1, modifiedImage2, modifiedImage3]
            labelArray = ["Original Image", "Simple Thresholding", "Adaptive Mean Thresholding", "Adaptive Gaussian Thresholding"]

            tellUser("Images Shown...", labelUpdates)

            plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
        else:
            # save image
            destinationFolder = "Segmented_Individual_Images"

            newMessage = "AdaptiveMean_"
            success = saveFile(destinationFolder, imgName, newMessage, modifiedImage2)   
            if (success):
                tellUser("First Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

            newMessage = "AdaptiveGaussian_"
            success = saveFile(destinationFolder, imgName, newMessage, modifiedImage3)   
            if (success):
                tellUser("First Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)


    elif (intVal == 4):
        # Otsu's Method

        # global thresholding
        ret1,th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Otsu's thresholding
        ret2,th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        ret3,th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if (show):
            # plot all the images and their histograms
            modifiedImageArray = [img, th1, img, th2, blur, th3]
            labelArray = ['Original Image','Global Thresholding (v=127)','Original Image',"Otsu's Thresholding",'Gaussian filtered Image',"Otsu's Thresholding"]

            numRows = 3
            numColumns = 3
            
            x = 0
            for i in range(3):
                x += 1
                fig.add_subplot(numRows, numColumns, x)

                plt.imshow(modifiedImageArray[i*2], cmap='gray')
                plt.title(labelArray[i], wrap=True)
                plt.axis('off') #Removes axes

                x += 1
                fig.add_subplot(numRows, numColumns, x)
                plt.hist(modifiedImageArray[i*2].ravel(), 256)
                plt.title("Histogram", wrap=True)

                x += 1
                fig.add_subplot(numRows, numColumns, x)
                plt.imshow(modifiedImageArray[i + 1], cmap='gray')
                plt.title(labelArray[i*2 +1], wrap=True)
                plt.axis('off') #Removes axes

            tellUser("Images Shown...", labelUpdates)

            plt.tight_layout()
            plt.show()
        else:
            # save image
            destinationFolder = "Segmented_Individual_Images"
            newMessage = "OtsuThresholding_"
            success = saveFile(destinationFolder, imgName, newMessage, th3)   
            if (success):
                tellUser("Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

    else:
        # should never execute
        tellUser("Select an option...", labelUpdates)
###

#------------------------------------------------------------------------------------Image Transformation Functions-------------

def chooseImageTransformation():
    # print("Inside chooseImageTransformationOption()")

    window.filename = openGUI("Select an Image...")

    success, imgGrayscale= imgToGrayscale(window.filename)

    if (success):
        imageTransformationWindow = Toplevel(window)
        imageTransformationWindow.title("Choose a kind of Image Transformation...")
        imageTransformationWindow.geometry("300x300")

        imageTransformOption = IntVar()
        imageTransformOption.set(0)

        Radiobutton(imageTransformationWindow, text="Apply Fourier Transform", variable=imageTransformOption, value=1, width=30).pack(anchor=W, side="top")
        Radiobutton(imageTransformationWindow, text="Apply Haar Transform", variable=imageTransformOption, value=2, width=30).pack(anchor=W, side="top")
        Radiobutton(imageTransformationWindow, text="Apply Discrete Cosine Transform", variable=imageTransformOption, value=3, width=30).pack(anchor=W, side="top")

        Button(imageTransformationWindow, text="Choose Segmentation Option and Show", width=50, bg='gray',
            command=lambda: executeImageTransformationChoice(intVal=imageTransformOption.get(), img=imgGrayscale, imgName=window.filename, show=True)
        ).pack(anchor=W, side="top")
        Button(imageTransformationWindow, text="Choose Segmentation Option and Save", width=50, bg='gray',
            command=lambda: executeImageTransformationChoice(intVal=imageTransformOption.get(), img=imgGrayscale, imgName=window.filename, show=False)
        ).pack(anchor=W, side="top")
        Button(imageTransformationWindow, text="Close Plots", width=50, bg='gray',
            command=lambda: ( plt.close("Image Transformation Changes") )
        ).pack(anchor=W, side="top")
    else:
        tellUser("Unable to Get Grayscale Image for Image Transformation Window...", labelUpdates)
###

def executeImageTransformationChoice(intVal, img, imgName, show):
    # print("Inside executeImageTransformationOption()")

    fig = plt.figure(num="Image Transformation Changes", figsize=(8, 4))
    plt.clf() # Should clear last plot but keep window open? 
    numRows = 1 # used in matplotlib function below
    numColumns = 2 # used in matplotlib function below

    if (intVal == 1):
        # Fourier Transform
        numRows = 2
        numColumns = 2

        # discrete fourier transformation
        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])) # fourier transformed image

        rows, cols = img.shape
        crow, ccol = rows//2 , cols//2

        # create a mask first, center square is 1, remaining all zeros
        mask1 = np.zeros((rows,cols,2),np.uint8)
        mask1[crow-30:crow+30, ccol-30:ccol+30] = 1

        # apply mask and inverse DFT --> Low Pass Filter
        fshift_low_pass = dft_shift*mask1
        f_ishift_low_pass = np.fft.ifftshift(fshift_low_pass)
        img_back_low_pass = cv2.idft(f_ishift_low_pass)
        img_back_low_pass = cv2.magnitude(img_back_low_pass[:,:,0],img_back_low_pass[:,:,1])

        # High Pass Filter
        mask2 = np.fft.fftshift(np.fft.fft2(img))
        mask2[crow-30:crow+30, ccol-30:ccol+30] = 0
        f_ishift_high_pass = np.fft.ifftshift(mask2)
        img_back_high_pass = np.fft.ifft2(f_ishift_high_pass)
        img_back_high_pass = np.log(np.abs(img_back_high_pass))

        if (show):
            modifiedImageArray = [img, magnitude_spectrum, img_back_low_pass, img_back_high_pass]
            labelArray = ["Original Image", "Magnitude Spectrum", "Low Pass Filter", "High Pass Filter"]

            plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
        else:
            # save image
            destinationFolder = "Transformed_Individual_Images"

            newMessage = "LowPassFilter_"
            success = saveFile(destinationFolder, imgName, newMessage, img_back_low_pass)   
            if (success):
                tellUser("First Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

            newMessage = "HighPassFilter_"
            success = saveFile(destinationFolder, imgName, newMessage, img_back_high_pass)   
            if (success):
                tellUser("Second Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

    elif (intVal == 2):
        #Haar Transform

        # haar method comes from mahotas package
        haar_transform = haar(img)

        if (show):
            modifiedImageArray = [img, haar_transform]
            labelArray = ["Original Image", "Haar Transform"]

            plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
        else:
            # save image
            destinationFolder = "Transformed_Individual_Images"
            newMessage = "Haar_"
            success = saveFile(destinationFolder, imgName, newMessage, haar_transform)   
            if (success):
                tellUser("First Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

    elif (intVal == 3):
        numRows = 1
        numColumns = 3

        # Discrete Cosine Transform, from scipy package
        dct_img = fft.dct(img)
        idct_img = fft.idct(dct_img)

        if (show):
            modifiedImageArray = [img, dct_img, idct_img]
            labelArray = ["Original Image", "DCT Image Spectrum", "DCT Transformed Image"]

            plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
        else:
            # save image
            destinationFolder = "Transformed_Individual_Images"
            newMessage = "DCTTransformed_"
            success = saveFile(destinationFolder, imgName, newMessage, idct_img)   
            if (success):
                tellUser("First Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

    else:
        # should never execute
        tellUser("Select an option...", labelUpdates)
###

#------------------------------------------------------------------------------------Compression Functions Below----------------

def chooseCompression():
    # print("Inside chooseImageTransformationOption()")

    window.filename = openGUI("Select an Image...")

    success, imgGrayscale= imgToGrayscale(window.filename)

    if (success):
        compressionWindow = Toplevel(window)
        compressionWindow.title("Choose a kind of Image Transformation...")
        compressionWindow.geometry("300x300")

        compressOption = IntVar()
        compressOption.set(0)

        Radiobutton(compressionWindow, text="Apply DCT Compression", variable=compressOption, value=1, width=30).pack(anchor=W, side="top")

        Button(compressionWindow, text="Choose Compresion Option and Show", width=50, bg='gray',
            command=lambda: executeCompressionChoice(intVal=compressOption.get(), img=imgGrayscale, imgName=window.filename, show=True)
        ).pack(anchor=W, side="top")
        Button(compressionWindow, text="Choose Compresion Option and Save", width=50, bg='gray',
            command=lambda: executeCompressionChoice(intVal=compressOption.get(), img=imgGrayscale, imgName=window.filename, show=False)
        ).pack(anchor=W, side="top")
        Button(compressionWindow, text="Close Plots", width=50, bg='gray',
            command=lambda: ( plt.close("Compression Changes") )
        ).pack(anchor=W, side="top")
    else:
        tellUser("Unable to Get Grayscale Image for Compression Window...", labelUpdates)
###



def executeCompressionChoice(intVal, img, imgName, show):
    # print("Inside executeCompressionOption()")

    fig = plt.figure(num="Compression Changes", figsize=(8, 4))
    plt.clf() # Should clear last plot but keep window open? 
    numRows = 1 # used in matplotlib function below
    numColumns = 2 # used in matplotlib function below

    if (intVal == 1):
        # DCT compression

        imgsize = img.shape
        dct = np.zeros(imgsize)

        # Do 8x8 DCT on image (in-place)
        for i in r_[:imgsize[0]:8]:
            for j in r_[:imgsize[1]:8]:
                dct[i:(i+8),j:(j+8)] = dct2( img[i:(i+8),j:(j+8)] )

        pos = 128
        # 8X8 image block
        img_slice = img[pos:pos+8,pos:pos+8]
        # An 8X8 dct block
        dct_slice = dct[pos:pos+8,pos:pos+8]

        # Threshold our dct image
        thresh = 0.012
        dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))

        # Use thresholded image to compress
        img_idct = np.zeros(imgsize)

        for i in r_[:imgsize[0]:8]:
            for j in r_[:imgsize[1]:8]:
                img_idct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )

        if (show):
            numRows = 2
            numColumns = 3
            modifiedImageArray = [img, dct, img_slice, dct_slice, dct_thresh, img_idct]
            labelArray = ["Original Image", "DCT Image", "An 8X8 Image block", "An 8X8 DCT Block", "DCT Thresholding", "Using DCT to compress"]

            plotImagesSideBySide(fig, modifiedImageArray, labelArray, numRows, numColumns)
        else:
            # save image
            destinationFolder = "Compressed_Individual_Images"
            newMessage = "DCTCompressed_"
            success = saveFile(destinationFolder, imgName, newMessage, img_idct)   
            if (success):
                tellUser("Image Saved successfully", labelUpdates)
            else:
                tellUser("Unable to Save File...", labelUpdates)

    else:
        # should never execute
        tellUser("Select an option...", labelUpdates)
###

def dct2(a):
    return dct( dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )
###

def idct2(a):
    return idct( idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')
###
#------------------------------------------------------------------------------------Messing Functions---------------------------
def conductBulkMessUP():
    currentDir = getcwd()
    folder = "Notes_DataSet"
    path = walk(currentDir + "\\" + folder)
    destinationFolder = currentDir + "\\MessedUp_Notes_DataSet"

    count1 = 0
    for root, directories, files in path:
        for file in files:
            count1 += 1

            temp = currentDir + "\\" + folder + "\\" + file
            image = cv2.imread(temp, cv2.IMREAD_UNCHANGED)

            messedUp = rdnMessup(image)
            # cv2.imwrite(destinationFolder + "\\" + file, resizedImage)
            success = saveFile(folder="Messed_Notes_DataSet", imgPath=currentDir + "\\" + folder + "\\" + file, imgNameToAppend="MessedUp_", image=messedUp)

    path = walk(destinationFolder)
    count2 = 0
    for root, directories, files in path:
        for file in files:
            count2 += 1
    
    if (count1 == count2):
        tellUser("Pictures messed up Successfully", labelUpdates)
    else:
        tellUser("Not all pictures are messsed...", labelUpdates)
###

def rdnMessup(img):
    tempImage = img
    noiseList = ["gaussian", "s&p", "poisson", "speckle"]
    lightIntensityList = [30, 40, 50, 60, 70 ,80, 90, 100, 110]
    scaleList = range(5,95, 5)
    angleList = range(5,360, 5)

    rdnNoise = random.choice(noiseList)
    rdnAngle = random.choice(angleList)
    rdnLightIntensiry = random.choice(lightIntensityList)
    rdnScale = random.choice(scaleList)
    rdnOption = random.randint(0,6)
    
    lightIntensityMatrix = np.ones(img.shape, dtype="uint8") * (rdnLightIntensiry)
    
    if rdnOption == 0:
        tempImage = brightenImage(tempImage, lightIntensityMatrix)
        tempImage = addNoise(tempImage, rdnNoise)
        tempImage = growImage(tempImage, rdnScale)
        tempImage = rotateImage(tempImage, rdnAngle)
    
    if rdnOption == 1:
        tempImage = darkenImage(tempImage, lightIntensityMatrix)
        tempImage = addNoise(tempImage, rdnNoise)
        tempImage = shrinkImage(tempImage, rdnScale)
        tempImage = rotateImage(tempImage, rdnAngle)
    
    if rdnOption == 2:
        tempImage = brightenImage(tempImage, lightIntensityMatrix)
        tempImage = addNoise(tempImage, rdnNoise)
        tempImage = shrinkImage(tempImage, rdnScale)
        tempImage = rotateImage(tempImage, rdnAngle)
    
    if rdnOption == 3:
        tempImage = darkenImage(tempImage, lightIntensityMatrix)
        tempImage = addNoise(tempImage, rdnNoise)
        tempImage = growImage(tempImage, rdnScale)
        tempImage = rotateImage(tempImage, rdnAngle)
    
    if rdnOption == 4:
        tempImage = brightenImage(tempImage, lightIntensityMatrix)
        tempImage = rotateImage(tempImage, rdnAngle)

    if rdnOption == 5:
        tempImage = darkenImage(tempImage, lightIntensityMatrix)
        tempImage = rotateImage(tempImage, rdnAngle)

    if rdnOption == 6:
        tempImage = brightenImage(tempImage, lightIntensityMatrix)
        tempImage = addNoise(tempImage, rdnNoise)
    
    if rdnOption == 7:
        tempImage = darkenImage(tempImage, lightIntensityMatrix)
        tempImage = addNoise(tempImage, rdnNoise)
    


    return tempImage
###


def brightenImage(img, lightIntensityMatrix):
    temp = img
    temp = cv2.add(img, lightIntensityMatrix)
    return temp
###


def darkenImage(img, lightIntensityMatrix):
    temp = img
    temp = cv2.subtract(img, lightIntensityMatrix)
    return temp
###

def addNoise(img, noise_typ):
    temp = img
    noise_img = random_noise(temp, mode=noise_typ)
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img
###

def growImage(img, scale):
    temp = img 

    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    grewImage = cv2.resize(temp, (width, height))
    return grewImage

def shrinkImage(img, scale):
    temp = img 

    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)

    grewImage = cv2.resize(temp, (width, height))
    return grewImage

def rotateImage(img, angle):
    # Rotate an Image
    rotated = rotate(img, angle)
    return rotated
###

def chooseNoise():
    window.filename = openGUI("Select an Image to Noise to")
    success, img= getImage(window.filename)

    if (success):
        # Open new window to choose enhancement
        noiseWindow = Toplevel(window)
        noiseWindow.title("Choose a noise...")
        noiseWindow.geometry("300x400")

        noiseOption = IntVar()
        noiseOption.set(0)

        Radiobutton(noiseWindow, text="Gaussian Noise", variable=noiseOption, value=1).pack(anchor=W, side="top")
        Radiobutton(noiseWindow, text="Salt and Pepper Noise", variable=noiseOption, value=2).pack(anchor=W, side="top")
        Radiobutton(noiseWindow, text="Poisson Noise", variable=noiseOption, value=3).pack(anchor=W, side="top")
        Radiobutton(noiseWindow, text="Speckle Noise", variable=noiseOption, value=4).pack(anchor=W, side="top")

        Button(noiseWindow, text="Apply Noise and Show", width=35, bg='gray',
            command=lambda: executeNoiseOption(intVal=noiseOption.get(), img=img, imgName=window.filename, show=True) 
        ).pack()
        Button(noiseWindow, text="Apply Noise and Save", width=35, bg='gray',
            command=lambda: executeNoiseOption(intVal=noiseOption.get(), img=img, imgName=window.filename, show=False) 
        ).pack()
        Button(noiseWindow, text="Close Plots", width=35, bg='gray',
            command=lambda: (plt.close("Noise Changes"))
        ).pack()

    else:
        tellUser("Unable to Get Image to add Noise...", labelUpdates)
    
    return True
###


def executeNoiseOption(intVal, img, imgName, show):

    fig = plt.figure(num="Noise Changes", figsize=(8, 4))
    plt.clf() # Should clear last plot but keep window open? 

    newImg = [[]]
    newMessage = ""

    # 4 options
    if (intVal == 1):
        # Gaussian Noise
        newImg = addNoise("gaussian", img)
        newMessage = "GaussianNoise_"

    elif (intVal == 2):
        # Salt and Pepper Noise
        newImg = addNoise(img)
        newMessage = "SaltAndPepperNoise_"

    elif (intVal == 3):
        # Poisson Noise
        newImg = addNoise(img)
        newMessage = "PoissonNoise_"

    elif (intVal == 4):
        # Speckle Noise
        newImg = addNoise(img)
        newMessage = "SpeckleNoise_"
        
    else:
        tellUser("Select an option...", labelUpdates)
 
    if (show):
        fig.add_subplot(1, 3, 1)

        plt.imshow(img)
        plt.title('Original Image of '+ getFileName(imgName), wrap=True)

        plotMask(fig, newImg, imgName, newMessage)

        plt.tight_layout() # Prevents title overlap in display
        plt.show()  
    else:
        # save image
        destinationFolder = "Masked_Individual_Images"
        success = saveFile(destinationFolder, imgName, newMessage, newImg)   
        if (success):
            tellUser("Image Saved successfully", labelUpdates)
        else:
            tellUser("Unable to Save File...", labelUpdates)
###


def plotNoise(fig, newImg, imgName, newMessage):
    fig.add_subplot(1, 3, 2)
    plt.imshow(newImg, cmap='gray')
    plt.title(newMessage + "of_" + getFileName(imgName), wrap=True)
    plt.axis('off') #Removes axes

    fig.add_subplot(1, 3, 3)
    plt.text(0.3, 0.7, "Mask")
    plt.table(cellText=mask, loc='center')
    plt.axis('off') #Removes axes
###
#------------------------------------------------------------------------------------Other Functions Below----------------------

# places updated label for user
def tellUser(str, label):
    oldText = label.cget("text")
    endStr = " - Watch this space for more updates..."
    newText = str + endStr

    #updates incase user sees same message twice
    if (oldText == newText):
        newText += "(NEW)"
    label.config(text = newText) #global var
###

def openGUI(message):
    currentDir = getcwd()
    temp = filedialog.askopenfilename(initialdir=currentDir, title=message, 
                                                    filetypes=(
                                                        ("All Files", "*.*"), ("jpg Files", "*.jpg"), 
                                                        ("png files", "*.png"), ("gif files", "*.gif"),
                                                        ("tiff files", "*.tiff"), ("bmp files", "*.bmp"),
                                                        ("raw files", "*.raw")
                                                    )
    )

    return temp
###

def getFileName(path):
    backslashLocation = path.rfind("/")
    if (backslashLocation == -1):
        backslashLocation = path.rfind("\\")
    return path[ backslashLocation + 1 : ]
###

# allows for any number of images to be placed in a grid
def plotImagesSideBySide(fig, imgArray, labelArray, numRows, numColumns):
    for i in range(len(imgArray)):
        fig.add_subplot(numRows, numColumns, i+1)
        plt.imshow(imgArray[i], cmap='gray')
        plt.title(labelArray[i], wrap=True)
        plt.axis('off') #Removes axes

    plt.tight_layout()
    plt.show()

    tellUser("Changes displayed...", labelUpdates)
###

def saveFile(folder, imgPath, imgNameToAppend, image):

    currentDir = getcwd()
    destinationFolder = currentDir + "\\" + folder

     # create directory
    try:
        mkdir(destinationFolder)
    except FileExistsError as uhoh:
        pass
    except Exception as uhoh:
        print("New Error:", uhoh)
        pass
    
    location = destinationFolder + "\\" + imgNameToAppend + getFileName(imgPath)

    # all converted images need to be jpg (incase .raw / .gif come up - for consistency)
    fullstop = location.rfind(".")
    location = location[ : fullstop] + ".jpg"

    success = cv2.imwrite(location, image) # True or False
    return success
###