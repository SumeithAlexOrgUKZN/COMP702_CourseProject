'''
Authors:    Sumeith Ishwanthlal (219006284)
            Alexander Goudemond (219030365)

Program name: functions_all.py

Goal: Holds most of the code used in the project

Summary:

'''

#----------------------------------------------------------------------------------------------------------------Packages Below

# Tkinter is the GUI 
from cProfile import label
import tkinter as tk
from tkinter import filedialog, Toplevel, Radiobutton, IntVar, Button, W, Label

# getcwd == Get Current Working Directory, walk = traverses a directory
from os import getcwd, walk, mkdir, remove
from types import NoneType

# library for image manipulation
import cv2
from cv2 import IMREAD_GRAYSCALE

from matplotlib import pyplot as plt
import numpy as np

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
        text = "10",
        width = 40,
        height = 5, 
        bg = "silver",
    )
    button11 = tk.Button(
        master = buttonFrameMiddle2,
        text = "11",
        width = 40,
        height = 5, 
        bg = "silver",
    )
    button12 = tk.Button(
        master = buttonFrameMiddle2,
        text = "12",
        width = 40,
        height = 5, 
        bg = "silver",
    )
    button13 = tk.Button(
        master = buttonFrameBottom1,
        text = "13",
        width = 40,
        height = 5, 
        bg = "silver",
    )
    button14 = tk.Button(
        master = buttonFrameBottom1,
        text = "14",
        width = 40,
        height = 5, 
        bg = "silver",
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
    button17.pack(side = tk.LEFT); buttonClose.pack(side = tk.LEFT)
###

#------------------------------------------------------------------------------------DataSet Exploration Functions--------------

# here, we look at the original dataset and place results in a matplotlib plot
def printDataSetInfo():
    # print("inside getDataSetInfo()")

    currentDir = getcwd()
    photoPath = currentDir + "\\Notes_DataSet"
    path = walk(photoPath)

    # 3 X 2D Array: Image Sizes, Min/Max Values, Num Pics
    dataSetInfo = getDataSetInfo(path)

    spacers = "-" * 60

    print(spacers, "List of Image Sizes:", spacers, sep="\n")
    for item in dataSetInfo[0]:
        print(item[1])

    print("", spacers, "Global minimum and maximum dimensions:", spacers, sep="\n")
    for item in dataSetInfo[1]:
        print(item[0], item[1])

    print("", spacers, "Total Number of Pictures in DataSet:", spacers, sep="\n")
    print(dataSetInfo[2][0][0], dataSetInfo[2][0][1])

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

    # path comes from os.path() --> enables traversal through directory
    for root, directories, files in path:
        for file in files:
            # get image size, by reading in as grayscale
            image = cv2.imread("Notes_DataSet" + "\\" + file, 0)
            (x, y) = image.shape
            temp = "(" + str(x) + "," + str(y) + ")"

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
    
    absoluteDimensions = [["MinX", str(minX)], ["MaxX", str(maxX)], ["MinY", str(minY)], ["MaxY", str(maxY)]]
    totalPics = [["Total Pics", str(numPics)]]

    # notice 3 X 2D shape
    return [dataSetSizes, absoluteDimensions, totalPics]
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
        conductIndividualResize(img)
    else:
        tellUser("Something went wrong... Unable to resize", labelUpdates)
###

def conductIndividualResize(image):
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
        command=lambda: individualResize(x=int( xValue.get() ), y=int( yValue.get() ), img = image)
    ).pack(anchor=W, side="top")
###

def individualResize(x, y, img):
    # print("Inside individualResize()")

    currentDir = getcwd()
    folder = "\\Notes_DataSet"
    path = walk(currentDir + folder)
    destinationFolder = currentDir + "\\Resized_Notes_DataSet"

    # create directory
    try:
        mkdir(destinationFolder)
    except FileExistsError as uhoh:
        pass
    except Exception as uhoh:
        print("New Error:", uhoh)
        pass

    resizedImage = cv2.resize(img, (y, x)) # note order
    aString = "Resized to (" + str(x) + "," + str(y) + ")"
    cv2.imshow(aString, resizedImage)
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
    folder = "\\Notes_DataSet"
    path = walk(currentDir + folder)
    destinationFolder = currentDir + "\\Resized_Notes_DataSet"

    # create directory
    try:
        mkdir(destinationFolder)
    except FileExistsError as uhoh:
        pass
    except Exception as uhoh:
        print("New Error:", uhoh)
        pass
    
    count1 = 0
    for root, directories, files in path:
        for file in files:
            count1 += 1

            temp = currentDir + folder + "\\" + file
            image = cv2.imread(temp, cv2.IMREAD_UNCHANGED)

            resizedImage = cv2.resize(image, (y, x)) # note order
            cv2.imwrite(destinationFolder + "\\" + file, resizedImage)

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
            
            aString += getFileName(window.filename)

            # create directory
            destinationFolder = "Converted_Notes_DataSet"
            try:
                mkdir(destinationFolder)
            except FileExistsError as uhoh:
                pass
            except Exception as uhoh:
                print("New Error:", uhoh)
                pass
            
            writtenSuccessfully = cv2.imwrite(destinationFolder + "\\" + aString, img)
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
            choicesWindow, text="Enhance", width=35, bg='silver',
            command=lambda: executeEnhancement(
                                intVal=enhanceOption.get(), img=imgGrayscale, 
                                imgName=window.filename
                            ) 
        ).pack()
        # Button above sends the user elsewhere

        Button(choicesWindow, text="Close All Plots", bg="gray", command=lambda: (plt.close('all')) ).pack()
        
    else:
        tellUser("Unable to Get Grayscale Image for Enhancement Window...", labelUpdates)
###

def executeEnhancement(intVal, img, imgName):
    tellUser("Opening now...", labelUpdates)

    # Lets us stick 5 plots in 1 window
    fig = plt.figure(num="Enhancement", figsize=(15, 8))
    plt.clf() # Should clear last plot but keep window open? 

    fig.add_subplot(2, 3, 1)
    message = "B/W JPG Image of: " + getFileName(imgName)
    plt.imshow(img, cmap='gray')
    plt.title(message, wrap=True)
    plt.axis('off') #Removes axes
    
    fig.add_subplot(2, 3, 2)
    message = "Histogram of B/W JPG of: " + getFileName(imgName)
    displayHist(img, message)

    if (intVal == 1):
        histEqualization(img, imgName, fig)
    elif (intVal == 2):
        negImage(img, imgName, fig)
    elif (intVal == 3):
        thresholding(img, imgName, fig)
    elif (intVal == 4):
        logTransform(img, imgName, fig)
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
                                                            float(cValue.get()), float(gammaValue.get()), fig)
                                            ).pack()
        Button(textBoxWindow, text="Close All Plots", bg="gray", command=lambda: (plt.close('all')) ).pack()

    # Because second panel needed for Gamma Transform, plt.show() appears in gammaTransformation()
    if (intVal != 5):
        plt.tight_layout() # Prevents title overlap in display
        plt.show()       
###


def TransformationFunction(message, input, output):
    plt.plot(input, output)
    plt.title(message, wrap=True)
    plt.xlabel('Input Intensity Values')
    plt.ylabel('Output Intensity Values')
###

def gammaTransform(img, imgName, cValue, gammaValue, fig):
    imageEnhanced = np.array(cValue*np.power(img,gammaValue))

    fig.add_subplot(2, 3, 4)
    message = "Histogram of **Enhanced** B/W JPG of: " + getFileName(imgName)
    displayHist(imageEnhanced, message)

    fig.add_subplot(2, 3, 5)
    message = "Gamma Transformation of Image: " + getFileName(imgName)
    plt.imshow(imageEnhanced, cmap='gray') 
    plt.title(message, wrap=True)
    plt.axis('off') #Removes axes

    fig.add_subplot(2, 3, 3)
    message = "Transformation Function: "
    TransformationFunction(message, img, imageEnhanced)

    plt.tight_layout() # Prevents title overlap in display
    plt.show()
###

def logTransform(img, imgName, fig):
    cValue = 255 / np.log(1 + np.max(img))
    imageEnhanced = cValue * np.log(1 + img) 
    # imageEnhanced.reshape(512,512)

    fig.add_subplot(2, 3, 4)
    message = "Histogram of **Enhanced** B/W JPG of: " + getFileName(imgName)
    displayHist(imageEnhanced, message)

    fig.add_subplot(2, 3, 5)
    message = "Logarithmic Transformation of Image: " + getFileName(imgName)
    plt.imshow(imageEnhanced, cmap='gray')
    plt.title(message, wrap=True)
    plt.axis('off') #Removes axes
    
    fig.add_subplot(2, 3, 3)
    message = "Transformation Function: "
    TransformationFunction(message, img, imageEnhanced)
###

def thresholding(img, imgName, fig):
    imageEnhanced = cv2.adaptiveThreshold(src=img, maxValue=255, adaptiveMethod=cv2.BORDER_REPLICATE, thresholdType=cv2.THRESH_BINARY,blockSize=3, C=10)
    
    fig.add_subplot(2, 3, 4)
    message = "Histogram of **Enhanced** B/W JPG of: " + getFileName(imgName)
    displayHist(imageEnhanced, message)

    fig.add_subplot(2, 3, 5)
    message = "Thresholding of Image: " + getFileName(imgName)
    plt.imshow(imageEnhanced, cmap='gray')
    plt.title(message, wrap=True)
    plt.axis('off') #Removes axes
    
    fig.add_subplot(2, 3, 3)
    message = "Transformation Function: "
    TransformationFunction(message, img, imageEnhanced)
###

def negImage(img, imgName, fig):
    imageEnhanced = cv2.bitwise_not(img)
    
    fig.add_subplot(2, 3, 4)
    message = "Histogram of **Enhanced** B/W JPG of: " + getFileName(imgName)
    displayHist(imageEnhanced, message)

    
    fig.add_subplot(2, 3, 5)
    message = "Negative Image of: " + getFileName(imgName)
    plt.imshow(imageEnhanced, cmap='gray')
    plt.title(message, wrap=True)
    plt.axis('off') #Removes axes
    
    fig.add_subplot(2, 3, 3)
    message = "Transformation Function: "
    TransformationFunction(message, img, imageEnhanced)
###

def histEqualization(img, imgName, fig):    
    imgEnhanced = cv2.equalizeHist(img)

    message = "Histogram of **Enhanced** B/W JPG of: " + getFileName(imgName)
    fig.add_subplot(2, 3, 4)
    displayHist(imgEnhanced, message)
    
    message = "Histogram Equalized Image of: " + getFileName(imgName)
    fig.add_subplot(2, 3, 5)
    plt.imshow(imgEnhanced, cmap='gray')
    plt.title(message, wrap=True)
    plt.axis('off') #Removes axes

    message = "Transformation Function: "
    fig.add_subplot(2, 3, 3)
    TransformationFunction(message, img, imgEnhanced)
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
            smoothingWindow, text="Smooth", width=35, bg='silver',
            command=lambda: executeSmoothing(
                                intVal=enhanceOption.get(), 
                                arraySize=int(arrayValue.get()),
                                img=imgGrayscale, 
                                imgName=window.filename
                            ) 
        ).pack()
        # Button above sends the user elsewhere

        Button(smoothingWindow, text="Close All Plots", bg="gray", command=lambda: (plt.close('all')) ).pack()
        
    else:
        tellUser("Unable to Get Grayscale Image for Smoothing Window...", labelUpdates)
###

def executeSmoothing(intVal, arraySize, img, imgName):
    tellUser("Opening now...", labelUpdates)

    fig = plt.figure(num="Smoothing", figsize=(8, 5))
    plt.clf() # Should clear last plot but keep window open? 

    fig.add_subplot(1, 2, 1)
    message = "B\W JPG Image of: " + getFileName(imgName)
    plt.imshow(img, cmap='gray')
    plt.title(message, wrap=True)
    plt.axis('off') #Removes axes

    fig.add_subplot(1, 2, 2)
    if (intVal == 1):
        # histEqualization(img, imgName, fig)
        simpleSmooth(img, imgName, arraySize)
    elif (intVal == 2):
        movingAverageSmooth(img, imgName, arraySize)
    elif (intVal == 3):
        gaussianSmooth(img, imgName, arraySize)
    else:
        medianSmooth(img, imgName, arraySize)

    plt.tight_layout() # Prevents title overlap in display
    plt.show()   
###

def medianSmooth(img, imgName, arraySize):
    median = cv2.medianBlur(img,arraySize)
    plt.imshow(median, cmap='gray')
    plt.title('Median Smooth of '+ getFileName(imgName), wrap=True)
    plt.axis('off') #Removes axes
###

def gaussianSmooth(img, imgName, arraySize):
    blur = cv2.GaussianBlur(img,(arraySize,arraySize),0)
    plt.imshow(blur, cmap='gray')
    plt.title('Gaussian Smooth of '+ getFileName(imgName), wrap=True)
    plt.axis('off') #Removes axes
###

def movingAverageSmooth(img, imgName, arraySize):
    kernel = np.ones((arraySize,arraySize), np.float32)/(arraySize * arraySize) # fills with all 1s
    dst = cv2.filter2D(img,-1,kernel)
    
    plt.subplot(122)
    plt.imshow(dst, cmap='gray')
    plt.title('Moving Average Smooth of '+ getFileName(imgName), wrap=True)
    plt.axis('off') #Removes axes
###

def simpleSmooth(img, imgName, arraySize):
    kernel = np.full((arraySize,arraySize), 1/(arraySize * arraySize)) # fills with numbers in array
    dst = cv2.filter2D(img,-1,kernel)
    
    plt.subplot(122)
    plt.imshow(dst, cmap='gray')
    plt.title('Simple Smooth of '+ getFileName(imgName), wrap=True)
    plt.axis('off') #Removes axes
###

#------------------------------------------------------------------------------------Sharpening Functions Below-----------------

def chooseSharpening():
    window.filename = openGUI("Select an Image to Sharpen")
    success, imgGrayscale = imgToGrayscale(window.filename)

    if (success):
        # Open new window to choose enhancement
        # sharpenWindow = Toplevel(window)
        # sharpenWindow.title("Image Sharpened Below")
        # sharpenWindow.geometry("300x300")

        figure = plt.figure(num="Sharpening", figsize=(10, 5))

        executeSharpening(imgGrayscale, imgName=window.filename, fig=figure) 
        
    else:
        tellUser("Unable to Get Grayscale Image for Sharpening Window...", labelUpdates)
###

def executeSharpening(imgGrayscale, imgName, fig):
    # This filter is enough!
    # kernel = np.array([ [0, -1, 0], 
    #                     [-1, 5, -1], 
    #                     [0, -1, 0] ])
    # blur = cv2.filter2D(imgGrayscale,-1,kernel)
    
    blur = cv2.medianBlur(imgGrayscale, 3)
    edgesOnly = imgGrayscale - blur
    sharpenedImage = imgGrayscale + edgesOnly

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
###

#------------------------------------------------------------------------------------Morphological Functions Below--------------

def chooseMorphology():
    window.filename = openGUI("Select an Image to Smooth")
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

        Button(morphWindow, text="Morph", width=35, bg='gray',
            command=lambda: executeMorphOption(intVal=enhanceOption.get(), binaryArray=imgBinary, imgName=window.filename) 
        ).pack()
        Button(morphWindow, text="Close All Plots", bg="gray", command=lambda: (plt.close('all')) ).pack()
        
    else:
        tellUser("Unable to Get Grayscale Image for Morphological Window...", labelUpdates)
    return True
###

def executeMorphOption(intVal, binaryArray, imgName):
    fig = plt.figure(num="Morphological Changes", figsize=(8, 4))
    plt.clf() # Should clear last plot but keep window open? 

    fig.add_subplot(1, 2, 1)

    plt.imshow(binaryArray, cmap='gray')
    plt.title('Binary Image of '+ getFileName(imgName), wrap=True)

    fig.add_subplot(1, 2, 2)

    if (intVal == 1):
        dilatedArray = executeDilation(array=binaryArray)
        
        plt.imshow(dilatedArray, cmap='gray')
        plt.title('Dilated Binary Image of '+ getFileName(imgName), wrap=True)
    elif (intVal == 2):
        dilatedArray = executeErosion(array=binaryArray)
        
        plt.imshow(dilatedArray, cmap='gray')
        plt.title('Eroded Binary Image of '+ getFileName(imgName), wrap=True)
    elif (intVal == 3):
        dilatedArray = executeOpening(array=binaryArray)
        
        plt.imshow(dilatedArray, cmap='gray')
        plt.title('Opening Binary Image of '+ getFileName(imgName), wrap=True)
    elif (intVal == 4):
        dilatedArray = executeClosing(array=binaryArray)
        
        plt.imshow(dilatedArray, cmap='gray')
        plt.title('Closing Binary Image of '+ getFileName(imgName), wrap=True)
    else:
        dilatedArray = executeBoundaryExtraction(array=binaryArray)
        
        plt.imshow(dilatedArray, cmap='gray')
        plt.title('Boundary of Binary Image of '+ getFileName(imgName), wrap=True)

    plt.show()
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
    return path[ backslashLocation + 1 : ]
###

def showArrays(fig, array, numRows, numColumns):
    for i in range(len(array)):
        fig.add_subplot(numRows, numColumns, i+1)
        plt.table(cellText=array[i], loc='center')
        plt.axis('off') #Removes axes

    plt.tight_layout()
    plt.show()

    tellUser("Changes displayed...", labelUpdates)
###

# allows for any number of images to be placed in a grid
def plotImagesSideBySide(fig, imgArray, labelArray, numRows, numColumns):
    for i in range(len(imgArray)):
        fig.add_subplot(numRows, numColumns, i+1)
        plt.imshow(imgArray[i]) # , cmap='gray' removed
        plt.title(labelArray[i], wrap=True)
        plt.axis('off') #Removes axes

    plt.tight_layout()
    plt.show()

    tellUser("Changes displayed...", labelUpdates)
###