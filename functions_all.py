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
from tkinter import Toplevel, Label

# getcwd == Get Current Working Directory, walk = traverses a directory
from os import getcwd, walk

# library for image manipulation
import cv2

from matplotlib import pyplot as plt

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
    buttonFrameMiddle = tk.Frame(experimentWindow)
    buttonFrameBottom = tk.Frame(experimentWindow)

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
        text = "2",
        width = 40,
        height = 5, 
        bg = "silver",
        command = chooseExperimentMethod
    )
    button3 = tk.Button(
        master = buttonFrameTop,
        text = "3",
        width = 40,
        height = 5, 
        bg = "silver",
    )
    button4 = tk.Button(
        master = buttonFrameMiddle,
        text = "4",
        width = 40,
        height = 5, 
        bg = "silver",
    )
    button5 = tk.Button(
        master = buttonFrameMiddle,
        text = "5",
        width = 40,
        height = 5, 
        bg = "silver",
        command = chooseExperimentMethod
    )
    button6 = tk.Button(
        master = buttonFrameMiddle,
        text = "6",
        width = 40,
        height = 5, 
        bg = "silver",
    )
    button7 = tk.Button(
        master = buttonFrameBottom,
        text = "7",
        width = 40,
        height = 5, 
        bg = "silver",
    )
    button8 = tk.Button(
        master = buttonFrameBottom,
        text = "8",
        width = 40,
        height = 5, 
        bg = "silver",
        command = chooseExperimentMethod
    )
    buttonClose = tk.Button(
        master = buttonFrameBottom,
        text = "Exit the Program",
        width = 40,
        height = 5, 
        bg = "silver",
        command = window.quit
    )

    experimentFrame.pack()
    buttonFrameTop.pack(); buttonFrameMiddle.pack(); buttonFrameBottom.pack()

    button1.pack(side = tk.LEFT); button2.pack(side = tk.LEFT); button3.pack(side = tk.RIGHT)
    button4.pack(side = tk.LEFT); button5.pack(side = tk.LEFT); button6.pack(side = tk.RIGHT)
    button7.pack(side = tk.LEFT); button8.pack(side = tk.LEFT); buttonClose.pack(side = tk.RIGHT)
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

def showArrays(fig, array, numRows, numColumns):
    for i in range(len(array)):
        fig.add_subplot(numRows, numColumns, i+1)
        plt.table(cellText=array[i], loc='center')
        plt.axis('off') #Removes axes

    plt.tight_layout()
    plt.show()

    tellUser("Changes displayed...", labelUpdates)
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