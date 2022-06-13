from cProfile import label
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import ones
from os import getcwd, walk, mkdir
from os.path import exists

from functions_all import imageToColourRGB

'''
    This function takes an image and returns a 3D array, 
    containing Average values and Mode Values for each
    colour channel:

    [
	    [ ["Red Average", "Red Mode"], 		["Value1", "Value2"] ],
	    [ ["Green Average", "Green Mode"], 	["Value1", "Value2"] ],
	    [ ["Blue Average", "Blue Mode"], 	["Value1", "Value2"] ]
    ]
'''
def getColourInfo(img):
    red, green, blue = img[ : ,  : , 0], img[ : ,  : , 1], img[ : ,  : , 2]
    globalAverages = [np.average(red), np.average(blue), np.average(green)]

    globalModes = []
    tempArray = [img[ : ,  : , 0], img[ : ,  : , 1], img[ : ,  : , 2]]
    for i in range(3):
        #find unique values in array along with their counts
        vals, counts = np.unique(tempArray[i], return_counts=True)

        #find mode - mode_value is the index
        mode_value = np.argwhere(counts == np.max(counts))

        # append corresponding value, [][][] Because of image shape - only 1 value contained inside
        globalModes.append(vals[mode_value][0][0])

    # stick them together!
    answer = []
    labels = ["Red", "Green", "Blue"]
    tempArray = []
    for i in range(3):
        tempArray =  [ [labels[i] + " Average", labels[i] + " Mode"] ]
        tempArray.append( [ globalAverages[i] , globalModes[i] ] )

        answer.append(tempArray)

    return answer
###
'''
    Assume that the folderName exists and contains 55 different
    bills.

    This function creates and returns a 3D array in the following way:

    [
	    [ 
          ["Name1", Name2", ..., "Name10", "Name11"],
	      ["Red Average 1", "Red Average 2", ..., "Red Average 10", Red Average 11"],
	      ["Green Average 1", "Green Average 2", ..., "Green Average 10", Green Average 11"],
          ["Blue Average 1", "Blue Average 2", ..., "Blue Average 10", Blue Average 11"],
          ["Red Mode 1", "Red Mode 2", ..., "Red Mode 10", Red Mode 11"],
          ["Green Mode 1", "Green Mode 2", ..., "Green Mode 10", Green Mode 11"],
          ["Blue Mode 1", "Blue Mode 2", ..., "Blue Mode 10", Blue Mode 11"], 
        ],
        
        ...
    ]
'''
def getClustersOfImages(folderName):
    currentDir = getcwd()
    path = walk(currentDir + "\\" + folderName)

    answerArray, tempNames = [], []
    redAverages, greenAverages, blueAverages = [], [], []
    redModes, greenModes, blueModes = [], [], [] 
    count = 0
    for root, directories, files in path:
        for file in files:

            # every 11 pictures, add lists to answerArray
            if (count % 11 == 0) and (count != 0):
                answerArray.append( [tempNames, redAverages, greenAverages, blueAverages, redModes, greenModes, blueModes ] )
                
                # reset
                tempNames = []
                redAverages, greenAverages, blueAverages = [], [], []
                redModes, greenModes, blueModes = [], [], [] 

            
            # get features from file
            temp = currentDir + "\\" + folderName + "\\" + file
            success, img = imageToColourRGB(temp)
            if (success):
                # recall that features is a 3D array - positions of data is known in advance
                features = getColourInfo(img)
                # print(features)

                tempNames.append(file)
                redAverages.append(features[0][1][0]); greenAverages.append(features[1][1][0]); blueAverages.append(features[2][1][0])
                redModes.append(   features[0][1][1]); greenModes.append(   features[1][1][1]); blueModes.append(   features[2][1][1])
            else:
                print(file, "Could not be read in colour...")
                break
                
            count += 1
    
    # after for loop runs - need 1 more entry added
    answerArray.append( [tempNames, redAverages, greenAverages, blueAverages, redModes, greenModes, blueModes ] )

    return answerArray
###

def save3DArray(array, folderName, fileName):

    currentDir = getcwd()
    destinationFolder = currentDir + "\\" + folderName

     # create directory
    try:
        mkdir(destinationFolder)
    except FileExistsError as uhoh:
        pass
    except Exception as uhoh:
        print("New Error:", uhoh)
        pass

    # create string for text file; Recall array is 3D
    rowString = ""
    for a in range(len(array)):
        for b in range(len(array[0])):
            for c in range(len(array[0][0])):
                rowString += str(array[a][b][c]) + " "

            rowString.strip()
            rowString += "\n"

        # rowString += "\n"

    file = open(destinationFolder + "\\" + fileName, "w+")
    file.write(rowString[ : -2]) # ignore last 2 chars
    file.close()
###

'''
    This function will use the existing cluster of colour features, and
    print the results in a human digestible format
'''
def getColourTrends():
    # 1) assume "resized_Notes_DataSet" exists
    
    # 2) check if desiredFile exists
    desiredFile = "Reference_Materials\\all_resized_pictures_colour_features.txt"
    if ( not exists(desiredFile) ):
        folderName = "Resized_Notes_DataSet"
        array = getClustersOfImages(folderName)
        save3DArray(array, "Reference_Materials", "all_resized_pictures_colour_features.txt")

    # 3) read in data from desiredFile
    with open(desiredFile) as f:
        lines = f.readlines()
    
    # should have 5*7 sets of data
    counter = 0
    average, variation = 0.0, 0.0
    answerArray, tempArray = [], []
    labelArray = ["R10", "R20", "R50", "R100", "R200"]
    colourArray = ["Red", "Green", "Blue"]
    valueArray = ["Average", "Average", "Average", "Mode", "Mode", "Mode"]
    labelIndex, colourIndex, valueIndex = 0, 0, 0

    # 4) create vectors
    for line in lines:
        # skip these rows
        if (counter == 0):
            counter += 1
            continue

        if (counter % 7 == 0):
            answerArray.append( [labelArray[labelIndex], tempArray] )
            tempArray = []

            labelIndex += 1
            counter += 1
            continue

        # print(line[:-1]) # remove final newline char
        array = np.array(line[:-1].split())
        float_array = array.astype(float)

        # get average of the array's elements
        average = np.average(float_array)
        variation = (np.max(float_array, axis=0) - np.min(float_array, axis=0)) / 2

        # print(array, ":::", average, variation)

        myString = colourArray[colourIndex] + " " + valueArray[valueIndex]
        tempArray.append([myString, average, variation])

        counter += 1
        colourIndex = (colourIndex + 1) % 3 # keep cycling between 3 colours
        valueIndex = (valueIndex + 1) % 6

    # after for loop runs - need 1 more entry added
    answerArray.append( [labelArray[labelIndex], tempArray] )

    # print(answerArray)
    # print("Yay")
    # 5) return to user
    return answerArray
###

def displayColourTrends():
    array = getColourTrends()

    temp = []

    for a in range(len(array)):
        print(array[a][0]) # print "R10", etc.
        for b in range(1, len(array[0])):
            for c in range(0, len(array[0][1])):
                temp =array[a][b][c]

                print(temp[0], ": ", temp[1], ", Variation: ", temp[2], sep="")
        print()
###

# file1 = "Notes_DataSet\\010_back_current_1.png"
# img1 = plt.imread(file1)

# info = getColourInfo(img1)

# print(info[0])
# print(info[1])
# print(info[2])

#--------------------------------------------------------------

# values = getClustersOfImages("Resized_Notes_DataSet")
# print(values)

#--------------------------------------------------------------

# values = getClustersOfImages("Resized_Notes_DataSet")
# # print(values)
# save3DArray(values, "Reference_Materials", "all_resized_pictures_colour_features.txt")

#--------------------------------------------------------------

# array = getColourTrends()
# print(array)

#--------------------------------------------------------------
displayColourTrends()