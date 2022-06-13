'''
Authors:    Sumeith Ishwanthlal (219006284)
            Alexander Goudemond (219030365)

Program name: bank_note_recognition.py

Goal: Holds code for main GUI here

Summary:

'''
#----------------------------------------------------------------------------------------------------------------Packages Below
# global Vars to use
from functions_all import chooseExperimentMethod, conductPrediction, window, labelUpdates, updateFrame, resizedWidth, resizedHeight, checkForDependencies

# PIL to find Image Path for image in GUI, tkinter for GUI
from PIL import Image, ImageTk
import tkinter as tk

# getcwd == Get Current Working Directory
from os import getcwd


#------------------------------------------------------------------------------------Frames Below-------------------------------

frame = tk.Frame()
buttonFrame = tk.Frame()

#------------------------------------------------------------------------------------Labels Below-------------------------------
currentDir = getcwd()
# lets us add image inside Label
photoPath = Image.open(currentDir + "\\Notes_DataSet\\010_back_youngMandela_1.jpeg")

resizedPhoto = photoPath.resize( (resizedHeight, resizedWidth), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(resizedPhoto)
# place picture and info on frame
label = tk.Label(
    master = frame, 
    text = "Welcome! Please choose a button below to begin",
    font = ("Helvetica", 14),
    compound = 'bottom',
    image = photo,
    bg = "silver"
)

#------------------------------------------------------------------------------------Buttons Below------------------------------

buttonClassify = tk.Button(
    master = buttonFrame,
    text = "Classify an Image",
    width = resizedHeight // (10 * 2),
    height = 5, 
    bg = "silver",
    command = conductPrediction
)
buttonExperiment = tk.Button(
    master = buttonFrame,
    text = "Experiment with Images",
    width = resizedHeight // (10 * 2),
    height = 5, 
    bg = "silver",
    command = chooseExperimentMethod
)
buttonClose = tk.Button(
    master = buttonFrame,
    text = "Exit the Program",
    width = resizedHeight // (10 * 2),
    height = 5, 
    bg = "silver",
    command = window.quit
)

#------------------------------------------------------------------------------------pack() below-------------------------------

frame.pack()
updateFrame.pack()
buttonFrame.pack()

label.pack()
labelUpdates.pack()

buttonClassify.pack(side = tk.LEFT)
buttonExperiment.pack(side = tk.LEFT)
buttonClose.pack(side = tk.RIGHT)

#------------------------------------------------------------------------------------Open Below---------------------------------

print("\n---Program Starting---")

checkForDependencies()

# open the window
window.mainloop()


print("\n---Program Ending---\n")

#------------------------------------------------------------------------------------------------------------------Program Below
