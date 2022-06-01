'''
Authors:    Sumeith Ishwanthlal (219006284)
            Alexander Goudemond (219030365)

Program name: functions_all.py

Goal: Holds most of the code used in the project

Summary:

'''

#----------------------------------------------------------------------------------------------------------------Packages Below

# os for deleting files
import tkinter as tk
from tkinter import filedialog, Toplevel

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
    print("Inside chooseExperimentMethod()")

    experimentWindow = Toplevel(window)
    experimentWindow.title("Choose further options below")
    # experimentWindow.geometry("500x500")

    experimentFrame = tk.Frame(experimentWindow)
    buttonFrameTop = tk.Frame(experimentWindow)
    buttonFrameMiddle = tk.Frame(experimentWindow)
    buttonFrameBottom = tk.Frame(experimentWindow)

    button1 = tk.Button(
        master = buttonFrameTop,
        text = "1",
        width = 40,
        height = 5, 
        bg = "silver",
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