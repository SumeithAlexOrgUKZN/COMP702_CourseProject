from scipy.ndimage import rotate # this is the function needed to rotate an image back!

import cv2

import numpy as np
from math import atan2, pi, sqrt, cos, sin

# returns angle of largest rectangle
def getSkewAngle(imageContours):
    angle = -1
    maxArea = -1
    bestArray = [[[]]]

    # i is index, c is the 3D array
    for i, c in enumerate(imageContours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)

        # Ignore contours that are too small
        if area < 3700:
            continue

        if (maxArea < area):
            maxArea = area
            bestArray = c

    # Find the orientation of each shape
    angle = getOrientation(bestArray)

    return angle
###

def findBestCropDimensions(contours):
    best_box = [-1, -1, -1, -1]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if best_box[0] < 0:
            best_box = [x, y, x+w, y+h]
        else:
            if x < best_box[0]:
                best_box[0] = x
            if y < best_box[1]:
                best_box[1] = y
            if x+w > best_box[2]:
                best_box[2] = x+w
            if y+h > best_box[3]:
                best_box[3] = y+h

    return best_box
###

def cropAnImage(image, dimensions):
    a, b, c, d = dimensions[0], dimensions[1], dimensions[2], dimensions[3]
    # OPTIONAL, slight narrowing
    value = 6
    a, b, c, d = a+value, b+value, c-value, d-value
    cropped = image[ b:d , a:c ]

    return cropped
###

def rotateImage(img, angle):
    # Rotate an Image
    rotated = rotate(img, angle)
    return rotated
###

def getImageContours(img):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours
###

'''
    This functions uses Principal Component Analysis to reliably 
    detect the orientation of an object.

    draw: Boolean --> add to place info on image
    If draw == True, the image will have the orientation
    information placed on it.
'''
def getOrientation(pts):
    # [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # orientation in radians
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    # [visualization]
    # print((-int(np.rad2deg(angle)) - 90))

    return (-(int(np.rad2deg(angle)) % 360) - 90) 
###

# img passed by reference
def drawOrientation(img, contours):
    for index, pts in enumerate(contours):
        area = cv2.contourArea(pts)

        # Ignore contours that are too small
        if area < 3700:
            continue
        # [pca]
        # Construct a buffer used by the pca analysis
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i, 0] = pts[i, 0, 0]
            data_pts[i, 1] = pts[i, 0, 1]

        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
        # Store the center of the object
        cntr = (int(mean[0, 0]), int(mean[0, 1]))
        # [pca]

        # [visualization]
        # Draw the principal components
        cv2.circle(img, cntr, 3, (255, 0, 255), 2)
        p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
                cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
                cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
        drawAxis(img, cntr, p1, (255, 255, 0), 1)
        drawAxis(img, cntr, p2, (0, 0, 255), 5)

        # Label with the rotation angle
        val = (-int(np.rad2deg(angle))- 90)
        if (val < 0):   val = val % -360
        else:           val = val % 360

        label = "  Rotation Angle: " + \
            str(val)+ " degrees"
        textbox = cv2.rectangle(
            img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
        cv2.putText(img, label, (cntr[0], cntr[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
###


# Draws the Orthogonal Vector onto an image --> Note that Img is passed as reference
def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    # [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) +
                      (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])),
             (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])),
             (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])),
             (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
###

# image passed by reference
def drawAllContours(image, imageContours):
    for i, c in enumerate(imageContours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)

        # Ignore contours that are too small
        if area < 3700:
            continue

        # Draw each contour, for visualisation purposes
        cv2.drawContours(image, imageContours, i, (0, 0, 255), 2)
###

def drawLargestContour(image, imageContours):
    maxArea = -1
    bestVar = -1

    # i is index, c is the 3D array
    for i, c in enumerate(imageContours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)

        # Ignore contours that are too small
        if area < 3700:
            continue

        if (maxArea < area):
            maxArea = area
            bestVar = i

    # Draw largest contour only, for visualisation purposes
    cv2.drawContours(image, imageContours, bestVar, (0, 0, 255), 2)
###

file_path = "MessedUp_Notes_DataSet\\MessedUp_Resized_050_front_old_1.jpg"
image = cv2.imread(file_path)

# works!
# cv2.imshow("Original", image)
# cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# works!
# cv2.imshow("Gray", gray)
# cv2.waitKey(0)

# attempt to create white rectangle, notice threshold values
ret, thresh1 = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# works!
# cv2.imshow("Thresh", thresh1)
# cv2.waitKey(0)

imageContours = getImageContours(thresh1)

# lets try each function now
#----------------------------------------------------------------------------------------------

#works!
angle = getSkewAngle(imageContours)

print(angle)

# works!
# notice rule for inverse rotation
rotatedImage = rotateImage(image, -1 * (90 + angle))

cv2.imshow("Rotated", rotatedImage)
cv2.waitKey(0)

# cropDimensions = findBestCropDimensions(imageContours)

# new angle == -90!
# gray = cv2.cvtColor(rotatedImage, cv2.COLOR_BGR2GRAY)
# # attempt to create white rectangle, notice threshold values
# ret, thresh1 = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
# imageContours = getImageContours(thresh1)
# angle2 = getSkewAngle(imageContours)

# print("New Angle", angle2)
# rotatedImage2 = rotateImage(rotatedImage, 1*(90 + angle2))

# cv2.imshow("Rotated", rotatedImage2)
# cv2.waitKey(0)

grayStraight = cv2.cvtColor(rotatedImage, cv2.COLOR_BGR2GRAY)
# attempt to create white rectangle, notice threshold values
ret, thresh2 = cv2.threshold(grayStraight, 1, 255, cv2.THRESH_BINARY)
imageContoursStraight = getImageContours(thresh2)

cropDimensions = findBestCropDimensions(imageContoursStraight)
print(cropDimensions)

# still works!
# cv2.imshow("Rotated", rotatedImage)
# cv2.waitKey(0)

croppedPic = cropAnImage(rotatedImage, cropDimensions)
# cv2.imshow("Cropped", croppedPic)
# cv2.waitKey(0)

# # works!
# demoPic = image
# drawAllContours(demoPic, imageContours)

# cv2.imshow("Demo1", demoPic)
# cv2.waitKey(0)

# # works!
# demoPic = image
# drawLargestContour(demoPic, imageContours)

# cv2.imshow("Demo1", demoPic)
# cv2.waitKey(0)

demoPic = image
drawAllContours(demoPic, imageContours)
drawOrientation(demoPic, imageContours)

cv2.imshow("Demo1", demoPic)
cv2.waitKey(0)