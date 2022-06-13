# from scipy.ndimage import rotate
from cv2 import rotate
import numpy as np
import cv2 
from math import atan2, pi, sqrt, cos, sin


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
    # [visualization1]


def getOrientation(pts, img):
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

    # orientation in radians
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    # [visualization]
    # print((-int(np.rad2deg(angle)) - 90))

    # Label with the rotation angle
    label = "  Rotation Angle: " + \
        str(-int(np.rad2deg(angle)) - 90) + " degrees"
    textbox = cv2.rectangle(
        img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    cv2.putText(img, label, (cntr[0], cntr[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return (-int(np.rad2deg(angle)) - 90)


def rotateImage(img, angle):
    # Rotate an Image
    rotated = rotate(img, angle)
    return rotated
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


file_path = "MessedUp_Notes_DataSet\\MessedUp_Resized_050_front_old_1.jpg"
image = cv2.imread(file_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# attempt to create white rectangle
ret, thresh1 = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# cv2.imshow("IMG", thresh1)
# cv2.waitKey(0)

# contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(
    thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

for i, c in enumerate(contours):

    # Calculate the area of each contour
    area = cv2.contourArea(c)

    # Ignore contours that are too small or too large
    if area < 3700:
        continue
    print("eish")

    # Draw each contour only for visualisation purposes
    cv2.drawContours(image, contours, i, (0, 0, 255), 2)

    # Find the orientation of each shape
    angle = getOrientation(c, image)
    print(angle)

cv2.imshow('Changed Pic', image)
cv2.waitKey(0)

rotated = rotateImage(image, -1*(90 + angle))

cv2.imshow('Output Image', rotated)
cv2.waitKey(0)
# cv2.destroyAllWindows()

# (x, y) = (512, 1024)

# resizedImage = cv2.resize(rotated, (y, x)) # note order

# cv2.imshow('Output Image', resizedImage)
# cv2.waitKey(0)
