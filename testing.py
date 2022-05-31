from scipy import ndimage
import cv2

imgName = "Notes_DataSet\\10back_large.jpg"

image = cv2.imread(imgName)
cv2.imshow(imgName, image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#rotation angle in degree
rotated = ndimage.rotate(image, 15)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()