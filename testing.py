from scipy import ndimage
import cv2

# imgName = "Notes_DataSet\\050_front_old_1.jpg"

# image = cv2.imread(imgName)
# cv2.imshow(imgName, image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #rotation angle in degree
# rotated = ndimage.rotate(image, 15)
# cv2.imshow("Rotated", rotated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#---------------------------------------------------------------------------------------

# image = cv2.imread(imgName, 0)
# (x, y) = image.shape

# resizedImage = cv2.resize(image, (1024, 512), interpolation = cv2.INTER_AREA)
# cv2.imshow("Resized", resizedImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()