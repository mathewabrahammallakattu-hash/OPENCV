import cv2
import numpy as np


image = cv2.imread('va.jpg')
#cv2.imshow('Image', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Greyscale Image', greyscale_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

sobel_x = cv2.Sobel(greyscale_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_x_abs = np.absolute(sobel_x)
sobel_x_8u = np.uint8(sobel_x_abs)

sobel_y = cv2.Sobel(greyscale_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_y_abs = np.absolute(sobel_y)
sobel_y_8u = np.uint8(sobel_y_abs)


gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
normalized_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
normalized_magnitude = np.uint8(normalized_magnitude)

cv2.imshow('Original Grayscale', greyscale_image)
cv2.imshow('Sobel X (Vertical Edges)', sobel_x_8u)
cv2.imshow('Sobel Y (Horizontal Edges)', sobel_y_8u)
cv2.imshow('Final Gradient Magnitude', normalized_magnitude)
cv2.imwrite('edges.png', normalized_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()


