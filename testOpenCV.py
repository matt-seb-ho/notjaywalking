import cv2

print("OpenCV version")
print(cv2.__version__)

img = cv2.imread("clouds.jpg")
cv2.imshow("yay it worked", img)

cv2.waitKey(0)
