import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

#get file names of the frames
col_frames = os.listdir('frames')

#sort file names
col_frames.sort(key = lambda f: int(re.sub('\D', '', f)))

col_images=[]

for im in col_frames:
	# read the frames
	img = cv2.imread('frames/'+im)
	col_images.append(img)

frame1 = 13
frame2 = 14
#plt.figure()
f, axarr = plt.subplots(3, 1)

axarr[0].imshow(cv2.cvtColor(col_images[frame1], cv2.COLOR_BGR2RGB))
axarr[1].imshow(cv2.cvtColor(col_images[frame2], cv2.COLOR_BGR2RGB))

'''
for frame in [frame1, frame2]:
	axarr[frame - frame1].imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
	#plt.imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
	#axarr[frame - frame1].title("frame: " + str(frame))
	#plt.show()
'''


grayA = cv2.cvtColor(col_images[frame1], cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(col_images[frame2], cv2.COLOR_BGR2GRAY)

diff_img = cv2.absdiff(grayA, grayB)

ret, thresh = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations = 2)

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

dmy = col_images[frame2].copy()
valid_cntrs = []
for i, cntr in enumerate(contours):
	x, y, w, h = cv2.boundingRect(cntr)
	if cv2.contourArea(cntr) >= 25:
		valid_cntrs.append(cntr)
#		cv2.rectangle(dmy, (x, y), (x + w, y + h), color = (0, 255, 0), thickness = 1)


#cv2.drawContours(dmy, valid_cntrs, -1, (127, 200, 0), 2)

for cntr in valid_cntrs:
	x, y, w, h = cv2.boundingRect(cntr)
	cv2.rectangle(dmy, (x, y), (x + w, y + h), (255, 0, 0), 1)

axarr[2].imshow(dmy)
#plt.imshow(dilated, cmap = 'gray')
plt.show()

