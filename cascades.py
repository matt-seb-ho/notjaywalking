import cv2
import os
import re
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'cars.xml')


col_frames = os.listdir('frames')
col_frames.sort(key = lambda f: int(re.sub('\D', '', f)))
col_images=[]
for im in col_frames:
	img = cv2.imread('frames/' + im)
	col_images.append(img)

frames = [13, 17]
f, axarr = plt.subplots(len(frames), 1)

for i in range(len(frames)):
	frame = col_images[frames[i]]
	cars = car_cascade.detectMultiScale(frame, 1.05, 2)
	for (x, y, w, h) in cars:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	axarr[i].imshow(frame)

'''
frame = col_images[frame1]
cars = car_cascade.detectMultiScale(frame, 1.05, 2)
for (x, y, w, h) in cars:
	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
'''

plt.imshow(frame)
plt.show()
#cv2.imshow("Result", frame)
#cv2.waitKey(1)

