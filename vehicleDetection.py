import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

#get file names of the frames
col_frames = os.listdir('frames')

#sort file names
col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

col_images=[]

for im in col_frames:
	# read the frames
	img = cv2.imread('frames/'+im)
	col_images.append(img)

i = 13
for frame in [i, i + 1]:
	plt.imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
	plt.title("frame: " + str(frame))
	plt.show()

