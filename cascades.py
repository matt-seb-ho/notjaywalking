import cv2
import os
import re
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from collections import OrderedDict

car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'cars.xml')

class CentroidTracker():
	def __init__(self, max_disappeared = 50):
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.maxDisappeared = maxDisappeared
	def register(self, centroid):
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1
	def register(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]
	def update(self, rects):
		#check if no bounding boxes
		if len(rects) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			return self.objects

		#init array of input centroids for current frame
		inputCentroids = np.zeroes((len(rects), 2), dtype="int")

		#loop over each rectangle
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			#get centroid coords
			cX = int((startX + endX) * 0.5)
			cY = int((startY + endY) * 0.5)
			inputCentroids[i] = (cX, cY)

		#if not currently tracking, register all centroids
		if len(self.objects) == 0:
			for i in range(len(inputCentroids)):
				self.register(inputCentroids[i])
		else:
			# get objectIDs and their centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# get distance between each input and current centroid
			# (matching centroids from one frame to the next)
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			rows = D.min(axis = 1).argsort()
			cols = D.argmin(axis = 1)[rows]

			usedRows = set()
			usedCols = set()
			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue

				#found an input centroid that has the smallest distance to
				#an unmatched previous centroid
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

				# track unused centroids
				unusedRows = set(range(0, D.shape[0])).difference(usedRows)
				unusedCols = set(range(0, D.shape[1])).difference(usedCols)

				# check if centroids appeared or disappeared
				if D.shape[0] >= D.shape[1]:
					for row in unusedRows:
						objectID = objectIDs[row]
						self.disappeared[objectID] += 1
						
						if self.disappeared[objectID] > self.maxDisappeared:
							self.deregister(objectID)
				else:
					for col in unusedCols:
						self.register(inputCentroids[col])
			return self.objects

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
	axarr[i].set_title('frame: ' + str(frames[i]))

'''
frame = col_images[frame1]
cars = car_cascade.detectMultiScale(frame, 1.05, 2)
for (x, y, w, h) in cars:
	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
'''

plt.imshow(frame)
f.tight_layout()
plt.show()
#cv2.imshow("Result", frame)
#cv2.waitKey(1)

