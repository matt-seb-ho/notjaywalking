import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

# tutorial resource followed: 
# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
class CentroidTracker():
	def __init__(self, maxDisappeared = 50):
		self.nextObjectID = 0
		self.objects = OrderedDict()
		#self.prevFrameObjects = OrderedDict()
		self.deltaX = []
		self.disappeared = OrderedDict()
		self.maxDisappeared = maxDisappeared
	def register(self, centroid):
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1
	def deregister(self, objectID):
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
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		#loop over each rectangle
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			'''
			print("startX: ", startX)
			print("startY: ", startY)
			'''
			#get centroid coords
			cX = int((startX + endX) * 0.5)
			cY = int((startY + endY) * 0.5)
			inputCentroids[i] = (cX, cY)
			#print("inputCentroids: ")
			#print(inputCentroids)

		#if not currently tracking, register all centroids
		if len(self.objects) == 0:
			for i in range(len(inputCentroids)):
				self.register(inputCentroids[i])

		else:
			# clear deltaX
			deltaX = []

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

				#access centroid via objects[id]
				self.deltaX.append(inputCentroids[col][0] - self.objects[objectID][0])
				
				#set object to its successor HERE
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

				# track unused centroids
				unusedRows = set(range(0, D.shape[0])).difference(usedRows)
				unusedCols = set(range(0, D.shape[1])).difference(usedCols)

				# check if centroids appeared or disappeared
				# print("D.shape[0]: ", D.shape[0], ", [1]: ", D.shape[1])
			if D.shape[0] >= D.shape[1]:
				# more centroids in previous frame, increment disappeared counts
				# deregister as necessary
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			else:
				# more ceontroids in current frame, register them
				for col in unusedCols:
					self.register(inputCentroids[col])
		return self.objects
