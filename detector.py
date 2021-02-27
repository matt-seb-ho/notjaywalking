import cv2
import os
import re
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from collections import OrderedDict
import copy
import time

car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'cars.xml')

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

col_frames = os.listdir('frames')
col_frames.sort(key = lambda f: int(re.sub('\D', '', f)))
col_images=[]

ct = CentroidTracker()

for im in col_frames:
	img = cv2.imread('frames/' + im)
	col_images.append(img)

# load class names and random colors
classes = open('yolo-coco/coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
vehicle_classes = ['bicycle', 'car', 'motorbike', 'aeroplane', 'train', 'truck']
def is_vehicle(classname):
	return (classname in vehicle_classes)

# load network
yolopath = 'yolo-coco/'
net = cv2.dnn.readNetFromDarknet(yolopath + 'yolov3.cfg', yolopath + 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# determine the output layer
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def ydetect_motion_frames(frames, direction, frame_nums, showBoxes, conf):
	f, axarr = plt.subplots(len(frame_nums), 1)
	

	idirection = 1
	if direction == "left":
		idirection = -1
		
	motion = False

	for i in range(len(frame_nums)):
		frame = frames[frame_nums[i]]
		H, W = frame.shape[:2]
		#cars = car_cascade.detectMultiScale(frame, 1.05, 2)

		# construct blob from image
		blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
		r = blob[0, 0, :, :]

		net.setInput(blob)
		t0 = time.time()
		outputs = net.forward(ln)
		t = time.time()
		print('time = ', t - t0)

		
		boxes = []
		confidences = []
		#classIDs = []
		for output in outputs:
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				if is_vehicle(classes[classID]) and confidence > 0.5:
					# YOLO returns centerX, centerY, width, height
					box = detection[:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					box = [x, y, int(width), int(height)]
					#cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)
					#print("box: ", box)
					boxes.append(box)
					confidences.append(float(confidence))
				
		 
		rects = []
		indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
		if(len(indices) > 0):
			for j in indices.flatten():
				(x, y) = (boxes[j][0], boxes[j][1])
				(w, h) = (boxes[j][2], boxes[j][3])
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
				rects.append([x, y, x + w, y + h])

		print("len rects: ", len(rects))
		'''
		rects = []
		for (x, y, w, h) in cars:
			rect = (x, y, x + w, y + h)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
			rects.append(rect)
		#print(rects)
		'''

		objects = ct.update(rects)

		if i != 0:
			for dx in ct.deltaX:
				if idirection * dx > 0:
					print(direction + "wards motion")
					motion = True
					break
		
		#print(len(objects))
		for (objectID, centroid) in objects.items():
			print("objectID: ", objectID)
			cv2.putText(frame, "ID: " + str(objectID) , (centroid[0] - 10, centroid[1] - 10), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			#cv2.circle(frame
		#cv2.imshow("Frame: " + str(frames[i]), frame)
		axarr[i].imshow(frame)
		axarr[i].set_title('frame: ' + str(frame_nums[i]))

	plt.imshow(frame)
	f.tight_layout()
	if showBoxes:
		plt.show()

	return motion

def cdetect_motion_frames(frames, direction, frame_nums, showBoxes):
	f, axarr = plt.subplots(len(frame_nums), 1)
	idirection = 1
	if direction == "left":
		idirection = -1
		
	motion = False

	for i in range(len(frame_nums)):
		frame = frames[frame_nums[i]]
		cars = car_cascade.detectMultiScale(frame, 1.05, 2)
		rects = []
		for (x, y, w, h) in cars:
			rect = (x, y, x + w, y + h)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
			rects.append(rect)
		#print(rects)
		objects = ct.update(rects)
		if i != 0:
			for dx in ct.deltaX:
				if idirection * dx > 0:
					print(direction + "wards motion")
					motion = True
					break
		
		#print(len(objects))
		for (objectID, centroid) in objects.items():
			cv2.putText(frame, "ID: " + str(objectID) , (centroid[0] - 10, centroid[1] - 10), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			#cv2.circle(frame
		#cv2.imshow("Frame: " + str(frames[i]), frame)
		axarr[i].imshow(frame)
		axarr[i].set_title('frame: ' + str(frame_nums[i]))
	plt.imshow(frame)
	f.tight_layout()
	if showBoxes:
		plt.show()

	return motion

#detect_motion(col_images, "left", frame_nums, 0)

def detect_motion_video(filename, direction, frame_nums, showBoxes, conf, detector):
	col_images = []

	vidcap = cv2.VideoCapture(filename)
	success, frame = vidcap.read()
	count = 0
	while success:
		col_images.append(frame)
		success, frame = vidcap.read()

	'''
	plt.imshow(col_images[13])
	plt.show()
	'''
	if(detector == "cascades"):
		cdetect_motion_frames(col_images, direction, frame_nums, showBoxes)	
	elif(detector == "yolo"):
		ydetect_motion_frames(col_images, direction, frame_nums, showBoxes, conf)	
	else:
		print("not a valid model")
	
frame_nums = [100, 105]


motion_between_frames = detect_motion_video("cars.mp4", "right", frame_nums, 1, 0.5, "yolo")

if motion_between_frames:
	print("motion")
else:
	print("no motion")
#key = cv2.waitKey(1)
'''
for i in range(len(frames)):
	frame = col_images[frames[i]]
	cars = car_cascade.detectMultiScale(frame, 1.05, 2)
	for (x, y, w, h) in cars:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	axarr[i].imshow(frame)
	axarr[i].set_title('frame: ' + str(frames[i]))
'''

'''
frame = col_images[frame1]
cars = car_cascade.detectMultiScale(frame, 1.05, 2)
for (x, y, w, h) in cars:
	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
'''

#cv2.imshow("Result", frame)
#cv2.waitKey(1)

