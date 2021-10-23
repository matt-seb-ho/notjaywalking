import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import copy
import time
from os.path import isfile, join
from centroid_tracker import CentroidTracker

car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'cars.xml')

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

def ydetect_motion_frames(frames, direction, frame_nums, showBoxes, conf, nms_thresh, stop_early):
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
				if is_vehicle(classes[classID]) and confidence > conf:
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
		indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, nms_thresh)
		if(len(indices) > 0):
			for j in indices.flatten():
				(x, y) = (boxes[j][0], boxes[j][1])
				(w, h) = (boxes[j][2], boxes[j][3])
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
				rects.append([x, y, x + w, y + h])

		# print("len rects: ", len(rects))
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
					if stop_early:
						return True
					break
		
		#print(len(objects))
		for (objectID, centroid) in objects.items():
			# print("objectID: ", objectID)
			cv2.putText(frame, "ID: " + str(objectID) , (centroid[0] - 10, centroid[1] - 10), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			#cv2.circle(frame
		if showBoxes:
			cv2.imshow("Frame: " + str(frames[i]), frame)
			cv2.waitKey(0)
		axarr[i].imshow(frame)
		axarr[i].set_title('frame: ' + str(frame_nums[i]))
	plt.imshow(frame)
	f.tight_layout()
	if showBoxes:
		cv2.destroyAllWindows()
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
		#cv2.waitKey(0)
		axarr[i].imshow(frame)
		axarr[i].set_title('frame: ' + str(frame_nums[i]))
	plt.imshow(frame)
	f.tight_layout()
	if showBoxes:
		plt.show()

	return motion

#detect_motion(col_images, "left", frame_nums, 0)

# ydetect signature: (frames, direction, frame_nums, showBoxes, conf, nms_thresh, stop_early)

def detect_motion_video(filename, direction, frame_nums, showBoxes, conf, nms_thresh, detector, stop_early):
	col_images = []

	vidcap = cv2.VideoCapture(filename)
	success, frame = vidcap.read()
	count = 0
	while success:
		col_images.append(frame)
		success, frame = vidcap.read()

	# print("number of frames: ", len(col_images))
	'''
	plt.imshow(col_images[13])
	plt.show()
	'''
	if(detector == "cascades"):
		cdetect_motion_frames(col_images, direction, frame_nums, showBoxes)	
	elif(detector == "yolo"):
		ydetect_motion_frames(col_images, direction, frame_nums, showBoxes, conf, nms_thresh, stop_early)	
	else:
		print("not a valid model")

def checkForMovingCars(filename, direction, framerate = 2, conf = 0.5, nms_thresh = 0.3):
	col_images = []

	vidcap = cv2.VideoCapture(filename)
	success, frame = vidcap.read()
	count = 0
	while success:
		col_images.append(frame)
		success, frame = vidcap.read()

	return ydetect_motion_frames(col_images, direction, range(0, len(col_images) - 1, framerate), 0, conf, nms_thresh, 1)	
	
	
frame_nums = [100, 105, 107, 109]

# detect_motion_video takes (filename, direction, frame_numbers, showBoxes, confidence threshold, NMS threshold, which detector, stop_early)
motion_between_frames = detect_motion_video("cars.mp4", "right", frame_nums, 1, 0.5, 0.3, "yolo", 0)
# checkForMovingCars("cars.mp4", "right")
