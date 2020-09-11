# import the necessary packages
from modules import yolo_coco_config as config
from modules.centroidtracker import CentroidTracker
from modules.trackableobject import TrackableObject
from modules.detection import detect_people
from modules.transform import four_point_transform
from imutils.video import VideoStream
from imutils.video import FPS
from scipy.spatial import distance as dist
from collections import OrderedDict
import modules.functions as fun
import numpy as np
import argparse
import imutils
import math
import time
import dlib
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-t", "--top_view", type=int, default=1,
	help="has the original video a view from the top (birdview)? (1 if yes, 0 if no)")
ap.add_argument("-b", "--birdview", type=str, default="",
	help="path to birdeye view image")
ap.add_argument("-c", "--csv", type=str, default="",
	help="# csv dataset name")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
ap.add_argument("-s", "--skip-frames", type=int, default = (config.FRAMES_X_SEC * 5),
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=70)
trackers = []
trackableObjects = {}
activityCheck = []
inactiveIDs = []

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalPeople = 0
totalViolations = 0

# grab the first frame and set the BirdView image
frame = vs.read()
image = warped = frame
(H, W) = (H_frame, W_frame) = (0, 0)

if args["top_view"] == 0:
	frame = frame[1] if args.get("input", False) else frame
	frame = imutils.resize(frame, width=900)
	image = cv2.imread(args["birdview"])
	clone = image.copy()

	(H_frame, W_frame) = frame.shape[:2]
	(H, W) = image.shape[:2]

	# initialize the list of reference points for the perspective warp
	pts = fun.choosePoints(frame, 4, "Choose 4 strategically placed anchor points (ex: the base of a statue)")
	anchor_pts = fun.choosePoints(image, 4, "Point out where the points you choose before are in this image")

	# apply the four point tranform to obtain a "birds eye view" of
	# the image
	Transform_Matrix = four_point_transform(pts, anchor_pts)

	# choose the min_distance between two pedestrians
	min_distance_pts = fun.choosePoints(image, 2, "Choose 2 points that will determine the minimum safe distance")
else:
	min_distance_pts = fun.choosePoints(frame, 2, "Choose 2 points that will determine the minimum safe distance")

pt_1 = min_distance_pts[0]
pt_2 = min_distance_pts[1]
min_distance = math.sqrt((pt_1[0] - pt_2[0])**2 + (pt_1[1] - pt_2[1])**2)

# start the frames per second throughput estimator
fps = FPS().start()

# create the csv file and initialize it
if args["csv"] is not None:
	csv = open(args["csv"], "w")
	print("ID", "frame", "coordX", "coordY", "trackedFor", "violationsNumber", "avgViolationTime", "maxViolationTime", sep=",", file=csv)


# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if args["input"] is not None and frame is None:
		break

	# initialize the set of indexes that violate the minimum social
	# distance
	violate = set()

	# resize the frame to have a maximum width of 900 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width=900)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	if (H_frame, W_frame) == (0,0):
		(H_frame, W_frame) = frame.shape[:2]

	if args["top_view"] == 0:
		image = clone.copy()
		warped = cv2.warpPerspective(frame, Transform_Matrix, (W, H))


	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["display"] == 1:
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

	# initialize  our list of bounding box rectangles
	rects = []

	# loop over the trackers
	for tracker in trackers:
		# update the tracker and grab the updated position
		tracker.update(rgb)
		pos = tracker.get_position()

		# unpack the position object
		startX = int(pos.left())
		startY = int(pos.top())
		endX = int(pos.right())
		endY = int(pos.bottom())

		# add the bounding box coordinates to the rectangles list
		rects.append((startX, startY, endX, endY))

	# check to see if we should look for new detections
	if totalFrames % args["skip_frames"] == 0:
		# detect people
		results = detect_people(frame, net, ln,
			personIdx=LABELS.index("person"))

		for r in results:

			startX = r[1][0]
			startY = r[1][1]
			endX = r[1][2]
			endY = r[1][3]

			r_cX = int((startX + endX) / 2.0)
			r_cY = int((startY + endY) / 2.0)
			r_c  = (r_cX, r_cY)

			new_tracker = True

			# check to see if the person detected is already tracked
			for tracker in trackers:

				pos = tracker.get_position()

				# unpack the position object
				t_startX = int(pos.left())
				t_startY = int(pos.top())
				t_endX = int(pos.right())
				t_endY = int(pos.bottom())

				t_cX = int((t_startX + t_endX) / 2.0)
				t_cY = int((t_startY + t_endY) / 2.0)
				t_c  = (t_cX, t_cY)

				# if centroids match no new tracker
				if fun.centroids_match(r_c, t_c, config.SAME_OBJECT):
					new_tracker = False
					break

			if new_tracker:

				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)
	objects_top = OrderedDict()
	violation_pairs = []

	if args["top_view"] == 0:
		#convert objects centroid into birdeyeview objects_top
		for (objectID, centroid) in objects.items():
			# calculate centroid coordinates from the top
			tf_centroid = fun.calculate_warped_coordinates(centroid, Transform_Matrix)
			objects_top[objectID] = tf_centroid
	else:
		objects_top = objects

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	if len(objects_top) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([centroid for (objectID, centroid) in objects_top.items()])

		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
				if D[i, j] < min_distance and (i or j) not in inactiveIDs:
					# update our violation set with the indexes of
					# the centroid pairs
					violate.add(i)
					violate.add(j)
					violation_pairs.append((i,j))

	# loop over the tracked objects
	for (objectID, centroid) in objects_top.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can count it
		else:

			# every second a movement is saved
			# if an average person walks 4 to 5 km/h we know that he will have moved at least 1 meter every second
			# centroid is saved and a violation check runs
			if totalFrames % config.FRAMES_X_SEC == 0:
				# if the object is at the edge of the image and it is not moving, it is marked as inactive
				originalCentroid = objects[objectID]
				if fun.at_the_edge(originalCentroid, W_frame, H_frame):
					if fun.centroids_match(centroid, to.centroids[len(to.centroids)-1], config.SAME_CENTROID):
						to.active = False
						inactiveIDs.append(objectID)
				# store the centroid in the list
				to.centroids.append(centroid)
				to.trackedFor += 1

				if objectID in violate:
					# add object to permanent violation set
					if not to.violated:
						totalViolations += 1
						to.violated = True

					for (i,j) in violation_pairs:
						if objectID == i:
							if j in to.violationTimes:
								to.violationTimes[j] += 1
							else:
								to.violationTimes[j] = 1

						elif objectID == j:
							if i in to.violationTimes:
								to.violationTimes[i] += 1
							else:
								to.violationTimes[i] = 1


			# check to see if the object has been counted or not
			if not to.counted:
				# if object not counted, add it to the number
				totalPeople += 1
				to.counted = True

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# write data to csv
		if to.active and totalFrames % config.FRAMES_X_SEC == 0:
			(coordX, coordY) = to.centroids[len(to.centroids)-1]

			sumVT = avgVT = maxVT = violationsNumber = 0

			if to.violated:
				# calculate avgViolationTime and maxViolationTime
				for (ID, time) in to.violationTimes.items():
					sumVT += time
					maxVT = max(maxVT, time)
					violationsNumber += 1

				avgVT = sumVT/violationsNumber

			print(to.objectID, totalFrames, coordX, coordY, to.trackedFor, violationsNumber, avgVT, maxVT, sep=",", file=csv)

		if to.active:
			color = (0, 255, 0)
			# if the index pair exists within the violation set, then
			# update the color
			if objectID in violate:
				color = (0, 0, 255)
			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)

			if args["top_view"] == 0:
				# on the image
				cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
				cv2.circle(image, (centroid[0], centroid[1]), 3, color, -1)
				# on the warped frame
				cv2.putText(warped, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
				cv2.circle(warped, (centroid[0], centroid[1]), 3, color, -1)

			# on the original frame
			orig_centroid = objects[objectID]
			cv2.putText(frame, text, (orig_centroid[0] - 10, orig_centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
			cv2.circle(frame, (orig_centroid[0], orig_centroid[1]), 3, color, -1)

	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("People", totalPeople),
		("Violations", totalViolations)
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H_frame - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		if args["top_view"] == 0:
			cv2.putText(image, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			cv2.putText(warped, text, (10, H_frame - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Original", frame)
	if args["top_view"] == 0:
		cv2.imshow("Warped", warped)
		cv2.imshow("BirdView", image)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
with open("dataset/appendix.csv", "w") as appendix:
	print("Time", "FPS", "frameW", "frameH", "totalPeople", "totalViolations", sep=",", file=appendix)
	print("{:.2f}".format(fps.elapsed()), "{:.2f}".format(fps.fps()), W, H, totalPeople, totalViolations, sep=",", file=appendix)
csv.close()

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()
