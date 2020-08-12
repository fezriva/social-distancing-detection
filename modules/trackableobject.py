from collections import OrderedDict

class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		self.active = True

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False

		# initialise a boolean and a frame counter used to know
		# if and for how many seconds
		# the object has violated the social distancing rule
		self.violated = False
		self.violationTimes = OrderedDict()
