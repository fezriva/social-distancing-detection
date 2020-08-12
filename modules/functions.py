from modules.trackableobject import TrackableObject
import cv2
import numpy as np

# algorithm to choose four points from a predefined image
def choosePoints(image, num_points):

    refPt = []

    def mouse_event(event, x, y, flags, param):
    	nonlocal refPt
    	if event == cv2.EVENT_LBUTTONDOWN:
    		refPt.append((x,y))
    		cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
    		cv2.imshow("Choose_Points", image)

    clone = image.copy()
    cv2.namedWindow("Choose_Points")
    cv2.setMouseCallback("Choose_Points", mouse_event)

    while True:
    	# display the image and wait for a keypress
    	cv2.imshow("Choose_Points", image)
    	key = cv2.waitKey(1) & 0xFF
    	# if the 'r' key is pressed, reset the cropping region
    	if key == ord("r"):
    		image = clone.copy()
    		refPt = []
    	# if the 'c' key is pressed, break from the loop
    	elif key == ord("c"):
    		if len(refPt) == num_points:
    			cv2.destroyWindow("Choose_Points")
    			break
    		else:
    			image = clone.copy()
    			refPt = []

    return np.array(refPt, dtype = "float32")

# check if two centroid match
# https://stackoverflow.com/questions/481144/equation-for-testing-if-a-point-is-inside-a-circle#:~:text=Find%20the%20distance%20between%20the,the%20circumference%20of%20the%20circle.
def centroids_match(c1, c2, error):
    dx = abs(c1[0] - c2[0])
    dy = abs(c1[1] - c2[1])
    R = error

    if dx > R or dy > R:
        return False

    if dx + dy <= R:
        return True

    if dx^2 + dy^2 <= R^2:
        return True
    else:
        return False

# change centroids values after warp perspective application
def calculate_warped_coordinates(centroid, M):
    tf_x = (M[0][0]*centroid[0] + M[0][1]*centroid[1] + M[0][2]) / ((M[2][0]*centroid[0] + M[2][1]*centroid[1] + M[2][2]))
    tf_y = (M[1][0]*centroid[0] + M[1][1]*centroid[1] + M[1][2]) / ((M[2][0]*centroid[0] + M[2][1]*centroid[1] + M[2][2]))
    return (int(tf_x), int(tf_y)) # after transformation

# check if a centroid is close to the edge of the image
# I use a 5px safety margin
def at_the_edge(centroid, W, H):
    if (not 5 < centroid[0] < W - 5) or (not 5 < centroid[1] < H - 5):
        return True
    else:
        return False
