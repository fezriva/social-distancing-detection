# base path to YOLO directory
MODEL_PATH = "yolo-coco"

# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = False

#Frames per seconds
FRAMES_X_SEC = 25

# define the error (in pixels) for which two objects or centroids are counted as equal
SAME_OBJECT = 15
SAME_CENTROID = 1
