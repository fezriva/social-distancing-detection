# Social Distancing Detection from Web Streaming Cameras
Dissertation for my Brunel University Data Science and Analytics MSc

## Required

Python3

### Packages
- NumPy
- OpenCV
- dlib
- imutils

### Files
yolov3.weights have to be downloaded from the web due to the excessive file size
they must then be placed inside the yolo-coco folder

## Usage

### Arguments
--input: the path to the input video
--top_view: has the original video a view from the top (birdview)? (1 if yes, 0 if no) (default is 1)
--birdview: the path to the image which will determine the view from the top
--csv: the path to the output dataset
--output: the path to the output video
--display: set to 1 if you want to save the output, to 0 if you do not (default is 1)
--skip_frames: every how many frames do you want to detect new pedestrians (default is FRAMES_X_SEC (=25) times 5)
### Example
python3 track_distancing_birdview.py --input videos/Campo-fiori.mp4 --output output/campo-dei-fiori.avi --top_view 0 --birdview images/Campo_dei_fiori.PNG --csv dataset/campo_fiori.csv
