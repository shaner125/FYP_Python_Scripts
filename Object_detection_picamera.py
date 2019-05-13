######## Picamera Object Detection Using Tensorflow Classifier #########
#
# Author: Evan Juras
# Date: 4/15/18
# Description: 
# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# when executing this script from the terminal.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import threading
import pyrebase
import time
from flask_opencv_streamer.streamer import Streamer
from keyClipWriter import KeyClipWriter
import imutils
import datetime

configdb = {
    "apiKey": "AIzaSyBNZNvGzSo0BWLiy0ykfzpjKaRNKr7hTSs",
    "authDomain": "smart-feeder-1d272.firebaseapp.com",
    "databaseURL": "https://smart-feeder-1d272.firebaseio.com",
    "storageBucket": "smart-feeder-1d272.appspot.com",
    "serviceAccount": "smart-feeder-1d272-firebase-adminsdk-z648w-281cedb936.json"
}

firebase = pyrebase.initialize_app(configdb)

db = firebase.database()
detectStatus = db.child("data")

# Set up camera constants
##IM_WIDTH = 1280
##IM_HEIGHT = 720
IM_WIDTH = 640
IM_HEIGHT = 480

#flask requirements
port = 3030
require_login = False
streamer = Streamer(port, require_login)

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX
pet_detection = 0
kcw = KeyClipWriter(bufSize=18)
consecFrames = 0
pet_detection = 0

def stream_handler(message):
    global detectFlag
    global livestreamflag
    if message["path"] == "/detect":
        detectFlag = message["data"]
    elif message["path"] == "/livestream":
        livestreamflag = message["data"]
        print("new livestream flag: "+str(livestreamflag))   

my_stream = detectStatus.stream(stream_handler)

def detector():
    
    print("Starting Pet Detection...")
    global consecFrames
    global pet_detection
    

    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.rotation = 270
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)
##        flaskimg = cv2.resize(frame, (400,290))

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})    

        # Draw the results of the detection (aka 'visulaize the results')
        if ((int(classes[0][0]) == 17) or (int(classes[0][0]) == 18)):

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.30)
            if not kcw.recording:
                consecFrames = 0
                print("Starting clip Recording...")
                timestamp = datetime.datetime.now()
                p = "{}/{}.mp4".format('output',
                        timestamp.strftime("%Y%m%d-%H%M%S"))
                kcw.start(p, cv2.VideoWriter_fourcc(*'mp4v'),1)
            pet_detection = pet_detection + 1
            cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # Stream frame to flask server for live stream
        streamer.update_frame(frame)
        if not streamer.is_streaming:
            streamer.start_streaming()

        # otherwise, no action has taken place in this frame, so
        # increment the number of consecutive frames that contain
        # no action
        
        consecFrames += 1

        # update the key frame clip buffer
        kcw.update(frame)

        # if we are recording and reached a threshold on consecutive
        # number of frames with no action, stop recording the clip
        if kcw.recording and consecFrames == 18:
            print("saving clip..")
            kcw.finish()
            time.sleep(5)
            kcw.upload()
            
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        if pet_detection > 30:
            print("Your pet has been detected!!")
            pet_detection = 0

         # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
        
        rawCapture.truncate(0)

    camera.close()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

t = threading.Thread(target=detector)
t.start()

