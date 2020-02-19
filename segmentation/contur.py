######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description:
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
# from skimage.color import rgb2gray
import cv2
import numpy as np
import tensorflow as tf
import sys
from datetime import datetime
import time
# import imutils



# def is_contour_bad(c):
# 	peri = cv2.arcLength(c, True)
# 	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

# 	# the contour is 'bad' if it is not a rectangle
# 	return not len(approx) == 4

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
print("Инициализация утилит начала")
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
print("Инициализация утилит прошла")
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'img'

# Grab path to current working directory
CWD_PATH = ''

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','object-detection.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,'img')
print("пути есть")
# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
print("Загрузка архитектуры сети начата")
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print("Загрузка архитектуры сети закончана линковка")
# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
print("Загрузка архитектуры сети закончана")
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

start_time = datetime.now()
PATH_TO_SAVE='output'
with os.scandir(PATH_TO_IMAGE) as entries:
    for entry in entries:
        image = cv2.imread(os.path.join(PATH_TO_IMAGE,entry.name))
        start_time = datetime.now()
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        print(datetime.now() - start_time)
        scores=np.squeeze(scores)
        boxes=np.squeeze(boxes)
        classes=np.squeeze(classes)
        max_boxes_to_draw = boxes.shape[0]
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]

        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > 0.25:
                if classes[i]==2:
                    box = tuple(boxes[i].tolist())
                    ymin, xmin, ymax, xmax = box
                    start_point = (int(xmin*height), int(ymin*width))
                    end_point = (int(xmax*height), int(ymax*width))
                    color = (255, 0, 0)
                    image1=image[int(ymin*width):int(ymax*width),int(xmin*height):int(xmax*height)]
                    savename=str(i)+entry.name
                    cv2.imwrite(os.path.join(PATH_TO_SAVE,"class1","cropped",savename), image1)
                    img_grey = cv2.cvtColor(image[int(ymin*width):int(ymax*width),int(xmin*height):int(xmax*height)], cv2.COLOR_BGR2GRAY)
                    thresh  = cv2.adaptiveThreshold(img_grey,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)
                    cv2.imwrite(os.path.join(PATH_TO_SAVE,"class1","thresh",savename), thresh)
                    conturs,hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in conturs:
                        peri = cv2.arcLength(cnt, True)
                        if peri>100 and peri<300:
                            for c in cnt:
                                c[0][0]=c[0][0]+int(xmin*height)
                                c[0][1]=c[0][1]+int(ymin*height)
                            cv2.drawContours(image, [cnt], -1, (255, 0, 0),2, cv2.LINE_AA)
            if scores is None or scores[i] > 0.25:
                if classes[i]==1:
                    box = tuple(boxes[i].tolist())
                    ymin, xmin, ymax, xmax = box
                    start_point = (int(xmin*height), int(ymin*width))
                    end_point = (int(xmax*height), int(ymax*width))
                    color = (255, 0, 0)
                    savename=str(i)+entry.name
                    image1=image[int(ymin*width):int(ymax*width),int(xmin*height):int(xmax*height)]
                    cv2.imwrite(os.path.join(PATH_TO_SAVE,"class2","cropped",savename), image1)
                    img_grey = cv2.cvtColor(image[int(ymin*width):int(ymax*width),int(xmin*height):int(xmax*height)], cv2.COLOR_BGR2GRAY)
                    thresh  = cv2.adaptiveThreshold(img_grey,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)
                    cv2.imwrite(os.path.join(PATH_TO_SAVE,"class2","thresh",savename), thresh)
                    conturs,hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in conturs:
                        peri = cv2.arcLength(cnt, True)
                        if peri>100 and peri<200:
                            for c in cnt:
                                c[0][0]=c[0][0]+int(xmin*height)
                                c[0][1]=c[0][1]+int(ymin*height)
                            cv2.drawContours(image, [cnt], -1, (0, 255, 0),2, cv2.LINE_AA)
        vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.20)
        cv2.imwrite(os.path.join(PATH_TO_SAVE,entry.name), image)