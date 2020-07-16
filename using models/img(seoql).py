
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from datetime import datetime
import time
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'img'

CWD_PATH = ''

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'training','object-detection.pbtxt')

PATH_TO_IMAGE = os.path.join(CWD_PATH,'img')

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

start_time = datetime.now()
PATH_TO_SAVE='output1'
with os.scandir(PATH_TO_IMAGE) as entries:
    for entry in entries:
        print(entry.name)
        image = cv2.imread(os.path.join(PATH_TO_IMAGE,entry.name))
        start_time = datetime.now()
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        print(datetime.now() - start_time)
        print(np.squeeze(scores))
        scores=np.squeeze(scores)
        boxes=np.squeeze(boxes)
        classes=np.squeeze(classes)
        max_boxes_to_draw = boxes.shape[0]
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]

        for i in range(min(max_boxes_to_draw, boxes.shape[0])):

            if scores is None or scores[i] > 0.5:
                if classes[i]==2:
                    box = tuple(boxes[i].tolist())
                    ymin, xmin, ymax, xmax = box
                    start_point = (int(xmin*height), int(ymin*width))
                    end_point = (int(xmax*height), int(ymax*width))
                    color = (255, 0, 0)
                    center_coordinates=(int((xmax*height+xmin*height)/2),int((ymax*width+ymin*width)/2))
                    print ( center_coordinates)
                #image = cv2.rectangle(image, start_point, end_point, color, 3)
                    image = cv2.circle(image, center_coordinates, 15, color, 3)
        cv2.imwrite(os.path.join(PATH_TO_SAVE,entry.name), image)
