
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from datetime import datetime
import time

from utils import label_map_util
from utils import visualization_utils as vis_util
# os.environ['CUDA_VISIBLE_DEVICES']='1'
MODEL_NAME = 'inference_graph'
S_PATH = 'saved_model'
IMAGE_NAME = 'img'
# for otnositelnih petey
CWD_PATH = ''

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,S_PATH,'saved_model.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'data','object-detection.pbtxt')

PATH_TO_IMAGE = os.path.join(CWD_PATH,'img')

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Uncomment to using RTX 2060-2070. problems with memory allocation
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
# tf.device('/device:GPU:1')
# with tf.device('/gpu:0'):

# config = tf.ConfigProto()
# config.gpu_options.visible_device_list= '0'
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# sess = tf.Session(config=config)


# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

start_time = datetime.now()
PATH_TO_SAVE='output'
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
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=0.50)
        cv2.imwrite(os.path.join(PATH_TO_SAVE,entry.name), image)



