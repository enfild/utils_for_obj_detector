
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from datetime import datetime
import time
import pickle
import struct
import socket


from utils import label_map_util
from utils import visualization_utils as vis_util
print("start")
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'img'
# for otnositelnih petey
CWD_PATH = ''
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'object-detection.pbtxt')
PATH_TO_IMAGE = os.path.join(CWD_PATH, 'img')
NUM_CLASSES = 2
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print('INIT DONE')
# Load the Tensorflow model into memory.
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
print('INIT TF DONE')
PATH_TO_SAVE=input("Where save? : ")
start_time = datetime.now()
#init serv
HOST=''
PORT=9090
sock= socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')
sock.bind((HOST,PORT))
print('Socket bind complete')
sock.listen(10)
print('Socket now listening')
conn,addr=sock.accept()
print('KNOCK KNOCK')
data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))

NumbFrame = 0
while True:
    while len(data) < payload_size:
        print("Recv: {}".format(len(data)))
        data += conn.recv(4096)
    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    print('frame_data')
    data = data[msg_size:]
    print('DATA LOADED')
    # image = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    print('Image loads')
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    print('DECODE SUCESFULL')
# detect object
    image_expanded = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    print(datetime.now() - start_time)
# drawer
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.20)
    cv2.imwrite(os.path.join(PATH_TO_SAVE, NumbFrame.jpg), image)
    NumbFrame += 1
# conn.close()


