
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
# for otnositelnih putey
CWD_PATH = ''
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'object-detection.pbtxt')
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
PATH_TO_SAVE = 'output1'

#init serv
METHOD_RECEPTION = int(input('Reception method(0 - independ client, 1 - webcam): '))
METHOD_PROCESSING = int(input('Processing method(0 - SQUARE, 1 - SEOQL): '))
METHOD_SAVING = int(input('SAVING METHOD(0 - on disk, 1 - sender): '))

HOST = ''
PORT = 9090
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
sock.bind((HOST, PORT))
print('Socket bind complete')
sock.listen(20)
print('Socket now listening')
conn, addr = sock.accept()
print('KNOCK KNOCK')
data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
NumbFrame = 0
# cycle for recive and processing img
while True:
    start_time = datetime.now()
    print('start')
    while len(data) < payload_size:
        print("Recv: {}".format(len(data)))
        data += conn.recv(4096)
    print("Done Recv: {}".format(len(data)))
    print('time for recive SIZE DATA', datetime.now() - start_time)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(1024000)
    print('time for recive DATA', datetime.now() - start_time)

    ARROW_TIME = datetime.now()
    frame_data = data[:msg_size]
    print('frame_data')
    data = data[msg_size:]
    print('data load')
    print('ARROW_TIME: ', ARROW_TIME)

    start_pickle_load_time = datetime.now()
    # image = np.frombuffer(frame_data, np.uint8)
    image = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    print('time PIKLE LOAD IMG: ', datetime.now() - start_pickle_load_time)
    print('Image loads')
    print('time for recive img', datetime.now() - start_time)

# if you use webcam(black and white - GRAYSKALE)
    if(METHOD_RECEPTION == 1):
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        print('DECODE SUCESFULL')

# detect object
    image_expanded = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

# drawer of square
    if(METHOD_PROCESSING == 0):
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=0.50)

#definer center if cath
    elif (METHOD_PROCESSING == 1):
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes)
        max_boxes_to_draw = boxes.shape[0]
        height = image.shape[0]
        width = image.shape[1]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > 0.5:
                if classes[i] == 2:
                    box = tuple(boxes[i].tolist())
                    ymin, xmin, ymax, xmax = box
                    center_coordinates = (int((xmax * height + xmin * height) / 2), int((ymax * width + ymin * width) / 2))
                    print(center_coordinates)
    else:
        print('check METHOD_PROCESSING')
        conn.close()

# saver or sender
#   cv2.imshow('ImageWindow', image)
    if(METHOD_SAVING == 0):
        cv2.imwrite('{}/{}.png'.format(PATH_TO_SAVE, [datetime.now(), NumbFrame]), image)
    elif (METHOD_SAVING == 1):
        start_send_time = datetime.now()
        conn.sendall(center_coordinates)
        print('SEND_TIME: ', datetime.now() - start_send_time)
    else:
        print('check METHOD_SAVING')
        conn.close()
    NumbFrame += 1
    print('Time for one cycle: ', datetime.now() - start_time)
conn.close()
