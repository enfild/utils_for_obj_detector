
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

from threading import Thread
from queue import Queue
from utils import label_map_util
from utils import visualization_utils as vis_util


#________________________INIT_____________________________
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

# Uncomment to using RTX 2060-2070. problems with memory allocation
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

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
#_____________________INIT SOCKET-SERV__________________________
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
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))

q = Queue()

#_____________________________DEFS__________________________________
# cycle for recive and processing img
def get_img():
    data = b""
    while True:
        print('start get_img')
        start_time = datetime.now()
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += conn.recv(2024000)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        image = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        image = np.frombuffer(frame_data, np.uint16)

    if you use webcam(black and white - GRAYSKALE)
        if(METHOD_RECEPTION == 1):
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        q.put(image)
        print('Time for recive: ', datetime.now() - start_time)
        PROC_TH = Thread(target = processing_IMG, args = (image))
        print('PROC TH init')
        PROC_TH.start()

def processing_IMG():
    NumbFrame = 0
    avg_time = 0
    print('start PROCCESSING')
    while True:
        if q.empty():
            # print('queue is empty')
            continue
        image1 = q.get()
# detect object
        print('QUEUEEUEUEUEUEEU')
        start_time1 = datetime.now()
        image_expanded = np.expand_dims(image1, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
# drawer of square
        if(METHOD_PROCESSING == 0):
            vis_util.visualize_boxes_and_labels_on_image_array(
                image1,
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
            height = image1.shape[0]
            width = image1.shape[1]
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
        if(METHOD_SAVING == 0):
            cv2.imwrite('{}/{}.png'.format(PATH_TO_SAVE, [datetime.now(), NumbFrame]), image1)
        elif (METHOD_SAVING == 1):
            conn.sendall(center_coordinates)
        else:
            print('check METHOD_SAVING')
            conn.close()
        NumbFrame += 1
        print('Time for one cycle: ', datetime.now() - start_time1)
        # avg_time = avg_time * 0.8 + 0.2 * float(datetime.now()-start_time1)
        # print(avg_time)


if __name__ == "__main__":
    print("start")
    GET_TH = Thread(target = get_img, args = ())
    GET_TH.start()
    print('GET TH started')
    PROC_TH = Thread(target = processing_IMG, args = ())
    PROC_TH.start()

