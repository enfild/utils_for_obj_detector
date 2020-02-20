import cv2
import io
import socket
import struct
import time
import os
import sys
import pickle
import zlib
from datetime import datetime

CWD_PATH = ''
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 9090))
connection = client_socket.makefile('wb')

IMAGE_NAME = 'img'

PATH_TO_IMAGE = os.path.join(CWD_PATH, 'img')

while True:
    img_counter = 0
    with os.scandir(PATH_TO_IMAGE) as entries:
        for entry in entries:
            print(entry.name)
            image = cv2.imread(os.path.join(PATH_TO_IMAGE, entry.name))
            start_time = datetime.now()
#    data = zlib.compress(pickle.dumps(frame, 0))
            data = pickle.dumps(image, 0)
            size = len(data)
            print("{}: {}".format(img_counter, size))
            client_socket.sendall(struct.pack(">L", size) + data)
            img_counter += 1
connection.close()