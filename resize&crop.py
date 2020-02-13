
import os
import cv2
import sys
from datetime import datetime
import time
# import image
IMAGE_NAME = 'img'
# for otnositelnih petey
CWD_PATH = ''

PATH_TO_IMAGE = os.path.join(CWD_PATH, 'img')

start_time = datetime.now()

PATH_TO_SAVE = 'output'

with os.scandir(PATH_TO_IMAGE) as entries:
    for entry in entries:
        print(entry.name)
        image = cv2.imread(os.path.join(PATH_TO_IMAGE,entry.name))
        start_time = datetime.now()
        dsize = (1024, 1024)

        # resizedIMG = cv2.resize(image, dsize)

        x = 400
        y = 250
        h = 1024
        w = 1024
        crop_img = image[y:y+h, x:x+w]

        # cv2.imwrite(os.path.join(PATH_TO_SAVE, entry.name), resizedIMG)
        cv2.imwrite(os.path.join(PATH_TO_SAVE, entry.name), crop_img)



        print(datetime.now() - start_time)


