
import os
import cv2
import sys
from datetime import datetime
import time
# import image
IMAGE_NAME = "img"
# for otnositelnih petey
start_time = datetime.now()

CWD_PATH = ""
PATH_TO_IMAGE = os.path.join(CWD_PATH, "img")
PATH_TO_SAVE = "output"
# mode_name = sys.argv[1]
mode_name = input('mode: ')

with os.scandir(PATH_TO_IMAGE) as entries:
    for entry in entries:
        print(entry.name)
        image = cv2.imread(os.path.join(PATH_TO_IMAGE, entry.name))
        start_time = datetime.now()
        if mode_name == "crop":
            x = 400
            y = 250
            h = 1024
            w = 1024
            crop_img = image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(PATH_TO_SAVE, entry.name), crop_img)
        elif mode_name == "resize":
            dsize = (1024, 1024)
            resizedIMG = cv2.resize(image, dsize)
            cv2.imwrite(os.path.join(PATH_TO_SAVE, entry.name), resizedIMG)
        else:
            print("make your choice")

        print(datetime.now() - start_time)


