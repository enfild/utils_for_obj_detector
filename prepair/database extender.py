
import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from xml.dom import minidom

from datetime import datetime
from PIL import Image


def noisy(noise_typ,image):
    if noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

start_time = datetime.now()
CWD_PATH = ""
PATH_TO_IMAGE = os.path.join(CWD_PATH, "img")
PATH_TO_SAVE = "output"
mode_name = input('mode: ')
dom = minidom.parse()
with os.scandir(PATH_TO_IMAGE) as entries:
    Numb_img = 0
    for entry in entries:
        extension = pathlib.Path(entry).suffix
        print("extentions: " + extension)

        if extension == ".png" or extension == ".jpg":
            # my_file_path[:-1]
            name_xml_file = str(entry.name[0:-3] + "xml")
            print (name_xml_file)

            for xml_element in os.scandir(PATH_TO_IMAGE):
                if xml_element.is_file():
                    if xml_element.name == name_xml_file:
                        print("JOPA: " + xml_element.name)
                        dom = minidom.parse(xml_element.name)

            image = cv2.imread(os.path.join(PATH_TO_IMAGE, entry.name))
            start_time = datetime.now()
        
            if mode_name == "proc":  
                #negativ  
                neg_image = (255-image)
                cv2.imwrite(os.path.join(PATH_TO_SAVE, "neg_" + entry.name), neg_image)

                #Brigthness + 30 
                alpha = 1.0
                beta = 30
                image_brig = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                cv2.imwrite(os.path.join(PATH_TO_SAVE, "br+30" + entry.name), image_brig)

                #Brigthness -30 
                alpha = 1.0
                beta = -30
                image_brig = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                cv2.imwrite(os.path.join(PATH_TO_SAVE, "br-30" + entry.name), image_brig)

                #Contrast +0.3
                alpha = 1.3
                beta = 0
                image_cont = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                cv2.imwrite(os.path.join(PATH_TO_SAVE, "cont+03" + entry.name), image_cont)

                #Contrast -0.3
                alpha = 0.7
                beta = 0
                image_cont = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                cv2.imwrite(os.path.join(PATH_TO_SAVE, "cont-03" + entry.name), image_cont)
    
                #subpixel shift
                row, col, channel = image.shape
                M = np.float32([[1,0,0.5],[0,1,0.5]])
                image_shift = cv2.warpAffine(image, M, (col, row), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                cv2.imwrite(os.path.join(PATH_TO_SAVE, "shift_" + entry.name), image_shift)
    
                # elif proc_name == "conturs":
                #     alpha = 1.2
                #     beta = 60
                #     image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                #     image = cv2.Canny(image, 100, 200) 
                #     cv2.imwrite(os.path.join(PATH_TO_SAVE, entry.name), image)
    
            elif mode_name == "flip":
                image_ver = cv2.flip(image, 0)
                cv2.imwrite(os.path.join(PATH_TO_SAVE, entry.name), image_ver)
    
                image_hor = cv2.flip(image, 1)
                cv2.imwrite(os.path.join(PATH_TO_SAVE, entry.name), image_hor)
    
                image_bot = cv2.flip(image, -1)
                cv2.imwrite(os.path.join(PATH_TO_SAVE, entry.name), image_bot)
    
            else:
                print("make your choice")

        else:
            print(" hren' ")
    
        Numb_img += 1
        print(datetime.now() - start_time)



