import os
import cv2
import sys
from datetime import datetime
import time
import pathlib
import xml.etree.ElementTree as ET
start_time = datetime.now()

CWD_PATH = ""
PATH_TO_IMAGE = os.path.join(CWD_PATH, "test")
PATH_TO_SAVE = "output"

old_width = 0
old_height = 0

width = int(input("width: "))
height = int(input("height: "))

def xml_create(path):
    tree = ET.parse(path)
    root = tree.getroot()
    source_file = root.find('filename').text
    print('source_file: ')
    print(source_file)
    for elem in root.iter('filename'):
        elem.text = source_file

    for elem in root.iter('width'):
        old_width = int(elem.text)
        elem.text = str(width)

    for elem in root.iter('height'):
        old_height = int(elem.text)
        elem.text = str(height)

    for elem in root.iter('xmin'):
        new_xmin = int(int(elem.text) * (width / old_width))
        elem.text = str((new_xmin))
        print(new_xmin)

    for elem in root.iter('xmax'):
        new_xmax = int(int(elem.text) * (width / old_width))
        elem.text = str(new_xmax)
        print(new_xmax)

    for elem in root.iter('ymin'):
        new_ymin = int(int(elem.text) * (height / old_height))
        elem.text = str(new_ymin)
        print(new_ymin)

    for elem in root.iter('ymax'):
        new_ymax = int(int(elem.text) * (height / old_height))
        elem.text = str(new_ymax)
        print(new_ymax)

    output_file = source_file[0:-4] + ".xml"
    print("output_file: ")
    print(output_file)
    tree.write(output_file)


with os.scandir(PATH_TO_IMAGE) as entries:
    for entry in entries:
        extension = pathlib.Path(entry).suffix
        print("extentions: " + extension)

        if extension == ".png" or extension == ".jpg":
            # my_file_path[:-1]
            name_xml_file = str(entry.name[0:-3] + "xml")

            PATH_TO_XML = os.path.join(CWD_PATH, PATH_TO_IMAGE, name_xml_file)
            print(entry.name)
            image = cv2.imread(os.path.join(PATH_TO_IMAGE, entry.name))
            start_time = datetime.now()
# resize img
            dsize = (width, height)
            resizedIMG = cv2.resize(image, dsize)
            cv2.imwrite(os.path.join(PATH_TO_SAVE, entry.name), resizedIMG)
#writing resize xml
            xml_create(PATH_TO_XML)

        else:
            print("   queue XML ")
    
        print(datetime.now() - start_time)