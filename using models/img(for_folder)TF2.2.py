import numpy as np
import argparse
import os
import sys
import tensorflow as tf
import cv2
import pathlib
from datetime import datetime
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

MODEL_NAME = 'inference_graph/saved_model'
IMAGE_NAME = 'img'

CWD_PATH = ''

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME)

PATH_TO_LABELS = os.path.join(CWD_PATH,'data','object-detection.pbtxt')

PATH_TO_IMAGE = os.path.join(CWD_PATH,'img')

detection_model = tf.saved_model.load(str(PATH_TO_CKPT))

def run_inference_for_single_image(model, image):

    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                    output_dict['detection_masks'], output_dict['detection_boxes'],
                                    image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    return output_dict


def run_inference(model, category_index, image):
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    cv2.imshow('object_detection', cv2.resize(image, (800, 600)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break


if __name__ == '__main__':
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    with os.scandir(PATH_TO_IMAGE) as entries:
        for entry in entries:
            print(entry.name)
            image = cv2.imread(os.path.join(PATH_TO_IMAGE,entry.name))
            start_time = datetime.now()
            run_inference(detection_model, category_index, image)
            print(datetime.now() - start_time)