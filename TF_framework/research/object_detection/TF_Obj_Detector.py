import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image
#from utils import label_map_util
#from utils import visualization_utils as vis_util

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# Path to frozen detection graph, the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(os.path.sep, 'mnt', 'dc', 'pb_UEC256_Res101_378643',
                            'frozen_inference_graph.pb')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(os.path.sep, 'mnt2', 'projects', 'TF_obj_detection', 'label_maps', 'UEC_label_map.pbtxt')

NUM_CLASSES = 256


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def get_file_list(dir_path, extensions):
    """
    return abs_path of all files who ends with __ in all sub-directories as a list
    extensions: a tuple to specify the file extension
    ex: ('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG', '.PNG', '.BMP')
    """
    file_list = []
    for f in os.listdir(dir_path):
        path = os.path.join(dir_path, f)
        if os.path.isdir(path):
            file_list = file_list + get_file_list(path, extensions)
        elif f.endswith(extensions):
            file_list.append(path)

    return file_list
super_folder_path = os.path.join(os.path.sep, '/mnt/dc/', 'UEC256_images')
TEST_IMAGE_PATHS = get_file_list(super_folder_path, (
    '.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG', '.PNG', '.BMP'))
print('num of images = {}'.format(len(TEST_IMAGE_PATHS)))



def main():
    print('loading models')
    s_time = time.time()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('loading model done, took {}'.format( time.time() - s_time))

    with detection_graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[
                        key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            for image_path in TEST_IMAGE_PATHS:
                s_time = time.time()
                print(image_path)
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np = np.expand_dims(image_np, 0)
                img_batch = image_np

                # example code of using batch images
                BATCH = False
                if BATCH:
                    # batch-input, the shape should be (nb_batch, w, h, nb_channel)
                    # ex: (578, 432, 3) -> (1, 578, 432, 3)
                    # TODO: need to resize images to same size
                    # TODO: decode the batch output
                    img_batch = np.vstack((image_np, image_np))

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0],
                                               [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0],
                                               [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0],
                        image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)

                image_tensor = tf.get_default_graph().get_tensor_by_name(
                    'image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: img_batch})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                print('one iteration took {}'.format(time.time() - s_time))

                """
                decode detection results:
                default we get 300 bounding boxes, with classes and score
                the order is from big to small by score
                """
                print('num_detections = ', output_dict['num_detections'])
                print('top 3 boxes:')
                for i in range(3):
                    print('bbox-{} = {}, score={}, bbox={}'.format(i+1,
                    output_dict['detection_classes'][i], output_dict['detection_scores'][i], output_dict['detection_boxes'][i]))


if __name__ == '__main__':
    main()
