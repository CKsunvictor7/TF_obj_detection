"""
based on object_detection_tutorial.ipynb

this code loads a obj detection model and the corresponding label_maps,
then do detection on images and save the results as images
"""

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
from utils import visualization_utils, label_map_util

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


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


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Faster_RCNN_ResNet101_Foodinc_950k.pb

finc_v0 = {
    'NUM_CLASSES':67,
    # Path to frozen detection graph, the exp13_detection_155exp13_detection_155exp13_detection_155actual model that is used for the object detection.
    'PATH_TO_CKPT':os.path.join(os.path.sep, '/mnt2/models', 'Faster_RCNN_ResNet101_Foodinc_950k.pb'),
    # List of the strings that is used to add correct label for each box.
    'PATH_TO_LABELS':'label_maps/foodinc_label_map.pbtxt',
    'IMG_PATH':os.path.join(os.path.sep, '/mnt2/DB/155'),
    'SAVE_FIG':True,
    'PATH_OF_SAVE_FIG':'/mnt2/results/fincv0_155',
    'SHOW_INFO':False
}

exp13 = {
    'NUM_CLASSES':110,
    # Path to frozen detection graph, the exp13_detection_155exp13_detection_155exp13_detection_155actual model that is used for the object detection.
    'PATH_TO_CKPT':os.path.join(os.path.sep, '/mnt2/models/pb/exp13',
                                'frozen_inference_graph.pb'),
    # List of the strings that is used to add correct label for each box.
    'PATH_TO_LABELS':os.path.join(os.path.sep, '/mnt2/projects/TF_obj_detection/label_maps',
                                  'exp13_label_map.pbtxt'),
    'IMG_PATH':os.path.join(os.path.sep, '/mnt2/DB/155'),
    'SAVE_FIG':True,
    'PATH_OF_SAVE_FIG':'/mnt2/results/TF_visualization_export_dir',
    'SHOW_INFO':False,
}

exp15_gpu8 = {
    'NUM_CLASSES': 130,
    # Path to frozen detection graph, the exp13_detection_155exp13_detection_155exp13_detection_155actual model that is used for the object detection.
    'PATH_TO_CKPT': os.path.join(os.path.sep, '/mnt2/models/pb/exp15',
                                 'frozen_inference_graph.pb'),
    # List of the strings that is used to add correct label for each box.
    'PATH_TO_LABELS': os.path.join(os.path.sep,
                                   '/mnt2/projects/TF_obj_detection/label_maps',
                                   'exp15_label_map.pbtxt'),
    'IMG_PATH': os.path.join(os.path.sep, '/mnt2/DB/test_samples/'),
    'SAVE_FIG': True,
    'PATH_OF_SAVE_FIG': '/mnt2/DB/test_samples/test_vis/',#'/mnt2/results/exp15_on156_vis',
    'SHOW_INFO': False,
}

exp16_gpu8 = {
    'NUM_CLASSES': 165,
    # Path to frozen detection graph, the exp13_detection_155exp13_detection_155exp13_detection_155actual model that is used for the object detection.
    'PATH_TO_CKPT': os.path.join(os.path.sep, '/mnt2/models/pb/exp16_772399',
                                 'frozen_inference_graph.pb'),
    # List of the strings that is used to add correct label for each box.
    'PATH_TO_LABELS': os.path.join(os.path.sep,
                                   '/mnt2/projects/TF_obj_detection/label_maps',
                                   'exp16_label_map.pbtxt'),
    'IMG_PATH': os.path.join(os.path.sep, '/mnt2/DB/test_samples/'), # '/mnt2/DB/156'
    'SAVE_FIG': True,
    'PATH_OF_SAVE_FIG': '/mnt2/DB/test_samples/test_vis/',
    'SHOW_INFO': False,
}

config = exp16_gpu8


TEST_IMAGE_PATHS = get_file_list(config['IMG_PATH'], ('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG', '.PNG', '.BMP'))
print('num of images = {}'.format(len(TEST_IMAGE_PATHS)))


def main():
    print('loading models')
    s_time = time.time()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(config['PATH_TO_CKPT'], 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('loading model done, took {} sec'.format( time.time() - s_time))

    # convert id to category name
    label_map = label_map_util.load_labelmap(config['PATH_TO_LABELS'])
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=config['NUM_CLASSES'], use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

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
                img_batch = np.expand_dims(image_np, 0)
                #img_batch = image_np

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
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: img_batch})

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
                if config['SHOW_INFO']:
                    print('num_detections = ', output_dict['num_detections'])
                    print('top 3 boxes:')
                    for i in range(3):
                        print('bbox-{} = {}, score={}, bbox={}'.format(i+1,
                        output_dict['detection_classes'][i], output_dict['detection_scores'][i], output_dict['detection_boxes'][i]))

                    print('detection_classes = ', output_dict['detection_classes'])
                    print('detection_scores = ', output_dict['detection_scores'])
                    for i in output_dict['detection_boxes']:
                        print(i)

                # print
                if config['SAVE_FIG']:
                    visualization_utils.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.asarray(output_dict['detection_boxes']), #np.squeeze(boxes),
                        np.squeeze(output_dict['detection_classes']).astype(np.int32),
                        np.squeeze(output_dict['detection_scores']), category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    # plt.figure(figsize=IMAGE_SIZE)
                    # plt.imshow(image_np)
                    im = Image.fromarray(image_np)
                    im.save(os.path.join(config['PATH_OF_SAVE_FIG'], image_path.split(os.path.sep)[-1]))


if __name__ == '__main__':
    main()

