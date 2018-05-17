import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
from utils import label_map_util, visualization_utils

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Path to frozen detection graph (.pb)
PATH_TO_CKPT = os.path.join(os.path.sep, 'mnt2', 'projects',
                            'ai_dcDemoMultiFood', 'webDemo','ensembleAPI', 'models', 'frcnn.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'foodinc_label_map.pbtxt'
PATH_TO_ANNOTATION = ''

# path of data directory
folder_path = os.path.join(os.path.sep, 'mnt', 'dc', 'web_food_imgs',
                           'miso_soup')
#
PATH_OF_SAVE = os.path.join(os.path.sep, 'mnt', 'dc', 'results_food_detector')

PATH_OF_MULTIPLE_FOOD = os.path.join(os.path.sep, 'mnt', 'dc', 'web_food_imgs_multi_obj',
                           'miso_soup')

NUM_CLASSES = 67

# load annotations
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  image = image.convert('RGB')
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def overlapped_ratio(area_1, area_2):
    """
    :param area_1 & param area_2: [  ymin, xmin,  ymax, xmax ]
    :return: overlapped ratio
    """
    overlapped_area = (min([area_1[3], area_2[3]]) - max([area_1[1], area_2[1]]))*(min([area_1[2], area_2[2]]) - max([area_1[0], area_2[0]]))
    if overlapped_area > .0:
        overlapped_area_1 = overlapped_area/((area_1[2]-area_1[0])*(area_1[3]-area_1[1]))
        overlapped_area_2 = overlapped_area/((area_2[2]-area_2[0])*(area_2[3]-area_2[1]))
        print('{:.4f}  {:.4f}'.format(overlapped_area_1, overlapped_area_2))
        return max([overlapped_area_1, overlapped_area_2])
    else:
        return .0


def main():
    # Load a (frozen) Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # convert id to category name
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    TEST_IMAGE_PATHS = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', 'jpeg','.png', '.bmp', '.JPG', 'JPEG','.PNG', '.BMP'))]
    print('num of images = {}'.format(len(TEST_IMAGE_PATHS)))

    print('detecting...')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
                print('now detecting ', image_path)
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # remove the bboxes whose score is lower than threshold
                # & and overlapped ratio is too high
                bbox_threshold = 0.5
                overlapped_ratio_threshold = 0.7
                thresholded_boxes = []
                for b, s in zip(boxes[0], scores[0]):
                    if s > bbox_threshold:
                        for tb in thresholded_boxes:
                            # print('{} - {} with overlapped ratio:{}'.format(b, tb, overlapped_ratio(tb, b)))
                            if overlapped_ratio(tb, b) > overlapped_ratio_threshold:
                                break
                        else:
                            thresholded_boxes.append(b)
                    else:
                        break

                # TODO
                # save the info (one image one txt)
                # label ymin xmin  ymax xmax
                if True:
                    pass

                # move the data into single food, >2 food
                if(len(thresholded_boxes)) > 1:
                    pass

                """
                # Visualization of the results of a detection.
                visualization_utils.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.asarray(thresholded_boxes), #np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores), category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                # plt.figure(figsize=IMAGE_SIZE)
                # plt.imshow(image_np)
                im = Image.fromarray(image_np)
                im.save(os.path.join(PATH_OF_SAVE, image_path.split(os.path.sep)[-1]))
                """
    print('completed!')


if __name__ == '__main__':
    main()





