import numpy as np
import shutil
import sys
import tensorflow as tf
from PIL import Image
from utils import label_map_util, visualization_utils
from UEC256_config import *
# from utils import ops as utils_ops
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


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
        # print('{:.4f}  {:.4f}'.format(overlapped_area_1, overlapped_area_2))
        return max([overlapped_area_1, overlapped_area_2])
    else:
        return .0


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

"""
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
          output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict
"""

def main():
    TEST_IMAGE_PATHS = get_file_list(super_folder_path, (
    '.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG', '.PNG', '.BMP'))
    print('num of images = {}'.format(len(TEST_IMAGE_PATHS)))

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

    print('detecting...')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
                print('now detecting ', image_path)
                image = Image.open(image_path)
                print(image.size)
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
                # bug here: InvalidArgumentError: NodeDef mentions attr 'identical_element_shapes'
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

                # save the info (one image one txt) as the 'img_name.txt'
                # under PATH_TO_SAVE_RESULTS
                if True:
                    with open(os.path.join(PATH_TO_SAVE_RESULTS, os.path.basename(image_path) + '_.txt'), 'w') as w:
                        for bbox, c in zip(thresholded_boxes, classes[0]):
                            # label ymin xmin ymax xmax   (save as ratio)
                            w.write('{},{},{},{},{}\n'.format(int(c), bbox[0], bbox[1], bbox[2], bbox[3]))
                            print('{},{},{},{},{}'.format(int(c), bbox[0], bbox[1], bbox[2], bbox[3]))

                # move the data into single food,
                # >2 food: multiple food or 67(bento)
                """
                if(len(thresholded_boxes)) > 1 or 67 in classes[0][:len(thresholded_boxes)]:
                    if not os.path.exists(
                            os.path.join(os.path.dirname(image_path), '2')):
                        os.makedirs(
                            os.path.join(os.path.dirname(image_path), '2'))
                    dst_path = os.path.join(os.path.dirname(image_path), '2',
                                            os.path.basename(image_path))
                    shutil.move(image_path, dst_path)
                    print(dst_path)
                else:
                    if not os.path.exists(
                            os.path.join(os.path.dirname(image_path), '1')):
                        os.makedirs(
                            os.path.join(os.path.dirname(image_path), '1'))
                    dst_path = os.path.join(os.path.dirname(image_path), '1',
                                            os.path.basename(image_path))
                    shutil.move(image_path, dst_path)
                    print(dst_path)
                """
                
                # Visualization of the results of a detection
                if SAVE_FIG:
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
                    im.save(os.path.join(PATH_OF_SAVE_FIG, image_path.split(os.path.sep)[-1]))

    print('completed!')


if __name__ == '__main__':
    main()





