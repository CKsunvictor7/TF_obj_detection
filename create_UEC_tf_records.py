"""
file processor for UEC

UEC256 structure:
    *1 ~ 256 folders as 256 categories.
    *bb_infos.txt is the annotations only contains bounding boxes of one categories(corresponding to this folder)
     , so that means there are same image in different categories

* create feasible annotations(one txt with all bounding boxes info)@ create_annotation_UEC
* create TF RECORD for @main
"""
import hashlib
import io
import logging
import os
import re
import numpy as np

import PIL.Image
import tensorflow as tf
from utils.tools import split_data
import dataset_util


img_dir = os.path.join(os.path.sep, 'mnt/dc', 'UEC256_images')

output_train_path = os.path.join(os.path.sep, 'mnt', 'dc', 'tfrecords',
                                 'UEC256_train.record')
output_eval_path = os.path.join(os.path.sep, 'mnt', 'dc', 'tfrecords',
                                'UEC256_eval.record')

category_path = os.path.join(os.path.sep, '/mnt/dc', 'UECFOOD256', 'category.txt')
annotation_dir = os.path.join(os.path.sep, '/mnt/dc/UEC_ground_truth')


def create_annotation_UEC():
    """
    create feasible annotations(one txt with all bounding boxes info)
    """
    for i in range(1, 257):
        with open(os.path.join(os.path.sep, '/mnt/dc/UECFOOD256', str(i),
                               'bb_info.txt'), 'r') as r:
            r.readline()  # discard ths first line
            for line in r.readlines():
                arrs = line.rstrip().split(' ')
                with open(os.path.join(annotation_dir, arrs[0] + '.txt'),
                          'a+') as w:
                    w.write(
                        '{} {} {} {} {}\n'.format(i, arrs[1], arrs[2], arrs[3],
                                                  arrs[4]))


def dict_to_tf_example(data, categories_name):
    print(data['abs_img_path'])
    with tf.gfile.GFile(data['abs_img_path'], 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    # TODO: OSError: cannot identify image file <_io.BytesIO object at 0x7f56dd982a98>
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width, height = image.size

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    for idx, obj in enumerate(data['boxes']):
        xmin.append(np.clip(obj[0] / width, 0, 1))
        ymin.append(np.clip(obj[1] / height, 0, 1))
        xmax.append(np.clip(obj[2] / width, 0, 1))
        ymax.append(np.clip(obj[3] / height, 0, 1))
        classes_text.append(
            categories_name[data['labels'][idx] - 1].encode('utf8'))
        classes.append(int(data['labels'][idx]))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        # bug: 'nasi campur' has type str, but expected one of: bytes  =>  .encode('utf8')
        'image/object/class/text': dataset_util.bytes_list_feature(
            classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example


def show_nb_bbox_for_each_images():
    annotation_list = [ os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir) if f.endswith(('.txt'))]
    ground_truth_bbox_info = {}
    for f in annotation_list:
        c = 0
        with open(f, 'r') as r:
            for l in r.readlines():
                c += 1
        ground_truth_bbox_info[os.path.basename(f)] = c

    for k, v in ground_truth_bbox_info.items():
        print(k, v)


def main(_):
    """
    make annotation and check the info
    """
    # create_annotation_UEC()
    #TODO should show the nb of each category
    #show_nb_bbox_for_each_images()
    #exit()

    # category.txt: a list with all category names
    categories_name = []
    with open(category_path, 'r') as r:
        # discard the first line
        r.readline()
        for line in r.readlines():
            print(line.rstrip().split('\t')[-1])
            categories_name.append(line.rstrip().split('\t')[-1])

    img_list = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    print("num of imgs = ", len(img_list))

    # TODO class balance
    train_list, eval_list = split_data(img_list, split_percentage=10)

    # tf.initialize_all_variables().run()
    writer = tf.python_io.TFRecordWriter(output_train_path)

    for idx, img_f in enumerate(train_list):
        if idx % 100 == 0:
            print('now progress... {:.3f}'.format(
                float(idx) / float(len(train_list))))

        # read annotation
        """
            Annotation file:
            *one for one images
            *label bboxes_info1 info2 info 3 info4 => * number of bbox
        """
        anno_path = os.path.join(annotation_dir, str(os.path.splitext(img_f)[0]))
        bboxes = []
        labels = []
        with open(anno_path, 'r') as r:
            for line in r.readlines():
                arrs = line.rstrip().split(' ')
                labels.append(int(arrs[0]))
                bboxes.append([float(arrs[1]), float(arrs[2]), float(arrs[3]),
                               float(arrs[4])])

        # TODO: survey this structure
        data = {
            'abs_img_path': os.path.join(img_dir, img_f),
            'filename': img_f,
            'boxes': bboxes,
            'labels': labels
        }

        tf_example = dict_to_tf_example(data, categories_name)
        writer.write(tf_example.SerializeToString())

    writer.close()

    writer_eval = tf.python_io.TFRecordWriter(output_eval_path)

    for idx, img_f in enumerate(eval_list):
        if idx % 100 == 0:
            print('now progress... {:.3f}'.format(
                float(idx) / float(len(eval_list))))

        # read annotation
        """
            Annotation file:
            *one for one images
            *label xmin ymin xmax ymax => * number of bbox
        """
        anno_path = os.path.join(annotation_dir, str(img_f.split('.')[0]))
        bboxes = []
        labels = []
        with open(anno_path, 'r') as r:
            for line in r.readlines():
                arrs = line.rstrip().split(' ')
                labels.append(int(arrs[0]))
                bboxes.append([float(arrs[1]), float(arrs[2]), float(arrs[3]),
                               float(arrs[4])])

        # TODO: survey this structure
        data = {
            'abs_img_path': os.path.join(img_dir, img_f),
            'filename': img_f,
            'boxes': bboxes,
            'labels': labels
        }

        tf_example = dict_to_tf_example(data, categories_name)
        writer_eval.write(tf_example.SerializeToString())

    writer_eval.close()


if __name__ == '__main__':
    tf.app.run()

"""
* detections[classes] =  [[28.0, 2.0]]
* detections[scores] =  [[0.3440466523170471, 0.7167795896530151]]

* {'total_hits': 14993, 'max_score': 1.1918111, 'hits': [{'_index': 'f762ef22-e660-434f-9071-a10ea6691c27', '_type': 'item', '_id': '57b7967a26e132f919d4224e', '_score': 1.1918111, 'fields': {'item_id': '57b7967a26e132f919d4224e', 'item_name': 'Japanese Style Noodles', 'brand_name': 'Shirakiku', 'nf_serving_size_qty': 1, 'nf_serving_size_unit': 'serving'}}]}
* protein {'usage_reports': [{'metric': 'hits', 'period': 'day', 'current_value': '406', 'max_value': '1000', 'period_start': '2017-09-07 00:00:00 +0000', 'period_end': '2017-09-08 00:00:00 +0000'}, {'metric': 'search', 'period': 'day', 'current_value': '206', 'max_value': '500', 'period_start': '2017-09-07 00:00:00 +0000', 'period_end': '2017-09-08 00:00:00 +0000'}, {'metric': 'itemView', 'period': 'day', 'current_value': '200', 'max_value': '200', 'period_start': '2017-09-07 00:00:00 +0000', 'period_end': '2017-09-08 00:00:00 +0000'}, {'metric': 'brandView', 'period': 'day', 'current_value': '0', 'max_value': '100', 'period_start': '2017-09-07 00:00:00 +0000', 'period_end': '2017-09-08 00:00:00 +0000'}, {'metric': 'upcLookUp', 'period': 'day', 'current_value': '0', 'max_value': '50', 'period_start': '2017-09-07 00:00:00 +0000', 'period_end': '2017-09-08 00:00:00 +0000'}, {'metric': 'v2_autocomplete', 'period': 'day', 'current_value': '0', 'max_value': '2000', 'period_start': '2017-09-07 00:00:00 +0000', 'period_end': '2017-09-08 00:00:00 +0000'}, {'metric': 'v2_natural', 'period': 'day', 'current_value': '0', 'max_value': '500', 'period_start': '2017-09-07 00:00:00 +0000', 'period_end': '2017-09-08 00:00:00 +0000'}, {'metric': 'v2_search', 'period': 'day', 'current_value': '0', 'max_value': '500', 'period_start': '2017-09-07 00:00:00 +0000', 'period_end': '2017-09-08 00:00:00 +0000'}, {'metric': 'v2_brand_search', 'period': 'day', 'current_value': '0', 'max_value': '200', 'period_start': '2017-09-07 00:00:00 +0000', 'period_end': '2017-09-08 00:00:00 +0000'}, {'metric': 'v2_item', 'period': 'day', 'current_value': '0', 'max_value': '50', 'period_start': '2017-09-07 00:00:00 +0000', 'period_end': '2017-09-08 00:00:00 +0000'}, {'metric': 'v2_brand', 'period': 'day', 'current_value': '0', 'max_value': '100', 'period_start': '2017-09-07 00:00:00 +0000', 'period_end': '2017-09-08 00:00:00 +0000'}, {'metric': 'trackNatural', 'period': 'day', 'current_value': '0', 'max_value': '200', 'period_start': '2017-09-07 00:00:00 +0000', 'period_end': '2017-09-08 00:00:00 +0000'}], 'error_code': None, 'error_message': 'usage limits are exceeded'}
* [

"""