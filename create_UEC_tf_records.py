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
import pandas as pd


#img_dir = os.path.join(os.path.sep, 'mnt/dc', 'UEC256_images')

#output_train_path = os.path.join(os.path.sep, 'mnt', 'dc', 'tfrecords', 'UEC_common_train.record')
#output_eval_path = os.path.join(os.path.sep, 'mnt', 'dc', 'tfrecords', 'UEC_common_eval.record')



gpu4 = {
    "ORIGINAL_DIR":'---',
    "img_dir":'/mnt2/DB/exp15',
    "YOLO_DIR":'/mnt2/DB/exp15_annos',
    'category_path':'label_maps/exp15.names',
    'output_train_path':'/mnt2/DB/TFrecord/UEC_exp15_train.record',
    'output_eval_path':'/mnt2/DB/TFrecord/UEC_exp15_val.record'
}

Min_num_list = 95
DO_MERGE = True
skip_country_list = ['southeast_asia', 'Chinese', 'Hawaii', 'Korea', 'Taiwan']
skip_list = ['unknown_food', 'skip']
DEBUG = False


server = gpu4

# id starts from 1, so remember to -1
UEC_skip_list = \
    [164, 167, 169, 171, 175, 176, 177, 178, 179, 180, 181, 182, 185, 186,
     191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 203, 204, 205, 207, 208,
     210, 214, 215, 219, 221, 222, 226, 227, 228, 229, 230, 231, 232, 233, 234,
     235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 251,
     252, 253, 254, 255, 256]

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
    #print(data['abs_img_path'])
    #print('labels = ', data['labels'])

    with tf.gfile.GFile(data['abs_img_path'], 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    # TODO: OSError: cannot identify image file <_io.BytesIO object at 0x7f56dd982a98>
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        print('error, Image format not JPEG')
        return
        #raise ValueError('Image format not JPEG')
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
            categories_name[data['labels'][idx]].encode('utf8'))
        # notice: label start from 1
        classes.append(int(data['labels'][idx]) + 1)

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


def convert_to_label_map(categories_name):
    with open('label_maps/label_map.pbtxt', 'w') as w:
        for id, c in enumerate(categories_name):
            w.write('item {\n')
            w.write('  id: {}\n'.format(id+1))
            w.write('  name: "'"{}"'"\n'.format(c))
            w.write('}\n\n')


def YOLO_2_coord(img_f, x, y, w, h):
    image = PIL.Image.open(os.path.join(server['img_dir'], str(img_f.split('.')[0]) + '.jpg'))

    width, height = image.size
    xmin = (x - w/2)*width
    ymin = (y - h/2)*height
    xmax = (x + w/2)*width
    ymax = (y + h/2)*height

    return xmin, ymin, xmax, ymax


def make_TFRecord_id_version(category_list, anno_list, TFRecord_name):
    """
    for the input format: id x y w d, id was generated by YOLOv3 MakeDB
    """
    nb_encoded = 0
    writer = tf.python_io.TFRecordWriter(TFRecord_name)

    for idx, img_f in enumerate(anno_list):
        if idx % 100 == 0:
            print('now progress... {:.3f}'.format(
                float(idx) / float(len(anno_list))))

        # read annotation
        """
            Annotation file:
            *one for one images
            *label bboxes_info1 info2 info 3 info4 => * number of bbox
        """
        anno_path = os.path.join(server['YOLO_DIR'], img_f)
        bboxes = []
        labels = []
        with open(anno_path, 'r') as r:
            for line in r.readlines():
                pieces = line.rstrip().split(' ')

                if DEBUG: print(line)

                if len(pieces) < 2:
                    print('sth wrong @',line )
                    print('@',img_f)
                    break
                try:
                    bboxes.append(YOLO_2_coord(img_f, float(pieces[1]),
                                           float(pieces[2]),
                                           float(pieces[3]),
                                           float(pieces[4])))
                    labels.append(int(pieces[0]))
                except:
                    print('{} does not exist, break'.format(img_f))
                    break

        if not labels:
            # print('skipped case, continue')
            continue
        nb_encoded += 1

        data = {
            'abs_img_path': os.path.join(server['img_dir'],
                                         str(img_f.split('.')[0]) + '.jpg'),
            'filename': str(img_f.split('.')[0]) + '.jpg',
            'boxes': bboxes,
            'labels': labels
        }

        with tf.gfile.GFile(data['abs_img_path'], 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        # TODO: OSError: cannot identify image file <_io.BytesIO object at 0x7f56dd982a98>
        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            print('error, Image format not JPEG')
            continue

        tf_example = dict_to_tf_example(data, category_list)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('encode {} files'.format(nb_encoded))



def make_TFRecord(category_list, anno_list, table, TFRecord_name):
    nb_encoded = 0
    writer = tf.python_io.TFRecordWriter(TFRecord_name)

    for idx, img_f in enumerate(anno_list):
        if idx % 100 == 0:
            print('now progress... {:.3f}'.format(
                float(idx) / float(len(anno_list))))

        # read annotation
        """
            Annotation file:
            *one for one images
            *label bboxes_info1 info2 info 3 info4 => * number of bbox
        """
        anno_path = os.path.join(server['YOLO_DIR'], img_f)
        bboxes = []
        labels = []
        with open(anno_path, 'r') as r:
            for line in r.readlines():
                pieces = line.rstrip().split(' ')

                if len(pieces) < 2:
                    print('sth wrong @',line )
                    print('@',img_f)
                    break

                v3_name = pieces[0]

                if DEBUG: print(line)

                if v3_name in skip_list:
                    continue

                if DO_MERGE:
                    # if this category is one of the category
                    if v3_name in category_list:
                        labels.append(category_list.index(v3_name))
                        bboxes.append(YOLO_2_coord(img_f, float(pieces[1]),
                                                   float(pieces[2]),
                                                   float(pieces[3]),
                                                   float(pieces[4])))
                        continue

                    new_name = \
                    table[table['En_name'] == v3_name]['merge'].values[0]
                    # print('new_name = ', new_name)
                    # if the merged category is one of the category
                    if str(new_name) != 'nan' and str(
                            new_name) in category_list:
                        # print('merged: {} -> {}'.format(v3_name, new_name))
                        labels.append(category_list.index(new_name))
                        bboxes.append(YOLO_2_coord(img_f, float(pieces[1]),
                                                   float(pieces[2]),
                                                   float(pieces[3]),
                                                   float(pieces[4])))
                else:
                    if v3_name not in category_list:
                        labels.append(category_list.index(v3_name))
                        bboxes.append(YOLO_2_coord(img_f, float(pieces[1]),
                                                   float(pieces[2]),
                                                   float(pieces[3]),
                                                   float(pieces[4])))

        if not labels:
            # print('skipped case, continue')
            continue
        nb_encoded += 1

        data = {
            'abs_img_path': os.path.join(server['img_dir'],
                                         str(img_f.split('.')[0]) + '.jpg'),
            'filename': str(img_f.split('.')[0]) + '.jpg',
            'boxes': bboxes,
            'labels': labels
        }

        tf_example = dict_to_tf_example(data, category_list)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('encode {} files'.format(nb_encoded))


def main(_):
    """
    make annotation and check the info
    """
    # create_annotation_UEC()
    #TODO should show the nb of each category
    #show_nb_bbox_for_each_images()
    #exit()

    table = pd.read_csv('labelsheet_v3.csv', encoding='utf-8')
    # remove NaN rows
    table.dropna(how='all', inplace=True)
    table.reset_index(drop=True, inplace=True)


    # category.txt: a list with all category names, one line one category
    category_list = []
    with open(server['category_path'], 'r') as r:
        for line in r.readlines():
            # print(line.rstrip())
            category_list.append(line.rstrip())
            #print(line.rstrip().split('\t')[-1])
            #categories_name.append(line.rstrip().split('\t')[-1])
    print('nb of category = ', len(category_list))

    # read category & convert to label_map.pbtxt
    convert_to_label_map(category_list)
    print('converted to label_maps/label_map.pbtxt')


    anno_list = [f for f in os.listdir(server['YOLO_DIR']) if f.endswith('.txt')]

    try:
        anno_list.remove(os.path.join(server['YOLO_DIR'], 'classes.txt'))
    except:
        print('no classes to remove')
    print("num of annotations = ", len(anno_list))

    # TODO class balance
    train_list, eval_list = split_data(anno_list, split_percentage=10)
    print('num of train data =', len(train_list))
    print('num of eval data =', len(eval_list))  # remember to set this value on config

    # tf.initialize_all_variables().run()
    make_TFRecord_id_version(category_list, train_list, server['output_train_path'])
    make_TFRecord_id_version(category_list, eval_list, server['output_eval_path'])


    """
    nb_val = 0
    writer_eval = tf.python_io.TFRecordWriter(server['output_eval_path'])
    for idx, img_f in enumerate(eval_list):
        if idx % 100 == 0:
            print('now progress... {:.3f}'.format(
                float(idx) / float(len(eval_list))))

        # read annotation
        anno_path = os.path.join(server['YOLO_DIR'], img_f)
        bboxes = []
        labels = []
        with open(anno_path, 'r') as r:
            for line in r.readlines():
                pieces = line.rstrip().split(' ')
                label = pieces[0]
                if label not in categories_name:
                    pass
                else:
                    labels.append(categories_name.index(label))
                    bboxes.append(YOLO_2_coord(img_f, float(pieces[1]),
                                                float(pieces[2]),
                                                float(pieces[3]),
                                                float(pieces[4])))

        if not labels:
            # print('skipped case, continue')
            continue
        nb_val += 1
        # TODO: can be optimized
        data = {
            'abs_img_path': os.path.join(server['img_dir'], str(img_f.split('.')[0]) + '.jpg'),
            'filename': str(img_f.split('.')[0]) + '.jpg',
            'boxes': bboxes,
            'labels': labels
        }

        tf_example = dict_to_tf_example(data, categories_name)
        writer_eval.write(tf_example.SerializeToString())

    writer_eval.close()
    print('final nb of train = ', nb_train)
    print('final nb of val = ', nb_val)
    """

if __name__ == '__main__':
    tf.app.run()

"""
for common DB: nb of train=20457, val=2285
for exp13: ('num of train data =', 22951) ('num of eval data =', 2535)
"""