"""
Generate TFrecord files for training and validation:

To be aligned with YOLOv3(id & category):
using same annotation files generated by YOLO, which format is 'id x y w h'
-> pass to make_TFRecord_id_version() to get id xmin, ymin, xmax, ymax
-> serielize to .ecord by dict_to_tf_example()

input:
1) 'YOLO_DIR': annotations generated by YOLO
2) 'category_path' of category.names storing category list

output:
1) label_map.pbtxt stored in 'label_maps_path'
2) train.record & val.record

------
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
import dataset_util
import pandas as pd

gpu4 = {
    "ORIGINAL_DIR":'---',
    "img_dir":'/mnt2/DB/exp15',
    "YOLO_DIR":'/mnt2/DB/exp15_annos',
    'category_path':'label_maps/exp15.names',
    'output_train_path':'/mnt2/DB/TFrecord/UEC_exp15_train.record',
    'output_val_path':'/mnt2/DB/TFrecord/UEC_exp15_val.record'
}

gpu8_exp16 = {
    "img_dir":'/mnt2/DB/exp15',
    "YOLO_DIR":'/mnt2/DB/exp16_annos',
    'category_path':'label_maps/exp16.names',
    'label_maps_path':'label_maps/exp16_label_map.pbtxt',
    'output_train_path':'/mnt2/DB/TFrecord/UEC_exp16_train.record',
    'output_val_path':'/mnt2/DB/TFrecord/UEC_exp16_val.record',
}

gpu4_exp16 = {
    "img_dir":'/mnt2/DB/exp15',
    "YOLO_DIR":'/mnt2/DB/exp16_annos',
    'category_path':'label_maps/exp16.names',
    'label_maps_path':'label_maps/exp16_label_map.pbtxt',
    'output_train_path':'/mnt2/DB/TFrecord/UEC_exp16_train.record',
    'output_val_path':'/mnt2/DB/TFrecord/UEC_exp16_val.record',
}

DEBUG = False

server = gpu4_exp16


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


def split_data(ids, split_percentage=20, stratified=True):
    num_all = len(ids)

    shuffled_index = np.random.permutation(
        np.arange(num_all))  # make a shuffle idx array

    # calcualate the train & validation index
    num_small = int(num_all // (100 / split_percentage))
    num_big = num_all - num_small

    ix_big = shuffled_index[:num_big]
    ix_small = shuffled_index[num_big:]

    # divide
    id_big = []
    for idx in ix_big:
        id_big.append(ids[idx])
    id_small = []
    for idx in ix_small:
        id_small.append(ids[idx])

    print('num of id_big = ', len(id_big))
    print('num of id_small = ', len(id_small))

    # y_valid must be np array as 'int64'
    return id_big, id_small

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


def YOLO_2_coord(img_f, x, y, w, h):
    image = PIL.Image.open(os.path.join(server['img_dir'], str(img_f.split('.')[0]) + '.jpg'))

    width, height = image.size
    xmin = (x - w/2)*width
    ymin = (y - h/2)*height
    xmax = (x + w/2)*width
    ymax = (y + h/2)*height

    return xmin, ymin, xmax, ymax


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


def make_TFRecord_id_version(category_list, anno_list, TFRecord_name):
    """
    make .recordd by the input format: id x y w d,
    which annotations were generated by YOLOv3 MakeDB
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


def convert_to_label_map(categories_name):
    with open(server['label_maps_path'], 'w') as w:
        for id, c in enumerate(categories_name):
            w.write('item {\n')
            w.write('  id: {}\n'.format(id+1))
            w.write('  name: "'"{}"'"\n'.format(c))
            w.write('}\n\n')


def make_DB_same_as_YOLO():
    table = pd.read_csv('labelsheet_v3.csv', encoding='utf-8')
    # remove NaN rows
    table.dropna(how='all', inplace=True)
    table.reset_index(drop=True, inplace=True)

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

    # anno_list is the list of all annotation path
    anno_list = [f for f in os.listdir(server['YOLO_DIR']) if f.endswith('.txt')]
    try:
        anno_list.remove(os.path.join(server['YOLO_DIR'], 'classes.txt'))
    except:
        print('no classes to remove')
    print("num of annotations = ", len(anno_list))

    # TODO class balance
    train_list, eval_list = split_data(anno_list, split_percentage=10)
    # record the two number & set them on training cmd
    print('num of train data =', len(train_list))
    print('num of eval data =', len(eval_list))

    # tf.initialize_all_variables().run()
    make_TFRecord_id_version(category_list, train_list, server['output_train_path'])
    make_TFRecord_id_version(category_list, eval_list, server['output_val_path'])


def main(_):
    make_DB_same_as_YOLO()


if __name__ == '__main__':
    tf.app.run()

