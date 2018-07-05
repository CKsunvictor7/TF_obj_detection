import os
import shutil
from tools import get_file_list, labelmap_maker


def create_annotation_UEC(annotation_dir):
    """
    create feasible annotations(one txt with all bounding boxes info)
    """
    for i in range(1, 257):
        with open(os.path.join(os.path.sep, '/mnt/dc/UECFOOD256', str(i), 'bb_info.txt'), 'r') as r:
            r.readline()  # discard ths first line
            for line in r.readlines():
                arrs = line.rstrip().split(' ')
                with open(os.path.join(annotation_dir, arrs[0] + '.txt'), 'a+') as w:
                    w.write(
                        '{} {} {} {} {}\n'.format(i, arrs[1], arrs[2], arrs[3], arrs[4]))


def file_mover(src_dir, dst_dir):
    img_list = get_file_list(src_dir)
    print('nb of img = ', len(img_list))
    for f in img_list:
        shutil.copyfile(f, os.path.join(dst_dir, os.path.basename(f)))



def main():
    labelmap_maker(
        os.path.join(os.path.sep, '/mnt/dc', 'UECFOOD256', 'category.txt'),
        'UEC_label_map.pbtxt')

    print('done')


if __name__ == '__main__':
    main()
















