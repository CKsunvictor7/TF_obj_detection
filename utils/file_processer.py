import shutil
import os
import pandas as pd
from PIL import Image


def file_indexer(dir_path, category_name):
    """
    rename the file in dir from name_1 to name_N
    :param dir_path: the path of the dir
    :return: None
    """
    for idx, f in enumerate(os.listdir(dir_path)):
        original_file_path = os.path.join(dir_path, f)
        if os.path.isfile(original_file_path) and f.endswith(('.jpg', 'jpeg','.png', '.bmp', '.JPG', 'JPEG','.PNG', '.BMP')):
            # check the corrupt file
            try:
                img = Image.open(original_file_path)
                dst_file_path = os.path.join(dir_path, '{}_{}{}'.format(category_name, str(idx+1),os.path.splitext(f)[-1]))
                # print(dst_file_path)
                shutil.move(original_file_path, dst_file_path)
            except:
                print('{} is corrupt, remove it'.format(original_file_path))
                os.remove(original_file_path)


def indexer_for_all_dir(super_dir_path):
    for f in os.listdir(super_dir_path):
        if os.path.isdir(os.path.join(super_dir_path,f)):
            file_indexer(os.path.join(super_dir_path,f), f)


def list_nb_in_dir(dir_path):
    return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])


def list_nb_all_dir(super_dir_path, csv_file_name):
    """
    list nb of file for all directories in this super_dir_path

    :param super_dir_path:
    :return: None
    """
    # tuple nb_list = (dir, nb)
    nb = []
    dir_list = []
    for dir in os.listdir(super_dir_path):
        if os.path.isdir(os.path.join(super_dir_path, dir)):
            dir_list.append(dir)
            # only list the nb of folder 2(multiple food)
            nb.append(list_nb_in_dir(os.path.join(super_dir_path, dir, '1')))
            # nb.append(list_nb_in_dir(os.path.join(super_dir_path, dir, '2')))

    data = {'dir':dir_list, 'nb':nb}
    df = pd.DataFrame(data)
    df.sort_values(['nb'], ascending=False, inplace=True)
    print(df)
    df.to_csv(csv_file_name, index=False, header=True)
    print('nb of categories = ', len(dir_list))
    print('nb of total images = ', sum(nb))

"""
client_path = '/Users/fincep004/Desktop/food_imgs/zip'
server_path = '/mnt/dc/web_food_imgs/'
"""
def uploader(client_path, server_path):
    for f in os.listdir(client_path):
        if os.path.isfile(os.path.join(client_path, f)):
            print('now uploading {} ...'.format(f))
            os.system('scp {} {}:{}'.format(os.path.join(client_path, f), 'gpu4',
                                        os.path.join(server_path, f)))


def zipper(source_path, dst_path):
    for dir in os.listdir(source_path):
        if dir is not 'zip' and os.path.isdir(os.path.join(source_path, dir)):
            if not os.path.exists(os.path.join(dst_path, dir)):
                print(dir)
                shutil.make_archive(os.path.join(dst_path, dir), 'zip',
                                    os.path.join(source_path, dir))
            else:
                print('already zipped')


def mapping_EN_to_JP(file_path, super_dir_path):
    name_mapper = pd.read_csv(file_path)
    jp = name_mapper['grandchild']
    en = name_mapper['English_Name_v1']

    # rename all directory
    for i in range(len(jp)):
        src_path = os.path.join(super_dir_path, en[i])
        dst_path = os.path.join(super_dir_path, jp[i])
        if os.path.isdir(src_path):
            pass
            #print('{} -> {}'.format(en[i], jp[i]))
            # shutil.move(src_path, dst_path)
        else:
            print('{} does not exit'.format(src_path))

"""
does not exist in food category:
アーモンドチョコ, あんまん, ドライフルーツ入りシリアル, 
"""
def mapping_JP_to_EN(file_path, super_dir_path):
    name_mapper = pd.read_csv(file_path)
    jp = name_mapper['grandchild']
    en = name_mapper['English_Name_v1']

    # rename all directory
    for i in range(len(en)):
        src_path = os.path.join(super_dir_path, jp[i])
        dst_path = os.path.join(super_dir_path, en[i])
        if os.path.isdir(os.path.join(src_path)):
            print('{} -> {}'.format(jp[i], en[i]))
            shutil.move(src_path, dst_path)
        else:
            print('{} does not exist'.format(src_path))


def non_food_mover(list_path, super_dir_path):
    dst_dir = '/Users/fincep004/Desktop/non_food_imgs'
    with open(list_path, 'r') as r:
        counter = 0
        for f in r.readlines():
            if counter > 0:
                src_path = os.path.join(super_dir_path, f.rstrip())
                dst_path = os.path.join(dst_dir, os.path.basename(f.rstrip()))

                img = Image.open(src_path)
                if img.size[0] < 224 or img.size[1] < 224:
                    dst_path = os.path.join(dst_dir, '***-{}'.format(os.path.basename(f.rstrip())))
                shutil.copyfile(src_path, dst_path)

            counter += 1
            if counter > 2000:
                break


def get_category_name(s):
    ss = ''
    for x in s.split('.')[0].split('_')[:-1]:
        if ss == '':
            ss = x
        else:
            ss = ss + '_' + x
    return ss


def filelist_writer():
    # path = '/Users/fincep004/Desktop/nonfood_FN'
    # with open('nonFood_FN_cases.txt', 'a+') as w:
    path = '/Users/fincep004/Desktop/food_FP'
    with open('Food_FP_cases.txt', 'a+') as w:
        for f in os.listdir(path):
            category_name = get_category_name(f)
            w.write('{}\n'.format(os.path.join(category_name, f)))


def food_filter(dir_path, nonfood_list):
    with open(nonfood_list, 'r') as r:
        for line in r.readlines():
            os.remove(os.path.join(dir_path, line.rstrip()))



def main():
    path = '/mnt/dc/web_food_imgs_f/'
    list_nb_all_dir(path, 'nb_of_singlefood_web_food_f.csv')
    # list_nb_all_dir('/Users/fincep004/Desktop/food_imgs/')
    # mapping_JP_to_EN('/Users/fincep004/Desktop/food_category_v1.csv', path)
    # check
    # mapping_EN_to_JP('/Users/fincep004/Desktop/food_category_v1.csv', path)
    # reindex all images
    # indexer_for_all_dir(path)
    # zipper(path, '/Users/fincep004/Desktop/food_zip')
    #client_path = '/Users/fincep004/Desktop/food_zip/'
    #server_path = '/mnt/dc/web_food_imgs_zip/'
    #uploader(client_path, server_path)
    # non_food_mover('/Users/fincep004/PycharmProjects/ai_foodNonFood/annam/prediction_v3.csv', path)
    # filelist_writer()
    # file_indexer('/Users/fincep004/Desktop/web_imgs_f/ingredients_vegetable_soup', 'ingredients_vegetable_soup')
    # food_filter('/Users/fincep004/Desktop/web_imgs_f', 'nonfood_list')



if __name__ == '__main__':
    main()