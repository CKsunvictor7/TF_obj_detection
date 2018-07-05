import os
import numpy as np

UEC256 = ['rice', 'eels on rice', 'pilaf', "chicken-'n'-egg on rice", 'pork cutlet on rice', 'beef curry',
          'sushi', 'chicken rice', 'fried rice', 'tempura bowl', 'bibimbap', 'toast', 'croissant', 'roll bread',
          'raisin bread', 'chip butty', 'hamburger', 'pizza', 'sandwiches', 'udon noodle', 'tempura udon', 'soba noodle',
          'ramen noodle', 'beef noodle', 'tensin noodle', 'fried noodle', 'spaghetti', 'Japanese-style pancake',
          'takoyaki', 'gratin', 'sauteed vegetables', 'croquette', 'grilled eggplant', 'sauteed spinach',
          'vegetable tempura', 'miso soup', 'potage', 'sausage', 'oden', 'omelet', 'ganmodoki', 'jiaozi', 'stew',
          'teriyaki grilled fish', 'fried fish', 'grilled salmon', 'salmon meuniere', 'sashimi', 'grilled pacific saury',
          'sukiyaki', 'sweet and sour pork', 'lightly roasted fish', 'steamed egg hotchpotch', 'tempura', 'fried chicken',
          'sirloin cutlet', 'nanbanzuke', 'boiled fish', 'seasoned beef with potatoes', 'hambarg steak', 'steak', 'dried fish',
          'ginger pork saute', 'spicy chili-flavored tofu', 'yakitori', 'cabbage roll', 'omelet', 'egg sunny-side up',
          'natto', 'cold tofu', 'egg roll', 'chilled noodle', 'stir-fried beef and peppers', 'simmered pork',
          'boiled chicken and vegetables', 'sashimi bowl', 'sushi bowl', 'fish-shaped pancake with bean jam',
          'shrimp with chill source', 'roast chicken', 'steamed meat dumpling', 'omelet with fried rice', 'cutlet curry',
          'spaghetti meat sauce', 'fried shrimp', 'potato salad', 'green salad', 'macaroni salad',
          'Japanese tofu and vegetable chowder', 'pork miso soup', 'chinese soup', 'beef bowl', 'kinpira-style sauteed burdock',
          'rice ball', 'pizza toast', 'dipping noodles', 'hot dog', 'french fries', 'mixed rice', 'goya chanpuru',
          'green curry', 'okinawa soba', 'mango pudding', 'almond jelly', 'jjigae', 'dak galbi', 'dry curry', 'kamameshi',
          'rice vermicelli', 'paella', 'tanmen', 'kushikatu', 'yellow curry', 'pancake', 'champon', 'crape', 'tiramisu',
          'waffle', 'rare cheese cake', 'shortcake', 'chop suey', 'twice cooked pork', 'mushroom risotto', 'samul', 'zoni',
          'french toast', 'fine white noodles', 'minestrone', 'pot au feu', 'chicken nugget', 'namero', 'french bread',
          'rice gruel', 'broiled eel bowl', 'clear soup', 'yudofu', 'mozuku', 'inarizushi', 'pork loin cutlet',
          'pork fillet cutlet', 'chicken cutlet', 'ham cutlet', 'minced meat cutlet', 'thinly sliced raw horsemeat', 'bagel',
          'scone', 'tortilla', 'tacos', 'nachos', 'meat loaf', 'scrambled egg', 'rice gratin', 'lasagna', 'Caesar salad',
          'oatmeal', 'fried pork dumplings served in soup', 'oshiruko', 'muffin', 'popcorn', 'cream puff', 'doughnut',
          'apple pie', 'parfait', 'fried pork in scoop', 'lamb kebabs', 'dish consisting of stir-fried potato, eggplant and green pepper',
          'roast duck', 'hot pot', 'pork belly', 'xiao long bao', 'moon cake', 'custard tart', 'beef noodle soup', 'pork cutlet',
          'minced pork rice', 'fish ball soup', 'oyster omelette', 'glutinous oil rice', 'trunip pudding', 'stinky tofu',
          'lemon fig jelly', 'khao soi', 'Sour prawn soup', 'Thai papaya salad',
          'boned, sliced Hainan-style chicken with marinated rice', 'hot and sour, fish and vegetable ragout',
          'stir-fried mixed vegetables', 'beef in oyster sauce', 'pork satay', 'spicy chicken salad', 'noodles with fish curry',
          'Pork Sticky Noodles', 'Pork with lemon', 'stewed pork leg', 'charcoal-boiled pork neck', 'fried mussel pancakes',
          'Deep Fried Chicken Wing', 'Barbecued red pork in sauce with rice', 'Rice with roast duck', 'Rice crispy pork',
          'Wonton soup', 'Chicken Rice Curry With Coconut', 'Crispy Noodles', 'Egg Noodle In Chicken Yellow Curry',
          'coconut milk soup', 'pho', 'Hue beef rice vermicelli soup', 'Vermicelli noodles with snails', 'Fried spring rolls',
          'Steamed rice roll', 'Shrimp patties', 'ball shaped bun with pork', 'Coconut milk-flavored crepes with shrimp and beef',
          'Small steamed savory rice pancake', 'Glutinous Rice Balls', 'loco moco', 'haupia', 'malasada', 'laulau', 'spam musubi',
          'oxtail soup', 'adobo', 'lumpia', 'brownie', 'churro', 'jambalaya', 'nasi goreng', 'ayam goreng', 'ayam bakar',
          'bubur ayam', 'gulai', 'laksa', 'mie ayam', 'mie goreng', 'nasi campur', 'nasi padang', 'nasi uduk', 'babi guling',
          'kaya toast', 'bak kut teh', 'curry puff', 'chow mein', 'zha jiang mian', 'kung pao chicken', 'crullers',
          'eggplant with garlic sauce', 'three cup chicken', 'bean curd family style', 'salt & pepper fried shrimp with shell',
          'baked salmon', 'braised pork meat ball with napa cabbage', 'winter melon soup', 'steamed spareribs', 'chinese pumpkin pie',
          'eight treasure rice', 'hot & sour soup']


folder_path = os.path.join(os.path.sep, 'mnt', 'diet', 'UECFOOD256')


def get_file_list(dir_path):
    """
    return abs_path of all files who ends with __ in all sub-directories as a list
    """
    file_list = []
    for f in os.listdir(dir_path):
        path = os.path.join(dir_path, f)
        if os.path.isdir(path):
            file_list = file_list + get_file_list(path)
        elif f.endswith(('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG', '.PNG', '.BMP')):
            file_list.append(path)

    return file_list


def show_num_category_UEC():
    nums = []
    for i in range(1, 257):
        sub_f_path = os.path.join(folder_path, str(i))
        tmp_f = os.listdir(sub_f_path)
        nums.append(len(tmp_f))
    nums_names = {}
    for idx,i in enumerate(nums):
        nums_names.setdefault(UEC256[idx],i)

    print(sorted(nums_names.items(), key=lambda d:d[1],reverse=False))


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


def labelmap_maker(category_path, labelmap_path):
    """
    make label_map using categories
    """
    with open(labelmap_path, 'w') as writer:
        with open(category_path, 'r') as reader:
            reader.readline() # skip first line
            for id, l in enumerate(reader.readlines()):
                category = l.rstrip().split('	')[1]
                print('item {')
                print('  id: {}'.format(id+1))
                print('  name: '"'{}'"''.format(category))
                print('}\n')
                writer.write('item {\n')
                writer.write('  id: {}\n'.format(id+1))
                writer.write('  name: '"'{}'"'\n'.format(category))
                writer.write('}\n\n')




def main():
    print('')



if __name__ == '__main__':
    main()
