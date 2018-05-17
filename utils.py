import os


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


def main():
    print('')
    show_num_category_UEC()


if __name__ == '__main__':
    main()
