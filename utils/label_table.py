"""
tools for label conversion

1. label <-> JP <-> EN

2. hierarchical L1, L2 merging function

3. check nb of data of each category


"""
import pandas as pd
import os
import numpy as np
from merge_lookup_table import merged_table

df = pd.read_csv('labelsheet_v1')


def look_up(key, col):
    return df[df[col] == key]



def main():
    #print(look_up('petit_tomatoes', 'en'))
    keyword = 'candy'
    if keyword in merged_table:
        print(merged_table[keyword])

if __name__ == '__main__':
    main()
