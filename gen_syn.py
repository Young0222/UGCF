# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import io
import sys


# 执行完，再执行
# sort -n syn_train_orig.txt > train.txt
# sort -n syn_test_orig.txt > test.txt

train_file_output = "syn_train_orig.txt"
test_file_output = "syn_test_orig.txt"

if __name__ == '__main__':

    num_user, num_item = 1000, 50
    user_list = np.arange(1, num_user+1)
    item_list = np.arange(1, num_item+1)
    like_prob = np.arange(0.11, 0.61, 0.01)
    print("like_prob: ", like_prob)
    item_pos_list = np.arange(100, 0, -2)
    print("item_pos_list: ", item_pos_list)
    exp_list = item_pos_list / like_prob
    print("exp_list: ", exp_list)
    train_res_set = {}
    test_res_set = {}
    test_split_prob = 0.2

    for item in item_list:
        pair_user_list = np.random.choice(user_list, item_pos_list[item-1])
        # print("pair_user_list: ", pair_user_list)
        for user in pair_user_list:
            if random.uniform(0, 1) > test_split_prob:
                if user not in train_res_set:
                    train_res_set[user] = [item]
                elif item not in train_res_set[user]:
                    train_res_set[user].append(item)
            else:
                if user not in test_res_set:
                    test_res_set[user] = [item]
                elif item not in test_res_set[user]:
                    test_res_set[user].append(item)
    with io.open(train_file_output, 'w', encoding='utf-8') as fout:
        for user in train_res_set:
            fout.write(str(user))
            for item in train_res_set[user]:
                fout.write(" "+str(item))
            fout.write("\n")
    with io.open(test_file_output, 'w', encoding='utf-8') as fout:
        for user in test_res_set:
            fout.write(str(user))
            for item in test_res_set[user]:
                fout.write(" "+str(item))
            fout.write("\n")

