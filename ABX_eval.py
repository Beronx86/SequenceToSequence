__author__ = 'v-penlu'
import cPickle
from random import  shuffle
import EmoClassify as EC
import numpy as np
from scipy.spatial.distance import cosine
import math
import os
import Emo_eval as EE


def Generate_ABXlist(test_emo, train_emo, emo_dic, num):
    rest_emo_l = emo_dic[test_emo]
    shuffle(rest_emo_l)
    X = rest_emo_l[:num]
    B = rest_emo_l[num:2 * num]
    train_emo_l = []
    for emo in train_emo:
        if emo != test_emo:
            train_emo_l.extend(emo_dic[emo])
    shuffle(train_emo_l)
    A = train_emo_l[:num]
    return A, B, X


def Load_csv(csv_dir, name_tuple):
    csv = os.path.join(csv_dir, ("Session%d" % name_tuple[0]), name_tuple[1],
                       ("%s.csv" % name_tuple[2]))
    arr = np.loadtxt(csv, delimiter=";", skiprows=1)
    dim = arr.shape[1]
    sample = [arr[i].reshape(dim, 1) for i in range(arr.shape[0])]
    return sample


def Eval_list(ABX_list, csv_dir, params, ext_mode):
    total_cnt = len(ABX_list[0])
    right_cnt = 0
    for A, B, X in zip(ABX_list[0], ABX_list[1], ABX_list[2]):
        seq_A = Load_csv(csv_dir, A)
        seq_B = Load_csv(csv_dir, B)
        seq_X = Load_csv(csv_dir, X)
        vec_A = EE.Extract_feature(params, seq_A)
        vec_B = EE.Extract_feature(params, seq_B)
        vec_X = EE.Extract_feature(params, seq_X)
        cos_AX = 1 - cosine(vec_A, vec_X)
        cos_BX = 1 - cosine(vec_B, vec_X)
        if cos_BX > cos_AX:
            right_cnt += 1
    return right_cnt / float(total_cnt)


def cosine_2(vec_1, vec_2):
    vec_1_pow = vec_1.dot(vec_1)
    vec_2_pow = vec_2.dot(vec_2)
    vec_1_norm = math.sqrt(vec_1_pow)
    vec_2_norm = math.sqrt(vec_2_pow)
    vec_1_2_p = vec_1.dot(vec_2)
    cos = vec_1_2_p / (vec_1_norm * vec_2_norm)
    return cos


if __name__ == '__main__':
    csv_dir = r"D:\IEMOCAP_full_release\emobase"
    num = 400
    params_n = r"save_params\hidden100-100-100_lmax3_params_epoch9.pkl"
    ext_mode = "lmax"
    test_emo = r"hap"
    train_emo = ["neu", "sad", "fru", "exc", "ang"]
    # train_emo = ["neu", "sad", "fru", "exc", "hap"]
    emo_dic_n = r"dic.pkl"
    emo_dic_f = open(emo_dic_n, 'rb')
    emo_dic = cPickle.load(emo_dic_f)
    emo_dic_f.close()
    params_f = open(params_n, "rb")
    params = cPickle.load(params_f)
    params_f.close()
    ABX_list = Generate_ABXlist(test_emo, train_emo, emo_dic, num)
    ABX_accuracy = Eval_list(ABX_list, csv_dir, params, ext_mode)
    print "%s accuracy %f" % (test_emo, ABX_accuracy)
    # A = emo_dic["neu"][8]
    # B = emo_dic["ang"][0]
    # X = emo_dic["neu"][5]
    # seq_A = Load_csv(csv_dir, A)
    # seq_B = Load_csv(csv_dir, B)
    # seq_X = Load_csv(csv_dir, X)
    # vec_A = Extract_feature(params, seq_A, ext_mode)
    # vec_B = Extract_feature(params, seq_B, ext_mode)
    # vec_X = Extract_feature(params, seq_X, ext_mode)
    # cos_AX = 1 - cosine(vec_A, vec_X)
    # cos_BX = 1 - cosine(vec_B, vec_X)
    # cos_AB = 1 - cosine(vec_A, vec_B)
    # print cos_AX, cos_BX, cos_AB
