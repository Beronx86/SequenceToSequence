__author__ = 'v-penlu'
import cPickle
from random import  shuffle
import EmoClassify as EC
import numpy as np
from scipy.spatial.distance import cosine
import os


def Generate_ABXlist(rest_emo, train_emo, emo_dic, num):
    rest_emo_l = emo_dic[rest_emo]
    shuffle(rest_emo_l)
    X = rest_emo_l[:num]
    B = rest_emo_l[num:2 * num]
    train_emo_l = []
    for emo in train_emo:
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


def Extract_feature(params, in_seq, ext_mode):
    if ext_mode == "lmax":
        vec = EC.Extract_feature(params, in_seq)
    # elif ext_mode == "kmax":
    # elif ext_mode == "lmax_full":
    # elif ext_mode == "kmax_full":
    else:
        print "extraction mode error"
        exit(1)
    return vec


def Eval_list(ABX_list, csv_dir, params, ext_mode):
    total_cnt = len(ABX_list[0])
    right_cnt = 0
    for A, B, X in zip(ABX_list[0], ABX_list[1], ABX_list[2]):
        seq_A = Load_csv(csv_dir, A)
        seq_B = Load_csv(csv_dir, B)
        seq_X = Load_csv(csv_dir, X)
        vec_A = Extract_feature(params, seq_A, ext_mode)
        vec_B = Extract_feature(params, seq_B, ext_mode)
        vec_X = Extract_feature(params, seq_X, ext_mode)
        cos_AX = 1 - cosine(vec_A, vec_X)
        cos_BX = 1 - cosine(vec_B, vec_X)
        if cos_BX > cos_AX:
            right_cnt += 1
    return right_cnt / float(total_cnt)


if __name__ == '__main__':
    csv_dir = r"D:\IEMOCAP_full_release\emobase"
    num = 400
    params_n = r"save_params\hidden100-100-100_lmax3_params_epoch9.pkl"
    ext_mode = "lmax"
    rest_emo = r"hap"
    train_emo = ["neu", "sad", "fru", "exc", "ang"]
    # train_emo = ["neu", "sad", "fru", "exc", "hap"]
    emo_dic_n = r"dic.pkl"
    emo_dic_f = open(emo_dic_n, 'rb')
    emo_dic = cPickle.load(emo_dic_f)
    emo_dic_f.close()
    params_f = open(params_n, "rb")
    params = cPickle.load(params_f)
    params_f.close()
    ABX_list = Generate_ABXlist(rest_emo, train_emo, emo_dic, num)
    ABX_accuracy = Eval_list(ABX_list, csv_dir, params, ext_mode)
    print "%s accuracy %f" % (rest_emo, ABX_accuracy)
