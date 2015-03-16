__author__ = 'v-penlu'
import EmoClassify as EC
import cPickle
import os
import numpy as np
import re


def load_pkl(f_name, num=1):
    pkl = open(f_name, "rb")
    ret = []
    for i in range(num):
        ret.append(cPickle.load(pkl))
    pkl.close()
    if num == 1:
        return ret[0]
    else:
        return ret


def print_header(arff, dim, emo_list):
    print >> arff, "@RELATION emo_classify\n"
    for i in range(dim):
        print >> arff, "@ATTRIBUTE feat_%d NUMERIC" % i
    class_str = ",".join(emo_list)
    class_str = "{" + class_str + "}"
    print >> arff, "@ATTRIBUTE class %s\n\n@DATA" % class_str


def output_arff(data_dic, csv_dir, arff_name, selected_emo=[]):
    arff = open(arff_name, "w")
    printed = False
    str_format = ""
    for emo, wav_list in data_dic.items():
        if len(selected_emo) != 0 and emo not in selected_emo:
            continue
        for wav in wav_list:
            csv = os.path.join(csv_dir, ("Session%d" % wav[0]), wav[1],
                               ("%s.csv" % wav[2]))
            arr = np.loadtxt(csv, delimiter=";", skiprows=1)
            dim = arr.shape[1]
            sample = [arr[i].reshape(dim, 1) for i in range(arr.shape[0])]
            vec = EC.Extract_feature(params, sample)
            if not printed:
                print_header(arff, vec.shape[0], selected_emo)
                str_format = "%.6f," * vec.shape[0] + "%s"
                printed = True
            l = vec.tolist()
            l.append(emo)
            print >> arff, str_format % tuple(l)
    arff.close()


if __name__ == "__main__":
    test_dic_n = "dic.pkl"
    params_n = 'params_epoch3.pkl'
    train_arff_n = "train_emo_feature.arff"
    test_arff_n = "test_emo_feature.arff"
    csv_dir = "D:\IEMOCAP_full_release\emobase"
    train_dic, valid_dic, test_dic = load_pkl(test_dic_n, 3)
    params = load_pkl(params_n)
    output_arff(train_dic, csv_dir, train_arff_n, test_dic.keys())
    output_arff(test_dic, csv_dir, test_arff_n)
