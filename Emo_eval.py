__author__ = 'v-penlu'
import EmoClassify as EC
import cPickle
import os
import numpy as np
import re


def load_pkl(f_name):
    pkl = open(f_name, "rb")
    ret = cPickle.load(pkl)
    pkl.close()
    return ret


def print_header(arff, dim, emo_list):
    print >> arff, "@RELATION emo_classify\n"
    for i in range(dim):
        print >> arff, "@ATTRIBUTE feat_%d NUMERIC" % i
    class_str = ",".join(emo_list)
    class_str = "{" + class_str + "}"
    print >> arff, "@ATTRIBUTE class %s\n\n@DATA" % class_str



if __name__ == "__main__":
    test_dic_n = "test_dic.pkl"
    params_n = 'params_epoch3.pkl'
    arff_n = "test_emo_feature.arff"
    csv_dir = "D:\IEMOCAP_full_release\emobase"
    test_dic = load_pkl(test_dic_n)
    params = load_pkl(params_n)
    arff = open(arff_n, "w")
    printed = False
    emo_list = test_dic.keys()
    str_format = ""
    for emo, wav_list in test_dic.items():
        for wav in wav_list:
            csv = os.path.join(csv_dir, ("Session%d" % wav[0]), wav[1],
                               ("%s.csv" % wav[2]))
            arr = np.loadtxt(csv, delimiter=";", skiprows=1)
            dim = arr.shape[1]
            sample = [arr[i].reshape(dim, 1) for i in range(arr.shape[0])]
            vec = EC.Extract_feature(params, sample)
            if not printed:
                print_header(arff, vec.shape[0], emo_list)
                str_format = "%.6f," * vec.shape[0] + "%s"
                printed = True
            l = vec.tolist()
            l.append(emo)
            print >> arff, str_format % tuple(l)
    arff.close()
