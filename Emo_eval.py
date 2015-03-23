__author__ = 'v-penlu'
import EmoClassify as EC
import EmoClassify_kmax as ECK
import EmoClassify_full as ECF
from scipy.spatial.distance import cosine
import cPickle
import os
import numpy as np
import re


def Extract_feature(params, sample):
    if "out" in params:
        vec = ECF.Extract_feature(params, sample)
    else:
        if "average" in params:
            vec = EC.Extract_feature(params, sample)
        else:
            vec = ECK.Extract_feature(params, sample)
    return vec


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
    avg_vec = {}
    emo_vec = {}
    for emo, wav_list in data_dic.items():
        if len(selected_emo) != 0 and emo not in selected_emo:
            continue
        vec_l = []
        for wav in wav_list:
            csv = os.path.join(csv_dir, ("Session%d" % wav[0]), wav[1],
                               ("%s.csv" % wav[2]))
            arr = np.loadtxt(csv, delimiter=";", skiprows=1)
            dim = arr.shape[1]
            sample = [arr[i].reshape(dim, 1) for i in range(arr.shape[0])]
            vec = Extract_feature(params, sample)
            if not printed:
                print_header(arff, vec.shape[0], selected_emo)
                str_format = "%.6f," * vec.shape[0] + "%s"
                printed = True
            vec_l.append(vec)
            l = vec.tolist()
            l.append(emo)
            print >> arff, str_format % tuple(l)
        arr = np.asarray(vec_l)
        avg_vec[emo] = np.mean(arr, axis=0)
        emo_vec[emo] = vec_l
    arff.close()
    pkl_name = os.path.basename(arff_name)
    pkl_name = os.path.splitext(pkl_name)[0] + '_avg.pkl'
    pkl = open(pkl_name, "wb")
    cPickle.dump(avg_vec, pkl)
    pkl.close()
    return emo_vec, avg_vec


def calculate_error(emo_vec, avg_vec):
    assign = {}
    emo_list = emo_vec.keys()
    for emo in emo_list:
        assign[emo] = np.zeros((len(emo_list)), dtype=np.int)
        for vec in emo_vec[emo]:
            max_val = -1
            max_idx = -1
            for i, avg_emo in enumerate(emo_list):
                cos = 1 - cosine(vec, avg_vec[avg_emo])
                if cos > max_val:
                    max_val = cos
                    max_idx = i
            assign[emo][max_idx] += 1
    d_format = "%4d" * len(emo_list)
    header = "    " + "%4s" * 5
    accur = []
    print header % tuple(emo_list)
    for i, emo in enumerate(emo_list):
        print emo + (d_format % tuple(assign[emo].tolist()))
        accur.append(assign[emo][i] / float(np.sum(assign[emo])))
    for i, emo in enumerate(emo_list):
        print emo, accur[i]
    total_acc = 0
    total = 0
    for i, emo in enumerate(emo_list):
        total_acc += assign[emo][i]
        total += len(emo_vec[emo])
    print "total_accuracy", total_acc / float(total)


if __name__ == "__main__":
    test_dic_n = "dic_5emo1.pkl"
    name = "hidden-100-100_kmax-3_full-100_coscos"
    params_n = os.path.join("save_params", name + ".pkl")
    train_arff_n = name + "_train_emo_feature.arff"
    test_arff_n = name + "_test_emo_feature.arff"
    csv_dir = "D:\IEMOCAP_full_release\emobase"
    train_dic, valid_dic, test_dic = load_pkl(test_dic_n, 3)
    params = load_pkl(params_n)
    _, avg_vec = output_arff(train_dic, csv_dir, train_arff_n, test_dic.keys())
    emo_vec, _ = output_arff(test_dic, csv_dir, test_arff_n, test_dic.keys())
    calculate_error(emo_vec, avg_vec)
