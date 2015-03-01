__author__ = 'v-penlu'

import os
from collections import Counter
import pickle
import string
import numpy as np


def LearnVocab(data_file, size=0, thres=0):
    """Learn vocabulary form data_file. Designate size or threshold to determine
    the vocab size
    :param data_file:
    :param size: if size is not 0, learn a vocabulary which contains "size"
           words.
    :param thres: if thres is not 0, detain the words which appears more than
           "thres" times.
    :return:
    """
    word_cnt = Counter()
    f = open(data_file, "r")
    for l in f:
        s = l.split()
        if s[-1] in string.punctuation:
            s.pop()
        s.append("<EOS>")
        for v in s:
            word_cnt[v] += 1
    f.close()
    if size > 0:
        vocab = word_cnt.most_common(size)
    elif thres > 0:
        i = 0
        for word, cnt in word_cnt.iteritems():
            if cnt < thres:
                break
            i += 1
        vocab = word_cnt.most_common(i)
    else:
        print "Params error: size and thres can not be 0 at the same time"
        exit(1)
    file_name = os.path.basename(data_file)
    vocab_name = file_name + ".vocab"
    f = open(vocab_name, "w")
    for word, cnt in vocab:
        print >> f, word, cnt
    f.close()
    return vocab


def Word2Idx(vocab):
    for i, (word, cnt) in enumerate(vocab):
        if word == "<EOS>":
            del vocab[i]
            break
    vocab.insert(0, ("<UNK>", 2))
    vocab.insert(0, ("<EOS>", 1))
    vocab.insert(0, ("<NUL>", 0))
    word_idx = dict()
    idx_word = dict()
    for i in range(len(vocab)):
        word_idx[vocab[i][0]] = i
        idx_word[i] = vocab[i][0]
    return word_idx, idx_word


def Sen2Idx(data_file, word_idx):
    f = open(data_file, "r")
    data = []
    for l in f:
        sen = []
        s = l.split()
        if s[-1] in string.punctuation:
            s.pop()
        s.append("<EOS>")
        for word in s:
            idx = word_idx.get(word, 2)  # 2 is <UNK>
            sen.append(idx)
        data.append(sen)
    return data


def ReadFile(data_file, vocab_size):
    vocab = LearnVocab(data_file, vocab_size)
    word_idx, idx_word = Word2Idx(vocab)
    sens = Sen2Idx(data_file, word_idx)
    return sens, word_idx, idx_word


def SortSamples(src_sens, des_sens):
    len_l = []
    for s, d in zip(src_sens, des_sens):
        len_l.append(len(s) + len(d))
    sorted_idx = [i[0] for i in sorted(enumerate(len_l), key=lambda x: x[1])]
    sorted_src = [src_sens[i] for i in sorted_idx]
    sorted_des = [des_sens[i] for i in sorted_idx]
    return sorted_src, sorted_des, sorted_idx


def FilterBad(sorted_src, sorted_des):
    pre_src = []
    pre_des = []
    deleted_idx = []
    length = len(sorted_src)
    for i in reversed(range(length)):
        cur_src = sorted_src[i]
        cur_des = sorted_des[i]
        if len(cur_src) == 1 or len(cur_des) == 1:
            del sorted_src[i]
            del sorted_des[i]
            deleted_idx.append(i)
        elif (len(pre_src) == len(cur_src) and len(pre_des) == len(cur_des) and
              pre_src == cur_src and pre_des == cur_des):
            del sorted_src[i]
            del sorted_des[i]
            deleted_idx.append(i)
        else:
            pre_src = cur_src
            pre_des = cur_des
    return deleted_idx


def ReadWithoutVocab(prefix, save_name):
    src_file = prefix + ".src"
    print "Reading source file:", src_file
    src_sens, src_dic, src_idx_word = ReadFile(src_file, 160000)
    des_file = prefix + ".des"
    print "Reading destiantion file:", des_file
    des_sens, des_dic, des_idx_word = ReadFile(des_file, 80000)
    print "Constructing samples"
    txt_path = save_name + ".txt"
    f = open(txt_path, "w")
    sorted_src, sorted_des, sorted_idx = SortSamples(src_sens, des_sens)
    print "Removing bad samples"
    deleted_idx = FilterBad(sorted_src, sorted_des)
    print "%d samples removed" % len(deleted_idx)
    for src, des in zip(sorted_src, sorted_des):
        ins = " ".join(map(str, src))
        outs = " ".join(map(str, des))
        print >> f, "%s|%s" % (ins, outs)
    f.close()
    file_name = os.path.basename(prefix)
    dic_name = file_name + "_dicts" + ".pkl"
    f = open(dic_name, 'w')
    pickle.dump(src_dic, f)
    pickle.dump(des_dic, f)
    pickle.dump(src_idx_word, f)
    pickle.dump(des_idx_word, f)
    f.close
    return src_dic, des_dic


def ReadWithVocab(prefix, save_name, src_dic, des_dic):
    src_file = prefix + ".src"
    src_sens = Sen2Idx(src_file, src_dic)
    des_file = prefix + ".des"
    des_sens = Sen2Idx(des_file, des_dic)
    txt_path = save_name + ".txt"
    f = open(txt_path, "w")
    sorted_src, sorted_des, sorted_idx = SortSamples(src_sens, des_sens)
    FilterBad(sorted_src, sorted_des)
    for src, des in zip(sorted_src, sorted_des):
        ins = " ".join(map(str, src))
        outs = " ".join(map(str, des))
        print >> f, "%s|%s" % (ins, outs)
    f.close()


if __name__ == "__main__":
    train_prefix = r"D:\WMT\bitexts.pc2\wmt_all_pc2"
    valid_prefix = r"D:\WMT\dev+test\ntst1213"
    test_prefix = r"D:\WMT\dev+test\ntst14"
    src_dic, des_dic = ReadWithoutVocab(train_prefix, "train")
    print "Reading valid file..."
    ReadWithVocab(valid_prefix, "valid", src_dic, des_dic)
    print "Reading test file..."
    ReadWithVocab(test_prefix, "test", src_dic, des_dic)
