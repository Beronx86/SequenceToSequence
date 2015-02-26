__author__ = 'v-penlu'

import os
from collections import Counter
import pickle
import string


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
    vocab.insert(0, ("<UNK>", 1))
    vocab.insert(0, ("<EOS>", 0))
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
            idx = word_idx.get(word, 1)  # 1 is <UNK>
            sen.append(idx)
        data.append(sen)
    return data


def ReadFile(data_file, vocab_size):
    vocab = LearnVocab(data_file, vocab_size)
    word_idx, idx_word = Word2Idx(vocab)
    f = open("word_idx.pkl", "wb")
    pickle.dump(word_idx, f)
    pickle.dump(idx_word, f)
    f.close()
    sens = Sen2Idx(data_file, word_idx)
    return sens


def ConsturctSamples(src_sens, des_sens):
    samples = []
    for src, des in zip(src_sens, des_sens):
        sample = []
        in_sen = src[:-1]
        out_sen = des[:-1]
        out_sen.insert(0, 0)  # insert <EOS> at position 0
        target_sen = des
        sample.append(in_sen)
        sample.append(out_sen)
        sample.append(target_sen)
        samples.append(sample)
    return samples


def ReadWithoutVocab(prefix, save_name):
    src_file = prefix + ".src"
    print "Reading source file:", src_file
    src_sens = ReadFile(src_file, 160000)
    des_file = prefix + ".des"
    print "Reading destiantion file:", des_file
    des_sens = ReadFile(des_file, 80000)
    print "Constructing samples"
    samples = ConsturctSamples(src_sens, des_sens)
    samples_path = save_name + ".pkl"
    f = open(samples_path, "wb")
    pickle.dump(samples, f)
    f.close()
    txt_path = save_name + ".txt"
    f = open(txt_path, "w")
    for s in samples:
        for ss in s:
            print >> f, ss
        print >> f
    f.close()


if __name__ == "__main__":
    train_prefix = r"D:\WMT\bitexts.pc4\wmt_all_pc4"
    ReadWithoutVocab(train_prefix, "train")
