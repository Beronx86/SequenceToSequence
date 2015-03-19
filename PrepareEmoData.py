import os
from collections import defaultdict
import re
from random import shuffle, randint
import cPickle

path = r"D:\IEMOCAP_full_release"
session_cnt = 5
emo_dic = defaultdict(list)

# dict element [session_idx, wav_name, sen_name]
for i in range(1, 1 + session_cnt):
    emoeva_path = os.path.join(path, ("Session%d" % i), "dialog",
                               "EmoEvaluation")
    for f_name in os.listdir(emoeva_path):
        if f_name.endswith(".txt"):
            wav_name = os.path.splitext(f_name)[0]
            emoeva_f = os.path.join(emoeva_path, f_name)
            with open(emoeva_f, "r") as f:
                for line in f:
                    if re.search(wav_name, line):
                        s = line.split("\t")
                        emo = s[2]
                        sen_name = s[1]
                        emo_dic[emo].append([i, wav_name, sen_name])

# use_emo = ["neu", "sad", "fru", "exc", "ang"]   # 5emo1
# use_emo = ["neu", "sad", "fru", "exc", "hap"] # 5emo2
use_emo = ["neu", "sad", "fru", "exc", "hap", "ang"]  #6emo
# Pair Scheme
# Pos 5 class, Neg 10 class, each class random choose 1000 samples.
# Choose 50 from each emo class as valid set,
# each pair class random choose 50 samples.
# Choose 100 as test set form each class.

num_classes = len(emo)
test_dic = defaultdict(list)
valid_dic = defaultdict(list)
train_pair = []
trains_pos = 1000
# trains_neg = trains_pos / (num_classes * (num_classes - 1) / 2) * num_classes
trains_neg = trains_pos
valid_pair = []
valids_pos = 50
# valids_neg = valids_pos / (num_classes * (num_classes - 1) / 2) * num_classes
valids_neg = valids_pos


def generate_pair(emolist_1, emolist_2=0):
    """ :param emo_1:
    :param emo_2: if emo_2 == 0, generate positive sample
    :return:
    """
    if emolist_2 == 0:
        length = len(emolist_1)
        idx_1 = randint(0, length - 1)
        idx_2 = randint(0, length - 1)
        return [emolist_1[idx_1], emolist_1[idx_2], True]
    else:
        length_1 = len(emolist_1)
        length_2 = len(emolist_2)
        idx_1 = randint(0, length_1 - 1)
        idx_2 = randint(0, length_2 - 1)
        return [emolist_1[idx_1], emolist_2[idx_2], False]

for emo in use_emo:
    shuffle(emo_dic[emo])
    test_dic[emo] = emo_dic[emo][-100:]
    valid_dic[emo] = emo_dic[emo][-150:-100]
    emo_dic[emo] = emo_dic[emo][:-150]

for emo in use_emo:
    for i in range(trains_pos):
        train_pair.append(generate_pair(emo_dic[emo]))
    for i in range(valids_pos):
        valid_pair.append(generate_pair(valid_dic[emo]))
for i in range(len(use_emo)):
    for j in range(i + 1, len(use_emo)):
        for k in range(trains_neg):
            train_pair.append(generate_pair(emo_dic[use_emo[i]],
                                            emo_dic[use_emo[j]]))
        for k in range(valids_neg):
            valid_pair.append(generate_pair(emo_dic[use_emo[i]],
                                            emo_dic[use_emo[j]]))


tv_pkl_n = "train_valid_list_6emo.pkl"
tv_txt_n = "train_valid_list_6emo.txt"
dic_n = "dic_6emo.pkl"

tv_f = open(tv_pkl_n, "wb")
cPickle.dump(train_pair, tv_f)
cPickle.dump(valid_pair, tv_f)
tv_f.close()
t_f = open(dic_n, "wb")
cPickle.dump(emo_dic, t_f)
cPickle.dump(valid_dic, t_f)
cPickle.dump(test_dic, t_f)
t_f.close()

f = open(tv_txt_n, "w")
print >> f, "train_list", len(train_pair)
for p in train_pair:
    print >> f, p
print >> f, "valid_list", len(valid_pair)
for p in valid_pair:
    print >> f, p
f.close()
