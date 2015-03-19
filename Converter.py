__author__ = 'v-penlu'

import argparse
import re
import os

# parser = argparse.ArgumentParser()
# parser.add_argument("in_file", help="Specify the input file name")
# parser.add_argument("out_file", help="Specify the output file name")
# args = parser.parse_args(["test.arff", "test_elm.txt"])


def convert(in_file, out_file):
    in_f = open(in_file, 'r')
    class_dic = {}
    data_started = False
    out_f = open(out_file, "w")
    for line in in_f:
        if data_started:
            line = line.strip()
            if line != "":
                old = line.split(",")
                new = old[:-1]
                new.insert(0, class_dic[old[-1]])
                new_line = " ".join(new)
                print >> out_f, new_line
        elif re.match("@ATTRIBUTE class", line):
            class_str = line.split()[-1]
            class_str = class_str[1:-1]
            for i, c in enumerate(class_str.split(",")):
                class_dic[c] = str(i)
        elif line.strip() == "@DATA":
            data_started = True
    in_f.close()
    out_f.close()


elm_dir = "elm_data"
if not os.path.exists("elm_data"):
    os.mkdir(elm_dir)
for f in os.listdir("."):
    if f.endswith(".arff"):
        out_name = os.path.splitext(f)[0]
        out_name = os.path.basename(out_name)
        out_name = os.path.join(elm_dir, out_name + "_elm.txt")
        convert(f, out_name)