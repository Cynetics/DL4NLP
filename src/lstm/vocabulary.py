# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Module for creating vocabulary for given file(s).
"""

import os.path
import argparse
import json
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("-dir_in", default="./data/wili-2018/",
                    help="directory of input file")
parser.add_argument("-dir_out", default="./data/vocabs/",
                    help="directory any output is saved in")
parser.add_argument("-input", default="x_train.txt, x_test.txt",
                    help="name of input file or files, separated by ,")
parser.add_argument("-output", default="vocab.json",
                    help="name of output file")
args = parser.parse_args()

input_files = args.input.split(",")
vocab = Counter()

for file_name0 in input_files:
    file_name = args.dir_in + file_name0
    if not os.path.isfile(file_name):
        raise ValueError("Given input file cannot be found.")

    file = open(file_name, "r")
    file_content = file.read()

    # split contents of input files by character while removing all new lines
    all_chars = [char for char in file_content if char != "\n"]
    vocab.update(all_chars)

# save vocabulary as json file
output_destination = args.dir_out + args.output
with open(output_destination, "w") as outfile:
    json.dump(vocab, outfile)

##### Example for loading vocabulary from file
# with open('out/vocab.json') as json_file:
#     data = Counter(json.load(json_file))
