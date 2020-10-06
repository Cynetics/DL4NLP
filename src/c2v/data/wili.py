import numpy as np
import os
from collections import Counter
import torch
import random
import json

class CorpusReader():
    """
    Read the contents of a directory of files, and return the results as
    either a list of lines or a list of words.
    """

    def __init__(self, data_path, label_path, window_size=5):
        self.data_path = data_path
        self.label_path = label_path

        self.lines, self.line_languages = self.load_lines()
        self.lines = [char for line in self.lines for char in line]
        self.char_frequency = Counter(self.lines)

        self.languages = list(set(self.line_languages))
        self.languages.sort()
        print("Number of languages: ", len(self.languages))
        self.lang_to_idx = { l:i for i,l in enumerate(self.languages) }
        self.idx_to_lang = { i:l for i,l in enumerate(self.languages) }

        self.vocab_dict = json.load(open('./data/vocabs/full_vocab.json'))
        self.vocab_list = [key for key in self.vocab_dict]
        self.vocab_size = len(self.vocab_list)

        print("Vocabulary of size: {}".format(self.vocab_size))

        self.char_to_idx = { ch:i for i,ch in enumerate(self.vocab_list) }
        self.idx_to_char = { i:ch for i,ch in enumerate(self.vocab_list) }

        self.lines = [self.char_to_idx[char] for char in self.lines]

    def load_lines(self):
        """
        Each line is a list of characters belonging to a specific language.
        Skipping the "/n" token.

        """

        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        lines = [list(paragraph)[:-1] for paragraph in lines]

        with open(self.label_path, 'r') as f:
            languages = f.readlines()
        languages = [language[:-1] for language in languages]

        print('Loaded language paragraphs from: %s (%d)' % (self.data_path, len(lines)))
        return lines, languages

    def get_mappings(self):
        return self.char_to_idx, self.idx_to_char, self.char_frequency

    def gen_targets(self, word_list, idx, window_size=25):
        return word_list[idx-window_size: idx] + word_list[idx+1:idx+window_size+1]

    def gen_batches(self, batch_size, window_size=25):
        word_list = self.lines
        n_batches = len(word_list)//batch_size
        words = word_list[:n_batches*batch_size]

        for idx in range(0, len(words), batch_size):
            x, y = [], []
            batch = words[idx:idx+batch_size]
            for i in range(len(batch)):
                batch_x = batch[i]
                batch_y = self.gen_targets(batch, i, window_size)
                y.extend(batch_y)
                x.extend([batch_x]*len(batch_y))
            yield x, y
