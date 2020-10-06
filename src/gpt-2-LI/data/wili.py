import os, sys

import random
import torch
import numpy  as np
from torch.utils.data import Dataset
import torch.utils.data as data
import json

class WiliDataLoader(Dataset):
    def __init__(self, data_path, label_path, sequence_length=32, predict=False, predict_offset=20):

        self.data_path = data_path
        self.label_path = label_path
        self.sequence_length = sequence_length

        self.lines, self.line_languages = self.load_lines()

        self.languages = list(set(self.line_languages))
        self.languages.sort()
        print("Number of languages: ", len(self.languages))
        self.lang_to_idx = { l:i for i,l in enumerate(self.languages) }
        self.idx_to_lang = { i:l for i,l in enumerate(self.languages) }

        self.vocab_dict = json.load(open('./data/vocabs/full_vocab.json', encoding='utf-8'))
        self.vocab_list = [key for key in self.vocab_dict]
        self.vocab_size = len(self.vocab_list)

        print("Vocabulary of size: {}".format(self.vocab_size))

        self.char_to_idx = { ch:i for i,ch in enumerate(self.vocab_list) }
        self.idx_to_char = { i:ch for i,ch in enumerate(self.vocab_list) }

        self.predict = predict
        self.prediction_offset = predict_offset


    def load_lines(self):
        """
        Each line is a list of characters belonging to a specific language.
        Skipping the "/n" token.

        """

        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [list(paragraph)[:-1] for paragraph in lines]

        with open(self.label_path, 'r', encoding='utf-8') as f:
            languages = f.readlines()
        languages = [language[:-1] for language in languages]

        print('Loaded language paragraphs from: %s (%d)' % (self.data_path, len(lines)))


        return lines, languages

    def predict_paragraph(self, predict=False):
        self.predict = predict

    def __getitem__(self, index):
        """
        Get the paragraph (line) and corresponding language.
        Get a random sequence of sequence length within that paragraph.

        """
        paragraph, language = self.lines[index], self.line_languages[index]
        paragraph_length = len(paragraph)
        inputs = []; target = [];
        if not self.predict:

            offset = np.random.randint(0, paragraph_length-self.sequence_length)
            inputs = np.array([self.char_to_idx[ch] for ch in paragraph[offset:offset+self.sequence_length]])
            target = self.lang_to_idx[language]

        else:
            offset = self.prediction_offset
            offset_space = paragraph_length - self.sequence_length
            #n_sequences = offset_space // offset

            for i in range(0,offset_space, offset):
                if i > 5000:
                    break
                inputs.append(np.array([self.char_to_idx[ch] for ch in paragraph[i:i+self.sequence_length]]))
                target.append(self.lang_to_idx[language])

            inputs = np.array(inputs); target = np.array(target);

        return inputs, target


    def __len__(self):
        return len(self.lines)


class WiliBytesDataLoader(Dataset):
    def __init__(self, data_path, label_path, sequence_length=30, n_slices=8, predict=False, predict_offset=10):

        self.data_path = data_path
        self.label_path = label_path
        self.sequence_length = sequence_length

        self.lines, self.line_languages = self.load_lines()

        self.languages = sorted(list(set(self.line_languages)))
        print("Number of languages: ", len(self.languages))

        self.lang_to_idx = { l:i for i,l in enumerate(self.languages) }
        self.idx_to_lang = { i:l for i,l in enumerate(self.languages) }

        self.vocab_dict = json.load(open('./data/vocabs/full_bytes_vocab.json', encoding='utf-8'))
        self.vocab_list = [key for key in self.vocab_dict]
        self.vocab_size = len(self.vocab_list)

        print("Vocabulary of size: {}".format(self.vocab_size))

        self.predict = False
        self.prediction_offset = predict_offset


    def load_lines(self):
        """
        Each line is a list of integers that represent subwords
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [[int(ch) for ch in paragraph.split()] for paragraph in lines]

        with open(self.label_path, 'r', encoding='utf-8') as f:
            languages = f.readlines()
        languages = [language[:-1] for language in languages]
        #languages = ['eng' if lan == 'eng' else 'noneng' for lan in languages]

        print('Loaded language paragraphs from: %s (%d)' % (self.data_path, len(lines)))

        return lines, languages


    def __oldgetitem__(self, index):
        """
        Get the paragraph (line) and corresponding language.
        Get a random sequence of sequence length within that paragraph.
        """
        paragraph, language = self.lines[index], self.line_languages[index]
        paragraph_length = len(paragraph)

        offset = np.random.randint(0, paragraph_length-self.sequence_length)
        inputs = np.array([ch for ch in paragraph[offset:offset+self.sequence_length]])
        target = self.lang_to_idx[language]

        return inputs, target

    def predict_paragraph(self, predict=False):
        self.predict = predict

    def __getitem__(self, index):
        """
        Get the paragraph (line) and corresponding language.
        Get a random sequence of sequence length within that paragraph.
        """
        paragraph, language = self.lines[index], self.line_languages[index]
        paragraph_length = len(paragraph)
        inputs = []; target = [];
        if not self.predict:

            offset = np.random.randint(0, paragraph_length-self.sequence_length)
            inputs = np.array([ch for ch in paragraph[offset:offset+self.sequence_length]])
            target = self.lang_to_idx[language]

        else:
            offset = self.prediction_offset
            offset_space = paragraph_length - self.sequence_length
            #n_sequences = offset_space // offset

            for i in range(0,offset_space, offset):
                if i > 5000:
                    break
                inputs.append(np.array([ch for ch in paragraph[i:i+self.sequence_length]]))
                target.append(self.lang_to_idx[language])
            inputs = np.array(inputs); target = np.array(target);

        return inputs, target #, index


    def __len__(self):
        return len(self.lines)
