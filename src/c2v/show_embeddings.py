import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from skipgram import SkipGram
from data.wili import CorpusReader
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    dataloader = CorpusReader("./data/wili-2018/x_train_sub.txt", "./data/wili-2018/y_train_sub.txt")
    char_to_idx, idx_to_char, char_frequency = dataloader.get_mappings()
    model = SkipGram(12300, 256, char_frequency)

    with open("./models/skipgram/5.pt", 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
        print("Model Loaded")

    save_embeddings = True
    if save_embeddings:
        central_embeddings = model.central_embedding.weight
        torch.save(central_embeddings, './models/character_embeddings.pt')
        print("{} Embedding Weights Saved".format(central_embeddings.shape))
    model = model.to(device)
    model.eval()

    similarities = model.vocabulary_similarities()
    show_chars = ['t', 'b', 'a', 'e', 'x', ',', '.', '@', '%', '4', '9', "բ", "Հ", "ñ", "名", "Θ"]
    show_results(show_chars, similarities, char_to_idx, idx_to_char)

if __name__ == '__main__':
    main()
