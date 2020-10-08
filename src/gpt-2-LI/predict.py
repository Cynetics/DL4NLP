
from torch.utils.data import DataLoader
from data.load_data import get_wili_data, get_wili_data_bytes

import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.config import GPT2Config
from utils.utils import *
from utils.config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model, validation_loader, validation_data, config):
    model.eval()

    if config.prediction_type=="deterministic":
        accuracy = validate_paragraphs(model, validation_data, validation_loader, subset=False, config=config)
        print("Validation Accuracy: {}".format(accuracy))
    else:
        accuracy = validate_uncertainty(model, validation_data, validation_loader, config=config)
        print("Validation Accuracy: {}".format(accuracy))

    print("Iterators Done")

def main():

    param_config = config()
    gpt_config = GPT2Config(vocab_size_or_config_json_file=param_config.input_dim, n_positions=param_config.sequence_length, n_ctx=param_config.sequence_length)

    model = GPT2LMHeadModel(gpt_config)

    if param_config.input == 'bytes':
        # Load Data for bytes
        _, validation_data = get_wili_data_bytes(param_config)
    else:
        # Load Data
        _, validation_data = get_wili_data(param_config)

    validation_loader = DataLoader(validation_data,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False)

    if param_config.model_checkpoint is not None:
        with open(param_config.model_checkpoint, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
            print("Model Loaded From: {}".format(param_config.model_checkpoint))

    model = model.to(device)
    predict(model, validation_loader, validation_data, param_config)


if __name__ == '__main__':
    main()
