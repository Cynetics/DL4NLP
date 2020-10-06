# -*- coding: utf-8 -*-
import numpy as np
import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.optim as optim

from LSTM import Model, KL
from data.load_data import get_wili_data, get_wili_data_bytes
from utils.config import LSTM_config
from utils.utils import validate_paragraphs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, training_loader, validation_loader, validation_data, config, model_name=""):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    kl = KL(divisor=25)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    avg_train_loss = []
    train_loss = []
    accuracy = 0
    for epoch in range(config.epochs):
        print("Starting Epoch: {}".format(epoch))
        for i, (inputs, labels) in enumerate(training_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            output = model(inputs)
            #log_alpha_1 = model._bayesian1._log_alpha
            log_alpha_2 = model._bayesian._log_alpha
            #kl1 = kl(log_alpha_1)
            kl2 = kl(log_alpha_2)
            kl_divergence = kl2

            loss = criterion(output, labels)
            #print("Loss: ", loss)

            loss += kl_divergence #/ 5
            #print("loss: ", loss)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            if  (i+1) % 42 == 0:
                print('%d iterations' % (i+1))
                avg = np.mean(train_loss[-50:])
                avg_train_loss.append(avg)
                print('Loss: %.3f' % avg)
                print()
                #accuracy = validate_paragraphs(model, validation_data, validation_loader, save_classification_report=False)
                #print("Current Accuracy: {}, After {} iterations in Epoch {}".format(accuracy, i, epoch))
                #model.train()
                #torch.save(model.state_dict(), "./models/LSTM/"+model_name+str(epoch)+"_"+str(accuracy)+".pt")

        scheduler.step()
        torch.save(model.state_dict(), "./models/LSTM/"+model_name+str(epoch)+"_"+str(config.batch_size)+"_"+str(config.input)+"_"+str(config.sequence_length)+".pt")
        #torch.save(model.state_dict(), "./models/LSTM/"+model_name+str(epoch)+"_"+str(accuracy)+".pt")
    #write_results((avg_train_loss, val_loss, val_accuracy), model_type+"_")
    print("Iterators Done")

def main():

    config = LSTM_config()

    if config.input == 'bytes':
        # Load Data for bytes
        training_data, validation_data = get_wili_data_bytes(config)
    else:
        # Load Data
        training_data, validation_data = get_wili_data(config)


    training_loader = DataLoader(training_data,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=False)

    validation_loader = DataLoader(validation_data,
                             batch_size=1,
                             shuffle=True,
                             drop_last=False)

    model = Model(config.input_dim, config.embedding_dim,
                  config.hidden_dim, config.num_layers, bidirectional=False)

    if config.model_checkpoint is not None:
        with open(config.model_checkpoint, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
            print("Model Loaded From: {}".format(config.model_checkpoint))
    model = model.to(device)
    train(model, training_loader, validation_loader, validation_data, config, model_name=config.model_name)


if __name__ == '__main__':
    main()
