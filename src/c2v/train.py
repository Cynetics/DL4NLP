import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.optim as optim
from skipgram import SkipGram, Loss
from data.wili import CorpusReader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=8e-4)
    criterion = Loss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    avg_train_loss = []
    train_loss = []
    accuracy = 0
    for epoch in range(10):
        for i, (inputs, targets) in enumerate(dataloader.gen_batches(2048)):
            inputs, targets = torch.LongTensor(inputs).to(device), torch.LongTensor(targets).to(device)
            batch_size = inputs.shape[0]

            #Forward pass
            central_embedding = model.forward_input(inputs)
            context_embedding = model.forward_output(targets)
            noise_embedding = model.forward_noise(batch_size)

            #Compute the loss
            loss = criterion(central_embedding, context_embedding, noise_embedding)
            #Zero the gradients, perform a backward pass and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if  (i+1) % 100 == 0:
                print('%d iterations' % (i+1))
                avg = np.mean(train_loss[-500:])
                avg_train_loss.append(avg)
                print('Loss: %.3f' % avg)
                torch.save(model.state_dict(), "./models/skipgram/"+str(epoch)+".pt")

        scheduler.step()
        torch.save(model.state_dict(), "./models/skipgram/"+str(epoch)+".pt")
    print("Iterators Done")

def main():

    dataloader = CorpusReader("./data/wili-2018/x_train_sub.txt", "./data/wili-2018/y_train_sub.txt")
    _, _, char_frequency = dataloader.get_mappings()
    model = SkipGram(12300, 256, char_frequency)

    #if config.model_checkpoint is not None:
    #    with open(config.model_checkpoint, 'rb') as f:
    #        state_dict = torch.load(f)
    #        model.load_state_dict(state_dict)
    #        print("Model Loaded From: {}".format(config.model_checkpoint))

    model = model.to(device)
    train(model, dataloader)


if __name__ == '__main__':
    main()
