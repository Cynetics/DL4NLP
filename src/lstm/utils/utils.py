# -*- coding: utf-8 -*-
import numpy as np
import torch
import pandas as pd
import datetime
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def get_accuracy(outputs, labels):
    return torch.sum(outputs==labels).float()

def get_paragraph_prediction(outputs, labels):
    unique_preds = torch.unique(outputs)
    counts = torch.tensor([(outputs==lang).float().sum() for lang in unique_preds])
    max_lang = unique_preds[torch.argmax(counts)]
    return max_lang

def get_mean_softmax(outputs):
    probabilities = torch.softmax(outputs, dim=1)
    probabilities = torch.mean(probabilities, dim=0)
    return probabilities

def get_entropy(outputs):
    probabilities = get_mean_softmax(outputs)
    probabilities = torch.tensor([1/235 for i in range(235)]).cuda()
    log_probs = torch.log(probabilities)
    entropy = -torch.sum(probabilities * log_probs)
    return entropy, probabilities


def validate_paragraphs(model, validation_data, validation_loader, save_classification_report=True, subset=True, config=None):
    n_batches = len(validation_loader)
    if subset: n_batches = 1000

    validation_data.predict_paragraph(True)
    model.eval()
    y_pred = []; y_true = [];
    accuracies = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_loader):

            if inputs.shape[1] < 1: continue

            inputs = inputs.to(device).squeeze(0).long() #.cpu()
            labels = labels.to(device).squeeze(0) #.cpu()
            logits = model(inputs, True)

            probs = get_mean_softmax(logits)

            prediction = torch.argmax(probs)


            y_pred.append(prediction.item())
            y_true.append(labels[0].item())

            correct = (prediction == labels[0].item()).float()
            accuracies.append(correct.item())
            if i == n_batches: break

    accuracy = round(np.sum(accuracies)/(n_batches),4)

    print(accuracy)
    def rowIndex(row):return row.name
    if save_classification_report:

        df = pd.DataFrame(classification_report(y_true, y_pred,  output_dict=True)).transpose()
        df['lan_index'] = df.apply(rowIndex, axis=1) # for z in df.index]
        df['lan'] = df['lan_index'].map(lambda z:validation_data.idx_to_lang[int(z)] if len(z) < 4 else '')
        df.to_csv('classification_report_LSTM_deterministic_{}_{}_{}.csv'.format(config.batch_size, config.input, config.sequence_length), index= True)

    return accuracy


def validate_uncertainty(model, validation_data, validation_loader, config=None, save_classification_report=True):
    n_batches = len(validation_loader)
    validation_data.predict_paragraph(True)
    model = model #.cpu()
    y_pred = []; y_true = [];
    accuracies = []

    with open("Bayesian_Results_LSTM.csv", "w") as file:
        file.write("Data_index; prediction; label; means; std\n")

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device).squeeze(0) #.cpu()
            labels = labels.to(device).squeeze(0) #.cpu()

            datapoint_probs = torch.zeros(50,235)
            for n in range(50):
                logits = model(inputs, False)
                probabilities = get_mean_softmax(logits)
                datapoint_probs[n,:] = probabilities

            standard_deviations = torch.std(datapoint_probs, dim=0)
            means = torch.mean(datapoint_probs, dim=0)
            prediction = torch.argmax(means).item()

            label = labels[0].item()

            y_pred.append(prediction)
            y_true.append(label)
            correct = (prediction == labels)#.float()

            means = [round(mean, 6) for mean in means.cpu().numpy()]
            std = [round(std_i, 6) for std_i in standard_deviations.cpu().numpy()]
            with open("Bayesian_Results_LSTM_{}_{}_{}.csv".format(config.batch_size, config.input, config.sequence_length), "a") as file:
                file.write(str(i)+"; "+validation_data.idx_to_lang[prediction]+"; " + \
                            validation_data.idx_to_lang[label]+"; "+str(means)+"; " + \
                            str(std)+"\n")

            accuracies.append(correct)

    accuracy = round(np.sum(accuracies)/(n_batches),4)

    print(accuracy)
    def rowIndex(row):return row.name
    if save_classification_report:
        df = pd.DataFrame(classification_report(y_true, y_pred,  output_dict=True)).transpose()
        df['lan_index'] = df.apply(rowIndex, axis=1) # for z in df.index]
        df['lan'] = df['lan_index'].map(lambda z:validation_data.idx_to_lang[int(z)] if len(z) < 4 else '')
        df.to_csv('classification_report_LSTM_stochastic_{}_{}_{}.csv'.format(config.batch_size, config.input, config.sequence_length), index= True)
    return accuracy
