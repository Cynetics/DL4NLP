import numpy as np
import torch
import datetime
import torch.nn as nn
from sklearn.metrics import classification_report
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def get_accuracy(outputs, labels):
    return torch.sum(outputs==labels).float() #/ labels.shape[0]

def get_paragraph_prediction(outputs, labels):
    unique_preds = torch.unique(outputs)
    #print("unique_preds: ", unique_preds)
    counts = torch.tensor([(outputs==lang).float().sum() for lang in unique_preds])
    #print("counts: ", counts)
    max_lang = unique_preds[torch.argmax(counts)]
    return max_lang

def get_mean_softmax(outputs):
    probabilities = torch.softmax(outputs, dim=1)
    probabilities = torch.mean(probabilities, dim=0)
    return probabilities

def validate_paragraphs(model, validation_data, validation_loader, save_classification_report=True, subset=True, config=None):
    n_batches = len(validation_loader)
    if subset: n_batches = 500
    validation_data.predict_paragraph(True)
    y_pred = []; y_true = [];
    accuracies = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device).squeeze(0)
            #print(inputs.shape)
            labels = labels.to(device).squeeze(0)
            logits = model(inputs, eval=True)
            #output = torch.argmax(output,dim=1).view(-1)
            #prediction = get_paragraph_prediction(output, labels)
            probs = get_mean_softmax(logits)
            #output = torch.argmax(logits,dim=1).view(-1)
            prediction = torch.argmax(probs)
            y_pred.append(prediction.item()); y_true.append(labels[0].item());
            correct = (prediction == labels[0].item()).float()
            #print("prediction: ", prediction, "label: ", labels[0].item())
            #print(correct)
            accuracies.append(correct.item())
            print(i)
            if i == n_batches: break

    accuracy = round(np.sum(accuracies)/(n_batches),4)
    if save_classification_report:
        target_names = validation_data.languages
        df = pd.DataFrame(classification_report(y_true, y_pred, target_names=target_names, output_dict=True)).transpose()
        df.to_csv('classification_report_deterministic_{}_{}.csv'.format(config.batch_size, config.input), index= True)
    return accuracy


def validate_uncertainty(model, validation_data, validation_loader, config=None):
    n_batches = len(validation_loader)
    validation_data.predict_paragraph(True)
    model = model #.cpu()
    y_pred = []; y_true = [];
    accuracies = []

    with open("Bayesian_Results_gpt.csv", "w") as file:
        file.write("Data_index; predictio; label; means; std\n")

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_loader):
            if i < 24000:
                continue
            inputs = inputs.to(device).squeeze(0) #.cpu()
            labels = labels.to(device).squeeze(0) #.cpu()
            datapoint_probs = torch.zeros(25,235)
            print(inputs.shape)
            for n in range(25):
                logits = model(inputs, eval=False)
                probabilities = get_mean_softmax(logits)
                datapoint_probs[n,:] = probabilities

            standard_deviations = torch.std(datapoint_probs, dim=0)
            means = torch.mean(datapoint_probs, dim=0)
            prediction = torch.argmax(means).item()

            label = labels[0].item()

            y_pred.append(prediction)
            y_true.append(label)

            correct = (prediction == label)#.float()

            means = [round(mean, 6) for mean in means.cpu().numpy()]
            std = [round(std_i, 6) for std_i in standard_deviations.cpu().numpy()]
            with open("Bayesian_Results_gpt.csv", "a") as file:
                file.write(str(i)+"; "+validation_data.idx_to_lang[prediction]+"; " + \
                            validation_data.idx_to_lang[label]+"; "+str(means)+"; " + \
                            str(std)+"\n")

            accuracies.append(correct)
            print(i)
            #if i == 24010:
            #    break

    accuracy = round(np.sum(accuracies)/(n_batches),4)
    target_names = validation_data.languages
    df = pd.DataFrame(classification_report(y_true, y_pred, target_names=target_names, output_dict=True)).transpose()
    df.to_csv('classification_report_stochastic_{}_{}.csv'.format(config.batch_size, config.input), index= True)
    print(accuracy)
    return accuracy
