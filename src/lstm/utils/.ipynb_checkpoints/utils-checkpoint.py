import numpy as np
import torch
import datetime
import torch.nn as nn
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def write_results(results, result_name):
    train_loss, val_loss, val_accuracy = results
    with open("./logs/"+result_name+str(datetime.datetime.now())[4:16]+".csv", "w") as f:
        for i in range(len(train_loss)):
            f.write(str(train_loss[i])+","+ str(val_loss[i])+"," + str(val_accuracy[i])+"\n")

def get_accuracy(outputs, labels):
    return torch.sum(outputs==labels).float() #/ labels.shape[0]

def validate_model(model, validation_data, validation_loader, batch_size, save_classification_report=True):
    n_batches = len(validation_loader)
    model.eval()
    accuracies = []
    y_pred= []; y_true = [];
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device);
            labels = labels.to(device)
            output = model(inputs)
            output = torch.argmax(output,dim=1).view(-1)
            y_pred += output.tolist()
            y_true += labels.view(-1).tolist()
            accuracy = get_accuracy(output, labels)
            accuracies.append(accuracy.item())
    
    accuracy = round(np.sum(accuracies)/(n_batches*batch_size),3)
    if save_classification_report:
        with open("classification_report.txt", 'w') as file:
            target_names = validation_data.languages
            file.write(classification_report(y_true, y_pred, target_names=target_names))
    return accuracy

    
    
