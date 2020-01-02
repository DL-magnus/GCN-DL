# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 22:19:52 2019

@author: umaer
"""
import torch
from CreateDataset import Dataset
import numpy as np
from torch_geometric.data import Data, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


data_path = 'C:/tmp/raw'
dataset_test = Dataset(path=data_path, train=False)
dataset_test = dataset_test.create_dataset()
dataset_test, idx = dataset_test
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=True)


classes = ('bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night-stand', 'sofa', 'table', 'toilet')
correct = 0
correct_categories = []
predictions = []
labels = []

total = 0
class_total = list(0. for i in range(len(classes)))
class_correct = list(0. for i in range(len(classes)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        outputs = model(data)
        outputs = torch.exp(outputs.data)
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted.tolist())
        labels.append(data.y.tolist())
        total += data.y.size(0)
        correct += (predicted == data.y).sum().item()
        correct_categories.append(predicted[predicted == data.y].tolist())
        c = (predicted == data.y).squeeze().cpu()
        
        for i in range(len(c)):
            label = data.y[i]
            class_correct[label] += c[i].numpy()
            class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of {:5s} : {:5.2f} %'.format(
                classes[i], 100 * class_correct[i] / class_total[i]))


print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))


labels_flat = [item for sublist in labels for item in sublist]
predictions_flat = [item for sublist in predictions for item in sublist]

class_predictions_all = []

for i in range(10):
    labels_idx = [index for index, value in enumerate(labels_flat) if value == i]
    class_predictions = np.asarray(predictions_flat)[np.asarray(labels_idx)]
    class_predictions_all.append(class_predictions)
    

plt.figure(figsize=(10,10))
for i in range(len(class_predictions_all)):
    if i == 0:
        n_bins = 9
    else:
        n_bins = 10
    plt.subplot(5,2,i+1)
    plt.hist(class_predictions_all[i], n_bins)
plt.show()
    
    
with open("C:/Users/umaer/OneDrive/Documents/Results/labels.txt", 'w') as f:
    for item in labels_flat:
        f.write("%s\n" % item)
        
with open("C:/Users/umaer/OneDrive/Documents/Results/predictions.txt", 'w') as f:
    for item in predictions_flat:
        f.write("%s\n" % item)


model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load('C:/Users/umaer/OneDrive/Documents/Results/model2.pt'))