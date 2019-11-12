import pandas as pd
import numpy as np


def average_precision(element):
    if element.sum() == 0:
        return 0
    gtp = element.sum()
    ap = 0
    for k in range(len(element)):
        ap += element[:k + 1].sum() / (k + 1) * element[k]
    return ap / gtp

def mean_average_precision(Prediction):
    map = dict()
    for key in Prediction:
        value = Prediction[key]
        map.update({key:average_precision(value)/len(value)})
    return map

#Read test
test = pd.read_csv('test_epin.csv', sep=',')
#Result of prediction, read, CHANGE  -----
train = pd.read_csv('-----', sep=',', header=None)
#To remove brackets and list
train[0] = train[0].str.strip('List(')
train[10] = train[10].str.strip(')')
#Transform into list of arrays
train_list = train.values
train_list = [item.astype(int) for item in train_list]
relevance = dict()
for element in train_list:
    lst = element[1:]
    #filter current node in test_set
    actual_nodes = test[test['source_node'] == element[0]]['destination_node'].values
    #trying to find 'destination_node' values in test_data
    is_correct = np.array([item in actual_nodes for item in lst]).astype(int)
       relevance.update({element[0]: np.array(is_correct)})
map = mean_average_precision(relevance)
print(map)
