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
    # We create dict
    map = dict()
    # We go throuth dictionary keys
    for key in Prediction:
        #Save current element of dict
        value = Prediction[key]
        #Change map value
        #Go to def average_precision(element):
        map.update({key:average_precision(value)})
    return map

#START READING HERE

#Read test
test = pd.read_csv('test_epin.csv', sep=',')
#Read Result of prediction, read
train = pd.read_csv('test_epin/part-00000', sep=',', header=None)

#To remove brackets and list
train[0] = train[0].str.strip('(')
train[1] = train[1].str.strip('(ArrayBuffer')
train[10] = train[10].str.strip(')')


#Transform into list of arrays. Example:
 #array([0, 7]),
 #array([0, 9]),
 #array([ 1, 11]),
 #array([ 1, 22])]
train_list = train.values
train_list = [item.astype(int) for item in train_list]

#empty dict
relevance = dict()
#work with every row in train_list
for element in train_list:
    #take destination node from row
    lst = element[1:]
    #searching current source_node in test set and making array of destination_nodes for him
    actual_nodes = test[test['source_node'] == element[0]]['destination_node'].values
    #create array, we check if destination node exist in test array, if he exist 1, no 0. Example:
    #actual_nodes = [1,2,3,4,5,6,7]
    #lst = [3,8,5]
    #array([1, 0, 1])
    is_correct = np.array([item in actual_nodes for item in lst]).astype(int)
    #We update dictionary, key source_node, value: array of relevence. Example: 0: ([1, 0, 1])
    relevance.update({element[0]: np.array(is_correct)})
#Go to def mean_average_precision(Prediction):
map = mean_average_precision(relevance)
map_1 = (sum(map.values()))/len(map)*100
print("Model precision", map_1, "%")
