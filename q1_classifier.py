import numpy as np
from scipy.stats import chisquare
import pandas as pd
from argparse import ArgumentParser


class Node:
    def __init__(self, split_feature=None, is_leaf=False, label=None):
        self.split_feature = split_feature
        self.is_leaf = is_leaf
        self.children = {}
        self.label = label


def read_data(xtrain_filename, ytrain_filename, xtest_filename, ytest_filename):
    xtrain = pd.read_csv(xtrain_filename, header=None, delimiter=' ')
    ytrain = pd.read_csv(ytrain_filename, header=None, delimiter=' ')
    xtest = pd.read_csv(xtest_filename, header=None, delimiter=' ')
    ytest = pd.read_csv(ytest_filename, header=None, delimiter=' ')

    return xtrain, ytrain, xtest, ytest


def entropy(label):
    probability = np.divide(np.unique(label, return_counts=True)[1], label.shape[0])
    entropy = -1 * np.sum(np.multiply(probability, np.log2(probability)))
    return entropy


def find_best_split(data, features):
    best_feature = -1

    max_IG = -1

    entropy_y = entropy(data['label'])

    if entropy_y == 0:
        return best_feature

    for i in features:
        values = np.unique(data[i])
        IG = entropy_y
        for j in values:
            y_j = data['label'][data[i] == j]
            IG -= entropy(y_j) * (y_j.shape[0] / data['label'].shape[0])
        if IG > max_IG:
            max_IG = IG
            best_feature = i

    if max_IG == entropy_y:
        return -1

    return best_feature


def getMajorityLabel(data):
    label = '1' if data[data['label'] == 1].shape[0] > data[data['label'] == 0].shape[0] else '0'
    return Node(is_leaf=True, label=label)


def getChisquare(data, best_feature):
    observed = []
    expected = []

    zeroPercentage = (data['label'] == 0).sum() / data['label'].count()
    onePercentage = (data['label'] == 1).sum() / data['label'].count()

    for value in np.unique(data[best_feature]):
        data_i = data[data[best_feature] == value]
        observed_frequency = np.unique(data_i['label'], return_counts=True)[1]
        if len(observed_frequency) == 1:
            observed_frequency = np.append(observed_frequency, 0)
        observed.extend(observed_frequency)
        expected.append(zeroPercentage * data_i['label'].count())
        expected.append(onePercentage * data_i['label'].count())

    return chisquare(observed, expected)[1]


def generateTree(data, features):
    if all(data['label'] == 0):
        return Node(is_leaf=True, label='0')
    if all(data['label'] == 1):
        return Node(is_leaf=True, label='1')
    if len(features) == 0:
        return getMajorityLabel(data)

    best_feature = find_best_split(data, features)

    if best_feature == -1:
        return getMajorityLabel(data)

    # use chi square to check stopping criterion
    p_value = getChisquare(data, best_feature)

    if p_value >= float(threshold):
        return getMajorityLabel(data)

    # features.remove(best_feature)

    root = Node(split_feature=best_feature, is_leaf=False)

    new_features = features[:]
    new_features.remove(best_feature)

    for i in np.unique(data[best_feature]):
        data_i = data[data[best_feature] == i][:]
        root.children[i] = generateTree(data_i, new_features)

    return root


def output_tree(node, level, f):
    f.write('%slevel %d\n' % ('\t' * level, level))
    if not node.is_leaf:
        f.write('%sinternal node, split on %d, has %d children\n' % ('\t' * level, node.split_feature, len(node.children)))
        for c in node.children.values():
            output_tree(c, level + 1, f)
    else:
        f.write('%sleaf node with label %s\n' % ('\t' * level, node.label))


def getNumOfNodes(node):
    if node.is_leaf:
        return 1
    else:
        sum_nodes = 0
        for c in node.children.values():
            sum_nodes += getNumOfNodes(c)
        return sum_nodes


def evaluate(node, data):
    if not node.is_leaf:
        if data[node.split_feature] not in node.children:
            return '0'
        return evaluate(node.children[data[node.split_feature]], data)
    else:
        return node.label


def getAccuracy(ypredict, ytest):
    correct = 0
    total = 0
    for i in range(len(ypredict)):
        if int(ypredict[i]) == ytest[0][i]:
            correct += 1
        total += 1

    return correct / total


def output_to_csv(ypredict, output_file):
    ypredict.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p')
    parser.add_argument('-f1')
    parser.add_argument('-f2')
    parser.add_argument('-o')
    parser.add_argument('-t')

    args = parser.parse_args()

    threshold = args.p

    xtrain_filename = args.f1
    ytrain_filename = 'trainlabs.csv'

    xtest_filename = args.f2
    ytest_filename = 'testlabs.csv'

    output_file = args.o

    tree = args.t

    xtrain, ytrain, xtest, ytest = read_data(xtrain_filename, ytrain_filename, xtest_filename, ytest_filename)

    ytrain.columns = ['label']

    node = generateTree(pd.concat([xtrain, ytrain], axis=1), list(range(xtrain.shape[1])))

    # get number of nodes in tree
    num_of_nodes = getNumOfNodes(node)

    # get the predict labels
    ypredict = []
    for i in range(xtest.shape[0]):
        ypredict.append(evaluate(node, xtest.iloc[i].tolist()))

    # output decision tree
    f = open(tree, 'w')
    output_tree(node, 0, f)
    f.close()

    # output prediction to given output filename
    output_to_csv(pd.DataFrame(ypredict), output_file)

    # get accuracy
    accuracy = getAccuracy(ypredict, ytest)

    # print final results
    print("Threshold is", threshold)
    print("The tree size:", num_of_nodes)
    print("Accuracy:", accuracy)
