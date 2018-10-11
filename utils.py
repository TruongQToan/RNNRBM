import numpy as np
import pandas as pd


def sigmoid(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def sample(probs, distribution='binomial'):
    ## sample from probs
    if distribution == 'binomial':
        return np.random.binomial(1, probs, probs.shape)
    elif distribution == 'gaussian':
        return np.random.normal(probs, scale=1, size=probs.shape)


def create_batches(data_set, batch_size):
    num_examples = data_set.shape[0]
    input_dim = data_set.shape[1]
    if batch_size >= num_examples:
        return np.array(data_set).reshape((1, -1, input_dim))
    # starts index of batches
    starts = list(range(0, num_examples, batch_size))
    # ends index of batches
    ends = starts[1:]
    ends.append(num_examples)
    batches = []
    for s, e in zip(starts, ends):
        d = np.array(data_set[s: e]).reshape((-1, input_dim))
        batches.append(d)
    return batches


def load_data(mesurement, minute):
    df = pd.read_csv('../data/data_resource_usage_{}Minutes_6176858948.csv'.format(minute))
    if mesurement == 'ram':
        x = df[df.columns[4]]
    elif mesurement == 'cpu':
        x = df[df.columns[3]]
    else:
        raise Exception("Use python train.py [cpu|ram]")
    length = x.shape[0]
    train_index = int(length * 0.8)
    train_data = np.array(x[:train_index])
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = train_data - mean
    train_data = train_data / std
    test_data = np.array(x[train_index:])
    test_data = test_data - mean
    test_data = test_data / std
    return mean, std, train_data, test_data


def preprocess_data(x, window_side):
    padding = np.array(x[0] * window_side)
    x = np.append(padding, x)
    x_edited = []
    y = []
    for i in range(len(x) - window_side):
        x_edited.append(x[i:i + window_side])
        y.append(x[i + window_side])
    return np.array(x_edited), np.array(y).reshape((-1, 1))