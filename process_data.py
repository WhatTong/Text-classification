from Config import *
import numpy as np
import tensorflow as tf

def build_word2id(train_path, validation_path, test_path):
    word2id = {'PAD': 0}
    paths = [train_path, validation_path, test_path]
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    tag2id = {'0': 0, '1': 1}

    return word2id, tag2id


def load_data(train_path, validation_path, test_path, word2id, tag2id):

    x_train, y_train = [], []

    x_validation, y_validation = [], []

    x_test, y_test = [], []

    x_train_id, x_validation_id, x_test_id = [], [], []
    y_train_id, y_validation_id, y_test_id = [], [], []

    with open(train_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            y_train.append(data[0])
            x_train.append(data[1].strip().split())

    with open(validation_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            y_validation.append(data[0])
            x_validation.append(data[1].strip().split())

    with open(test_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            y_test.append(data[0])
            x_test.append(data[1].strip().split())

    for i in range(len(x_train)):
        x_train_id.append([word2id[x] for x in x_train[i] if x in word2id])
        # y_train_id.append([tag2id[x] for x in y_train[i] if x in tag2id])
        y_train_id += [tag2id[x] for x in y_train[i] if x in tag2id]

    for i in range(len(x_validation)):
        x_validation_id.append([word2id[x] for x in x_validation[i] if x in word2id])
        # y_validation_id.append([tag2id[x] for x in y_validation[i] if x in tag2id])
        y_validation_id += [tag2id[x] for x in y_validation[i] if x in tag2id]

    for i in range(len(x_test)):
        x_test_id.append([word2id[x] for x in x_test[i] if x in word2id])
        # y_test_id.append([tag2id[x] for x in y_test[i] if x in tag2id])
        y_test_id += [tag2id[x] for x in y_test[i] if x in tag2id]

    return x_train_id, y_train_id, x_validation_id, y_validation_id, x_test_id, y_test_id


def process_data(out):
    x_train = tf.keras.preprocessing.sequence.pad_sequences(out[0], maxlen=60, padding='post', value=0)
    # y_train = tf.keras.utils.to_categorical(out[1])   # 1:[0,1]  0:[1,0]
    y_train = out[1]

    x_validation = tf.keras.preprocessing.sequence.pad_sequences(out[2], maxlen=60, padding='post', value=0)
    # y_validation = tf.keras.utils.to_categorical(out[3])
    y_validation = out[3]

    x_test = tf.keras.preprocessing.sequence.pad_sequences(out[4], maxlen=60, padding='post', value=0)
    # y_test = tf.keras.utils.to_categorical(out[5])
    y_test = out[5]

    return x_train, y_train, x_validation, y_validation, x_test, y_test