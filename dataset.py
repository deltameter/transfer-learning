import numpy as np
import pickle
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#used to diagnose nn
class ToyDataset():
    FEATURES = {
        'NUM_LABELS': 10,
        'IMAGE_SIZE': 28,
        'DESIRED_SIZE': 28,
        'NUM_CHANNELS': 1
    }

    def get_minibatch(self, batch_size):
        examples, labels = mnist.train.next_batch(batch_size)
        labels = np.argmax(labels, axis=1)
        return examples, labels

class Dataset():

    def __init__(self, is_transfering):
        self.batch_index = 0
        
        train_examples, test_examples, train_labels, test_labels = self.load_data()
        train_examples, test_examples = self.normalize(train_examples, test_examples)
        self.split_data(is_transfering, train_examples, test_examples, train_labels, test_labels)

    def normalize(self, train_examples, test_examples):
        # normalization by mean image
        train_examples = train_examples - np.mean(train_examples, axis=0)
        test_examples = test_examples - np.mean(test_examples, axis=0)

        return train_examples, test_examples

    def split_data(self, is_transfering, train_examples, test_examples, train_labels, test_labels):
        # split the dataset into two types: original examples and transfer examples
        train_original = np.where(np.array(train_labels) < self.TRANSFER_SPLIT)[0]
        train_transfer = np.where(np.array(train_labels) >= self.TRANSFER_SPLIT)[0]
        test_original = np.where(np.array(test_labels) < self.TRANSFER_SPLIT)[0]
        test_transfer = np.where(np.array(test_labels) >= self.TRANSFER_SPLIT)[0]

        self.original = {
                    'num_labels': self.TRANSFER_SPLIT,
                    'train_examples': train_examples[train_original],
                    'train_labels': [label for label in train_labels if label < self.TRANSFER_SPLIT],
                    'test_examples': test_examples[test_original],
                    'test_labels': [label for label in test_labels if label < self.TRANSFER_SPLIT],
                }


        self.transfer = {
                    'num_labels': self.FEATURES['NUM_LABELS'] - self.TRANSFER_SPLIT,
                    'train_examples': train_examples[train_transfer],
                    'train_labels': [label - self.TRANSFER_SPLIT for label in train_labels if label >= self.TRANSFER_SPLIT],
                    'test_examples': test_examples[test_transfer],
                    'test_labels': [label - self.TRANSFER_SPLIT for label in test_labels if label >= self.TRANSFER_SPLIT]
                }

        self.dataset = (self.original if not is_transfering else self.transfer)
        self.FEATURES['NUM_LABELS'] = self.dataset['num_labels']

    def get_minibatch(self, batch_size):
        if self.batch_index >= self.dataset['train_examples'].shape[0]:
            random_permutation = np.random.permutation(self.dataset['train_examples'].shape[0])
            self.dataset['train_examples'] = self.dataset['train_examples'][random_permutation]
            self.dataset['train_labels'] = [self.dataset['train_labels'][i] for i in random_permutation]
            self.batch_index = 0

        batch = (self.dataset['train_examples'][self.batch_index: self.batch_index + batch_size], 
            self.dataset['train_labels'][self.batch_index: self.batch_index + batch_size])

        self.batch_index += batch_size

        return batch

    def get_train_data(self):
        return self.dataset['train_examples'], self.dataset['train_labels']

    def get_test_data(self):
        return self.dataset['test_examples'], self.dataset['test_labels']

class Cifar100Dataset(Dataset):
    TRAINING_PATH = "./dataset/cifar100/train"
    TEST_PATH = "./dataset/cifar100/test"
    TRANSFER_SPLIT = 60

    FEATURES = {
        'NUM_LABELS': 100,
        'IMAGE_SIZE': 32,
        'DESIRED_SIZE': 28,
        'NUM_CHANNELS': 3
    }

    def load_data(self):
        fo = open(self.TRAINING_PATH, 'rb')
        train_data = pickle.load(fo, encoding="bytes")
        train_examples = train_data[b"data"]
        train_labels = train_data[b"fine_labels"]
        fo.close()

        fo = open(self.TEST_PATH, 'rb')
        test_data = pickle.load(fo, encoding="bytes")
        test_examples = test_data[b"data"]
        test_labels = test_data[b"fine_labels"]
        fo.close()

        return train_examples, test_examples, train_labels, test_labels

class CifarDataset(Dataset):
    TRAINING_PATH = "./dataset/cifar10/data_batch_"
    TEST_PATH = "./dataset/cifar10/test_batch"
    NUM_LOAD_BATCHES = 5
    TRANSFER_SPLIT = 6

    FEATURES = {
        'NUM_LABELS': 10,
        'IMAGE_SIZE': 32,
        'DESIRED_SIZE': 28,
        'NUM_CHANNELS': 3
    }

    def load_data(self):
        train_examples = np.empty(shape=(0,3072))
        train_labels = []

        for i in range(1, self.NUM_LOAD_BATCHES + 1):
            fo = open(self.TRAINING_PATH + str(i), 'rb')
            data = pickle.load(fo, encoding="bytes")

            train_examples = np.concatenate((train_examples, data[b"data"]), axis=0)
            train_labels += data[b"labels"]
            fo.close()

        fo = open(self.TEST_PATH, 'rb')
        test_data = pickle.load(fo, encoding="bytes")
        test_examples = test_data[b"data"]
        test_labels = test_data[b"labels"]
        fo.close()
        
        return train_examples, test_examples, train_labels, test_labels
