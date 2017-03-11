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
    TRAINING_PATH = "./dataset/data_batch_"
    TEST_PATH = "./dataset/test_batch"
    NUM_BATCHES = 5
    NUM_LABELS = 10

    FEATURES = {
        'NUM_LABELS': 10,
        'IMAGE_SIZE': 32,
        'DESIRED_SIZE': 28,
        'NUM_CHANNELS': 3
    }

    def __init__(self):
        self.batch_index = 0
        self.train_examples = np.empty(shape=(0,3072))
        self.train_labels = []

        for i in range(1, self.NUM_BATCHES + 1):
            fo = open(self.TRAINING_PATH + str(i), 'rb')
            data = pickle.load(fo, encoding="bytes")

            self.train_examples = np.concatenate((self.train_examples, data[b"data"]), axis=0)
            self.train_labels += data[b"labels"]
            fo.close()

        fo = open(self.TEST_PATH, 'rb')
        test_data = pickle.load(fo, encoding="bytes")
        self.test_examples = test_data[b"data"]
        self.test_labels = test_data[b"labels"]

        # normalization by mean image
        self.train_examples = self.train_examples - np.mean(self.train_examples, axis=0)
        self.test_examples = self.test_examples - np.mean(self.test_examples, axis=0)

        fo.close()

    def get_minibatch(self, batch_size):
        if self.batch_index >= self.train_examples.shape[0]:
            random_permutation = np.random.permutation(self.train_examples.shape[0])
            self.train_examples = self.train_examples[random_permutation]
            self.train_labels = [self.train_labels[i] for i in random_permutation]
            self.batch_index = 0

        batch = (self.train_examples[self.batch_index: self.batch_index + batch_size], 
            self.train_labels[self.batch_index: self.batch_index + batch_size])

        self.batch_index += batch_size

        return batch

    def get_test_data(self):
        return self.test_examples, self.test_labels





