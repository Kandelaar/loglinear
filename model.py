from dataset import dataset
from feature import Encoder
from parameters import *
from np_utils import *

from tqdm import tqdm
import time
import os
import pickle
from sklearn.model_selection import train_test_split


class loglinear:
    def __init__(self, train_file, test_file):
        self.train_data = dataset(train_file, 'train')
        self.test_data = dataset(test_file, 'test')
        '''
        texts = self.train_data.texts + self.test_data.texts
        targets = self.train_data.targets + self.test_data.targets
        self.train_data.texts, self.test_data.texts, self.train_data.targets, self.test_data.targets = train_test_split(
            texts, targets, test_size=0.2
        )
        self.train_data.size = len(self.train_data.targets)
        self.test_data.size = len(self.test_data.targets)
        '''
        self.encoder = Encoder()
        self.encoder.build_vocab(self.train_data.texts + self.test_data.texts)

        self.vocab_size = self.encoder.vocab_size
        self.feature_size = self.vocab_size * n_label
        print('feature count:', self.feature_size)

        self.timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

        self.v = get_matrix(n_label, self.vocab_size)

        print('model built')

    def train(self):
        print('start training')
        for epoch in range(train_epoch):
            insts = self.train_data.get_insts(shuffle=True)
            tqdm_insts = tqdm(insts)
            for text, target in tqdm_insts:
                delta = self.forward(text, target)
                tqdm_insts.set_description('epoch: {}'.format(epoch))
                self.update(delta)
            train_acc = self.eval('train')
            test_acc = self.eval('test')
            print('train_acc: ', train_acc)
            print('test_acc: ', test_acc)
            self.save_model(train_acc, test_acc, epoch)

    def forward(self, text, target=None):
        v_fs = []
        feature_vec, valid_id = self.encoder.encode(text)
        for y in range(n_label):
            v_f = scatter_add(self.v[y], valid_id)
            v_fs.append(v_f)
        if target is not None:
            dist = softmax(v_fs)
            delta = get_matrix(n_label, self.vocab_size)
            for y in range(n_label):
                for id in valid_id:
                    delta[y][id] = -dist[y]
            delta[target] = vec_add(delta[target], feature_vec)
            return delta
        else:
            return argmax(v_fs)

    def update(self, delta):
        # self.v = num_mul_mat((1.0 - normalization_ratio * lr), self.v)
        self.v = mat_add(self.v, num_mul_mat(lr, delta))

    def eval(self, name):
        if name == 'train':
            texts = tqdm(self.train_data.texts)
            gold = self.train_data.targets
            num = self.train_data.size
        else:
            texts = tqdm(self.test_data.texts)
            gold = self.test_data.targets
            num = self.test_data.size
        texts.set_description('evaluate ', name)
        predict = [self.forward(text) for text in texts]
        correct = 0
        for i in range(num):
            if gold[i] == predict[i]:
                correct += 1
        accuracy = correct / num
        return accuracy

    def save_model(self, train_acc, test_acc, epoch):
        if not os.path.exists('./save_model'):
            os.mkdir('./save_model')
        path = os.path.join('./save_model', self.timemark)
        if not os.path.exists(path):
            os.mkdir(path)
        file_name = 'loglinear_filter_{}_epoch_{}_acc_{:.3f}%_{:.3f}%'.format(
            filter_freq, epoch, train_acc * 100, test_acc * 100
        )
        file_name = os.path.join(path, file_name)
        with open(file_name, 'wb') as writer:
            pickle.dump(self.v, writer)
        print('model parameter saved to ', file_name)

    def load_model(self, filename):
        with open(filename, 'rb') as reader:
            self.v = pickle.load(reader)
