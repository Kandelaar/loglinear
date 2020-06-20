from dataset import dataset
from parameters import *
from np_utils import *

from tqdm import tqdm, trange
import time
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


class loglinear_tfidf:
    def __init__(self, train_file, test_file):
        train_data = dataset(train_file, 'train')
        test_data = dataset(test_file, 'test')
        texts = train_data.texts + test_data.texts
        targets = train_data.targets + test_data.targets
        train_data.texts, test_data.texts, train_data.targets, test_data.targets = train_test_split(
            texts, targets
        )
        train_data.size = len(train_data.targets)
        test_data.size = len(test_data.targets)

        tfidf = TfidfVectorizer(max_features=max_features, stop_words=stopwords)
        tfidf.fit(texts)
        self.vocab = {value: i for i, value in enumerate(tfidf.get_feature_names())}
        self.train_texts = train_data.texts
        self.train_targets = train_data.targets
        self.test_texts = test_data.texts
        self.test_targets = test_data.targets
        self.train_size = train_data.size
        self.test_size = test_data.size
        self.timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

        self.v = get_matrix(n_label, max_features)

    def train(self):
        for epoch in range(train_epoch):
            for i in trange(self.train_size):
                text = self.train_texts[i]
                target = self.train_targets[i]
                delta = self.forward(text, target)
                self.update(delta)
            train_acc, test_acc = self.eval()
            self.save_model(train_acc, test_acc, epoch)

    def forward(self, text, target=None):
        v_fs = []
        feature_vec, valid_id = self.encode(text)
        for y in range(n_label):
            v_fs.append(scatter_add(self.v[y], valid_id))
        if target is not None:
            dist = softmax(v_fs)
            delta = get_matrix(n_label, len(self.vocab))
            for y in range(n_label):
                for i in valid_id:
                    delta[y][i] = -dist[y]
            delta[target] = vec_add(delta[target], feature_vec)
            return delta
        else:
            return argmax(v_fs)

    def update(self, delta):
        self.v = num_mul_mat((1.0 - normalization_ratio * lr), self.v)
        self.v = mat_add(self.v, num_mul_mat(lr, delta))

    def eval(self):
        texts = self.train_texts + self.test_texts
        predict = [self.forward(text) for text in tqdm(texts)]
        train_acc = np.mean(predict[:self.train_size] == np.array(self.train_targets))
        test_acc = np.mean(predict[self.train_size:] == np.array(self.test_targets))
        return train_acc, test_acc

    def save_model(self, train_acc, test_acc, epoch):
        if not os.path.exists('./save_model'):
            os.mkdir('./save_model')
        path = os.path.join('./save_model', self.timemark)
        if not os.path.exists(path):
            os.mkdir(path)
        file_name = 'regularized_loglinear_filter_{}_epoch_{}_acc_{:.3f}%_{:.3f}%'.format(
            filter_freq, epoch, train_acc * 100, test_acc * 100
        )
        file_name = os.path.join(path, file_name)
        with open(file_name, 'wb') as writer:
            pickle.dump(self.v, writer)
        print('model parameter saved to ', file_name)

    def encode(self, text):
        feature_vec = [0] * len(self.vocab)
        valid_id = []
        for word in text.split():
            if not self.vocab.__contains__(word):
                continue
            pos = self.vocab.get(word, -1)
            assert pos != -1
            feature_vec[pos] = 1
            if pos not in valid_id:
                valid_id.append(pos)
        return feature_vec, valid_id


if __name__ == '__main__':
    model = loglinear_tfidf('./data/train.csv', './data/test.csv')
    model.train()
