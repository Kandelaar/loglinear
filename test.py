from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from dataset import *
import numpy as np
from parameters import *
from tqdm import tqdm
from utils import scatter_add, softmax


vocab_size = 0


def predict(valid_id, v):
    scores = []
    for y in range(n_label):
        scores.append(scatter_add(v[y], valid_id))
    return np.argmax(scores)


def encode(text, vocab):
    feature_vec = np.zeros(vocab_size)
    valid_id = []
    wordlist = text.split()
    for word in wordlist:
        pos = vocab.get(word, -1)
        if pos == -1:
            pass
        feature_vec[pos] = 1
        if pos not in valid_id:
            valid_id.append(pos)
    return feature_vec, valid_id


if __name__ == '__main__':
    train_data = dataset('./data/train.csv')
    test_data = dataset('./data/test.csv')
    texts = train_data.texts + test_data.texts
    targets = train_data.targets + test_data.targets
    X_train, X_test, y_train, y_test = train_data.texts, test_data.texts, train_data.targets, test_data.targets  # train_test_split(texts, targets, test_size=0.4)
    train_size = len(X_train)
    test_size = len(X_test)

    encoder = CountVectorizer()
    # encoder = TfidfVectorizer(max_features=15000)
    print('encoding...')
    data = encoder.fit_transform(X_train + X_test)
    X_train = data[:train_size]
    X_test = data[train_size:]

    print('naive bayes')
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))

    print('SGD')
    clf = SGDClassifier()
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))

    print('LinearSVC')
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))
