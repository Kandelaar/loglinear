from parameters import stopwords, filter_freq
from tqdm import tqdm
from sklearn.feature_extraction import text


def filter_ngram(ngram):
    if ngram[0] in stopwords or ngram[-1] in stopwords:
        return True
    return False


class Encoder:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    def build_vocab(self, texts):
        print('building vocab...')
        wordcnt = {}
        for text in tqdm(texts):
            unigrams = text.split()
            unigram_num = len(unigrams)
            for word in unigrams:
                if word not in stopwords:
                    if word in wordcnt.keys():
                        wordcnt[word] += 1
                    else:
                        wordcnt[word] = 1
            for n in [2, 3]:
                for i in range(unigram_num):
                    if unigram_num <= i + n - 1:
                        break
                    ngram = unigrams[i: i + n]
                    if not filter_ngram(ngram):
                        ngram = " ".join(ngram)
                        if ngram in wordcnt.keys():
                            wordcnt[ngram] += 1
                        else:
                            wordcnt[ngram] = 1
        vocab = {'[UNK]': 0}
        i = 1
        for word, cnt in wordcnt.items():
            if cnt >= filter_freq:
                vocab[word] = i
                i += 1
        self.vocab = vocab
        self.vocab_size = i
        print('vocab size:', self.vocab_size)
        '''
        vocab = {}
        i = 0
        for word, cnt in wordcnt.items():
            if cnt >= filter_freq:
                vocab[word] = i
                i += 1
        self.vocab = vocab
        self.vocab_size = i
        print('vocab size:', self.vocab_size)
        '''

    def encode(self, text):
        feature_vec = [0] * self.vocab_size
        valid_ids = []
        unigrams = text.split()
        unigram_num = len(unigrams)
        for word in unigrams:
            if word not in stopwords:
                pos = self.vocab.get(word, 0)
                feature_vec[pos] = 1
                if pos not in valid_ids:
                    valid_ids.append(pos)
        for n in [2, 3]:
            for i in range(unigram_num):
                if unigram_num <= i + n - 1:
                    break
                ngram = unigrams[i: i + n]
                if not filter_ngram(ngram):
                    ngram = " ".join(ngram)
                    pos = self.vocab.get(ngram, 0)
                    feature_vec[pos] = 1
                    if pos not in valid_ids:
                        valid_ids.append(pos)
        return feature_vec, valid_ids
