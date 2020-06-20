import csv
import random
import re


class dataset:
    def __init__(self, file, name='train'):
        print('reading {} data file and normalize...'.format(name))
        self.file = open(file, encoding='utf-8')
        self.texts = []
        self.targets = []
        self.size = 0
        self.name = name

        self.read_data()
        print('complete')

    def get_insts(self, shuffle=True):
        insts = []
        for i in range(self.size):
            insts.append((self.texts[i], self.targets[i]))
        if shuffle is True:
            random.shuffle(insts)
        return insts

    def read_data(self):
        csv.field_size_limit(500*1024*1024)
        reader = csv.DictReader(self.file)
        for row in reader:
            self.texts.append(row['data'])
            self.targets.append(int(row['target']))
            self.size += 1

    @staticmethod
    def normalize(text):
        return re.sub(r"([.!?])", r" ", text.lower().strip())
