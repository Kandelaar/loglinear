from model import loglinear
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loglinear')
    parser.add_argument('-train_file', default='./data/train.csv')
    parser.add_argument('-test_file', default='./data/test.csv')
    args = parser.parse_args()
    train_file = args.train_file
    test_file = args.test_file
    model = loglinear(train_file, test_file)
    model.train()
