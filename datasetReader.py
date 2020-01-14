from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from random import shuffle
import csv


def read_csv(dataset_path, train_sample_size=-1):
    dataset = []
    tweet = []
    labels = []
    pos = 0
    neg = 0
    with open(dataset_path, encoding='latin-1') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            if train_sample_size > 0:
                if line[0] == '0' and pos < train_sample_size/2:
                    pos += 1
                    dataset.append([line[5], line[0]])
                if line[0] == '4' and neg < train_sample_size/2:
                    neg += 1
                    dataset.append([line[5], line[0]])
            else:
                if line[0] == '0' or line[0] == '4':
                    dataset.append([line[5], line[0]])

    shuffle(dataset)

    for i in range(len(dataset)):
        tweet.append(dataset[i][0])
        labels.append(int(dataset[i][1])/4)

    return tweet, labels


def read_dataset(train_path, test_path, train_sample_size, max_tweet_length=50, num_words=2000):
    print('Reading training data ...')
    train_tweet, train_labels = read_csv(train_path, train_sample_size)
    assert len(train_tweet) == len(train_labels)

    print('Reading test data ...')
    test_tweet, test_labels = read_csv(test_path)
    assert len(test_tweet) == len(test_labels)

    print('Transforming word to vector ...')
    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(train_tweet + test_tweet)
    train_tweet_sequence = tokenizer.texts_to_sequences(train_tweet)
    test_tweet_sequence = tokenizer.texts_to_sequences(test_tweet)
    train_tweet_sequence = sequence.pad_sequences(train_tweet_sequence, maxlen=max_tweet_length)
    test_tweet_sequence = sequence.pad_sequences(test_tweet_sequence, maxlen=max_tweet_length)

    print('Dataset transformed')

    return train_tweet_sequence, train_labels, test_tweet_sequence, test_labels, tokenizer.word_index
