import numpy as np
import os
import pickle
import pandas as pd
import csv
import copy
import re
import logging
from sklearn.utils import shuffle
from collections import Counter
from sklearn.utils import shuffle
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Flatten, concatenate, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling2D

from path import Path

class Data(Path):
    # TODO: following
    """
    data part:
    - currently we convert all the name in training data into either one of <m0>, <f0>, <p>, to make learning easier.
    to reverege this, we should do this same procedure into test data when loading test dataset.

    - since <m0> <f0> <p> is not recorded in pretrained word embedding, we should allocate some non zero vector
    (e.g. average the embeddings of human names)

    training part:


    """

    def __init__(self, logger, data_limit=None, prepare_dummy=True):
        """
        :param data_limit: if specified, only top n dataset will be loaded.
        :param prepare_dummy: if specified, augment train dataset with fake ending.

        others:
        self.most_common: if specified, top n elements will be used from self.vocab
        self.max_seqlen: 30 as the previous exercise.

        self.train_dataset: whole train dataset with strings. length is fixed to 30 by using padding.
        self.train_dataset_ids: whole train dataset converted from string to integer ids.
        self.train_x: same as self.train_dataset. (contains 5 sentences as 1 story in each row.)

        self.test_dataset: whole test dataset with strings. length is 30, too.
        self.test_dataset_ids: whole test dataset converted from string to integer ids.
        self.test_x: first 4 sentences extracted from self.test_dataset_ids.
        self.test_e1: test ending1. one possible choice for true ending.
        self.test_e2: test ending2. either test_e1 or test_e2 is true.
        self.test_answers: 1 or 2.
        """

        self.most_common = 20000
        self.max_seqlen = 25
        self.n_dummy = 0 # will be updated in subsequent function
        self.prepare_dummy = prepare_dummy
        self.data_limit = data_limit
        self.logger = logger

        self.unk = "<UNK>"
        self.pad = "<PAD>"
        # TODO: do we really need bos, eos, pad ?

        self.train_dataset, self.y = self.load_train_text(Path.train_file_path)
        self.vocab = self.create_vocab()  # used in test
        self.vocab_size = len(self.vocab) # contain <PAD>
        self.w2i_dict = self.create_w2i_dict()  # contain <PAD>
        self.i2w_dict = self.create_i2w_dict()  # used in test
        self.train_dataset_ids = self.convert_w2i_dataset(self.train_dataset)
        self.train_x = self.train_dataset_ids
        self.embedding_matrix = self.create_pretrained_embedding_matrix(Path.glove_path)

        self.test_dataset, self.test_answers = self.load_test_text(Path.val_file_path)
        self.test_dataset_ids = self.convert_w2i_dataset(self.test_dataset)
        self.test_x, self.test_e1, self.test_e2 = self.split_test_dataset(self.test_dataset_ids)

        # report about variables
        params_string = ''
        params_string += '''
        ==================================================
        words with frequency less than {} is not in vocab.
        maximum sentencelength: {}
        train_x.shape:  {}
        test_x.shape:   {}
        test_e1.shape:  {}
        test_e2.shape:  {}
        len(vocab):     {}
        ==================================================
        '''.format(self.most_common, self.max_seqlen, self.train_x.shape,
                   self.test_x.shape, self.test_e1.shape, self.test_e2.shape, len(self.vocab))
        params_string += "\n".join(["{}: {}".format(key, val)
                                  for key, val in Path.__dict__.items() if key[0] != '_'])
        self.logger.info(params_string)
        self.logger.info("data loaded.")

    def retrieve_data(self):
        data_dict = {}
        data_dict['vocab'] = self.vocab
        data_dict['w2i_dict'] = self.w2i_dict
        data_dict['embedding_matrix'] = self.embedding_matrix
        return data_dict

    def split_test_dataset(self, dataset):
        # assuming dataset.shape = (datanum, 4 + 1 + 1, 30)
        # TODO: remove hardcoding
        return dataset[:, :4, :], dataset[:, 4, :], dataset[:, 5, :]

    def clean_text(self, string):
        """
        apply lowercasing, inserting space if one of { '.,!? } is found.x
        :param string:
        :return: string
        """
        string = string.lower()
        # insert space before special symbols
        string = re.sub("(['.,!?])", r' \g<1>', string)

        return string

    def create_pretrained_embedding_matrix(self, datapath):
        """
        the file format should be 'word0, v_0, v_1, ..., v_n \n word1, v_0, ...'
        :param datapath: pretrained embedding path (should have fixed format)
        :return: embedding matrix (np.ndarray)
        the i-th row of matrix should represent the embedding for word whose id is i.
        self.w2i_dict
        """
        self.logger.info("creating embedding matrix...")
        word_to_embedding = {}

        # create the word2embedding dictionary
        f = open(datapath)
        with open(datapath, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                word_to_embedding[word] = vector

        # create the embedding matrix by the above dictionary
        embedding_dim = len(vector)
        embedding_matrix = np.zeros((len(self.w2i_dict), embedding_dim))

        for word, index in self.w2i_dict.items():
            embedding = word_to_embedding.get(word)

            if embedding is None:
                # if the word is in dataset but not found in pretrained vector, leave the corresponding row as zero vectors
                self.logger.info("word: {} not found in pretrained embedding.".format(word))

            else:
                embedding_matrix[index] = embedding
                self.logger.debug("word: {}\t index:{}\t matrix[index]:{}\t w2v[word]:{}".format(word, index, embedding_matrix[index], word_to_embedding[word]))

        return embedding_matrix

    def load_test_text(self, datapath):
        """
        loads sentences from datapath and adding bos, eos, pad.
        :param datapath: path for csv file. (specified in Path class.)
        :return: list of stories, answers(array of 1 or 2).
        """
        df = pd.read_csv(datapath)

        story_ids = df['InputStoryid'].tolist()
        stories = (df[['InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4',
                       'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2']])
        answers = np.array(df['AnswerRightEnding'].tolist())

        lines = stories.values.tolist()
        if self.max_seqlen:
            lines = ([[[self.pad] * (self.max_seqlen - len(self.clean_text(string).split())) +
                       self.clean_text(string).split()
                       for string in line] for line in lines])

        self.test_story_ids = story_ids
        return lines, answers

    def augment_with_fake(self, df):
        """
        Given df, this function copies it and replace 'sentence5' with fake ending
        (picked from each of 'sentence1'-'sentence4' in the same story) and appending that new df to old df.
        so the dataset will be 5 times bigger.

        :param df: pandas.DataFrame
        :return: augmented pandas.DataFrame and answers
        """
        augmented_df = copy.copy(df)
        augmented_answers = np.ones(len(df))
        # columns = ['sentence1', 'sentence2', 'sentence3', 'sentence4']
        columns = ['sentence1']

        self.n_dummy = len(columns)

        for column_name in columns:
            fake_df = copy.copy(df)
            fake_answers = np.zeros(len(fake_df))
            fake_df['sentence5'] = shuffle(df[column_name])
            augmented_df = augmented_df.append(fake_df, ignore_index=True)
            augmented_answers = np.concatenate((augmented_answers, fake_answers), axis=0)

        # check if those 2 lengths are the same.
        assert len(augmented_df) == len(augmented_answers)
        return augmented_df, augmented_answers

    def load_train_text(self, datapath):
        """
        :param datapath:
        :return:
        """
        self.logger.info("loading dataset...(assuming 5 fake endings for each stories")
        if self.data_limit:
            self.logger.info("\tdata_limit:{}".format(self.data_limit))

        df = pd.read_csv(datapath)
        if self.data_limit:
            df = df.loc[:self.data_limit, :]


        stories = df[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']]
        answers = df['is_real_ending']

        lines = stories.values.tolist()
        if self.max_seqlen:
            lines = ([[ [self.pad] * (self.max_seqlen - len(self.clean_text(string).split()))
                        + self.clean_text(string).split()
                       for string in line] for line in lines])  # extract 'sentence1 - 5'
        else:
            raise AttributeError("specify max seqlen.")

        lines = np.array(lines)

        # shuffle the data
        random_indices = np.random.permutation(lines.shape[0])
        lines = lines[random_indices]
        answers = answers[random_indices]

        self.n_dummy = 5
        """
        stories = df[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']]
        answers = np.ones(len(stories)) # because 'sentence5' is always true ending.

        if self.prepare_dummy:
            stories, answers = self.augment_with_fake(stories)
        lines = stories.values.tolist()

        if self.max_seqlen:
            lines = ([[ [self.pad] * (self.max_seqlen - len(self.clean_text(string).split()))
                        + self.clean_text(string).split()
                       for string in line] for line in lines])  # extract 'sentence1 - 5'
        else:
            lines = [[self.clean_text(string).split() for string in line[2:]] for line in lines]

        lines = np.array(lines)

        # shuffle the data
        random_indices = np.random.permutation(lines.shape[0])
        lines = lines[random_indices]
        answers = answers[random_indices]

        # self.train_story_ids = story_ids
        # self.train_story_titles = story_titles
        """
        return lines, answers

    def create_vocab(self):
        self.logger.info("creating vocabulary...")
        flattened_dataset = [word for sentences in self.train_dataset for sentence in sentences[1:] for word in
                             sentence]
        self.logger.info("============== limiting vocabulary to top {}.==================".format(self.most_common))
        # vocab = dict(Counter(flattened_dataset), most_common=self.most_common)
        vocab = dict(Counter(flattened_dataset).most_common(self.most_common))
        self.logger.info("\ttop {} frequent vocabulary (and <UNK>) will be used..".format(self.most_common))
        vocab[self.unk] = 1
        return vocab

    def create_w2i_dict(self):
        """
        vocab which converts word to id
        """
        w2i_vocab = dict()
        w2i_vocab[self.pad] = 0
        i = 1
        for key, val in self.vocab.items():
            if w2i_vocab.get(key) == None:
                w2i_vocab[key] = i
                i += 1

        assert len(w2i_vocab) == len(self.vocab)
        return w2i_vocab

    def create_i2w_dict(self):
        return {v: k for k, v in self.w2i_dict.items()}

    def get_id(self, word):
        ind = self.w2i_dict.get(word)
        if ind == None:
            return self.w2i_dict[self.unk]
        return ind

    def convert_w2i_dataset(self, dataset):
        array = np.array(
            [[[self.get_id(word) for word in sentence] for sentence in sentences] for sentences in dataset])
        return array

    def convert_i2w_dataset(self):
        pass

    def prepare_training_data(self):
        train_x, train_y = self.dataset_ids[:, :4, :], self.dataset_ids[:, 4:, :]
        return train_x, train_y

    def depth(l):
        """
        get the depth of the list (unused for now)
        """
        if isinstance(l, list):
            return 1 + max(self.depth(item) for item in l)
        else:
            return 0

        # TODO: is test data converted to ids correctly?

def setup_logger(logger_name, file_name, level, add_console = True):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    handler = logging.FileHandler(file_name)

    if add_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    logger.addHandler(handler)
    return logger
