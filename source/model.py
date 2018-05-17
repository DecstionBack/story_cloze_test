from path import Path

import numpy as np
from keras.models import Model
# from keras.utils import plot_model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Flatten, concatenate, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling2D

class Classifier():
    def __init__(self, max_seqlen, vocab_size):
        self.embed_dim = 128
        self.hidden_dim = 64
        self.feature_dim = 32
        self.model = None

        self.batchsize = 64
        self.epochs = 10

        self.n_stories = 4
        self.n_options = 1

        self.max_seqlen = max_seqlen
        self.vocab_size = vocab_size

    def build_model(self):
        # keeping each sentence in list is making GPU a lot slower than CPU.
        # by making them 3d batch ?

        story_inputs = [Input(shape=(self.max_seqlen,)) for _ in range(self.n_stories)]
        option_inputs = [Input(shape=(self.max_seqlen,)) for _ in range(self.n_options)]

        inputs = story_inputs + option_inputs
        embed_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim,
                                input_length=self.max_seqlen, mask_zero=True)
        birnn_layer = Bidirectional(LSTM(self.hidden_dim))
        dense_layer = Dense(self.feature_dim, activation='relu')
        ending_dense_layer = Dense(self.n_stories * self.feature_dim, activation='relu')

        embeddings = [embed_layer(_input) for _input in inputs]
        birnn_outputs = [birnn_layer(embedding) for embedding in embeddings]
        fc_outputs = [dense_layer(birnn_output) for birnn_output in birnn_outputs[:self.n_stories]]
        ending_features = ending_dense_layer(birnn_outputs[4])

        story_features = concatenate(fc_outputs)
        story_features = multiply([story_features, ending_features])  # TODO make it more exact like paper do
        #         conv = Conv1D(16, kernel_size=3, activation='relu')(story_features)
        fc = Dense(1, activation='sigmoid')(story_features)

        model = Model(inputs=inputs, outputs=fc)
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        # plot_model(model, Path.image_save_path)
        print(model.summary())
        self.model = model

    def train(self, inputs, outputs):
        if self.model == None:
            raise ValueError("self.model is None. run build_model() first.")
        hist = self.model.fit(inputs, outputs, epochs=self.epochs, batch_size=self.batchsize, shuffle=True, validation_split=0.2, verbose=2)
        return hist

    def test(self, inputs, batchsize):
        if self.model == None:
            raise ValueError("self.model is None. run build_model() first.")
        prediction = self.model.predict(inputs, batch_size=batchsize)
        return prediction

    def calculate_accuracy(self, answer1, answer2, gt):
        result = np.argmax(np.concatenate((answer1, answer2), axis=1), axis=1) + 1
        acc = np.sum(result==gt)
        return acc

    def save_model(self, save_path):
        self.model.save(save_path)