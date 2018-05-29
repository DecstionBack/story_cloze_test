from path import Path

import numpy as np
from keras.models import Model
# from keras.utils import plot_model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Flatten, Reshape, Lambda, TimeDistributed, concatenate, multiply
from keras import optimizers
from keras import backend as K

def custom_metrics(y_true, y_pred):
    return K.sum(K.round(y_pred) == y_true)

class Classifier():
    def __init__(self, max_seqlen, vocab_size, n_dummy, pretrained_embedding,
                 params_logger, train_logger
                 ):
        self.embed_dim = 128
        self.hidden_dim = 64
        self.feature_dim = 32
        self.model = None
        self.use_pretrained_embedding = True
        self.pretrained_embedding = pretrained_embedding
        self.params_logger = params_logger
        self.train_logger = train_logger

        self.batchsize = 64
        self.epochs = 5


        self.n_stories = 4
        self.n_options = 1

        self.max_seqlen = max_seqlen
        self.vocab_size = vocab_size
        self.n_dummy = n_dummy
        self.class_weight = {0: 1.,
                             1: float(self.n_dummy)}

        # if the loss for class 1 were (pred-1)**2, then class weight makes it ((pred-1)**2) * (self.n_dummy)/1.
        # it increases the loss w.r.t the balance of training data labels

    def build_model(self):
        # TODO: make it faster -> DONE to some extent
        story1_input = Input(shape=(self.max_seqlen,))
        story2_input = Input(shape=(self.max_seqlen,))
        story3_input = Input(shape=(self.max_seqlen,))
        story4_input = Input(shape=(self.max_seqlen,))
        option_input = Input(shape=(self.max_seqlen,))
        # inputs = concatenate([story1_input, story2_input, story3_input, story4_input, option_input], axis=1)

        # TimeDistributed enables to apply Embedding function uniformly for each of sentence{1, 2, 3, 4, 5}
        if self.use_pretrained_embedding:
            # override the embed dimension size
            self.embed_dim = self.pretrained_embedding.shape[1]
            embed_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim,
                                    input_length=self.max_seqlen, mask_zero=True,
                                    weights=[self.pretrained_embedding], trainable=False)
            '''
            _embed_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim,
                                     input_length=self.max_seqlen, mask_zero=True)
            _embed_layer.set_weights(self.pretrained_embedding)
            _embed_layer.trainable = False
            '''
        else:

            embed_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim,
                                    input_length=self.max_seqlen, mask_zero=True)

        #TODO: make LSTM ignore <pad>
        # == layer definition ==
        """
        birnn_layer = TimeDistributed(Bidirectional(LSTM(self.hidden_dim, input_shape=(5, self.max_seqlen, self.embed_dim), return_sequences=True)), name="t_birirectionallstm")
        flatten_story_layer = TimeDistributed(Flatten())
        flatten_ending_layer = TimeDistributed(Flatten())
        dense_layer = TimeDistributed(Dense(self.feature_dim, activation='relu'), name='dense')
        ending_dense_layer = TimeDistributed(Dense(self.n_stories * self.feature_dim, activation='relu')
        """
        birnn_layer = Bidirectional(LSTM(self.hidden_dim, input_shape=(5, self.max_seqlen, self.embed_dim), return_sequences=False))
        dense_layer = Dense(self.feature_dim, activation='relu')
        ending_dense_layer = Dense(self.feature_dim, activation='relu')

        # calculation
        # embeddings = embed_layer(inputs) # (None, 5, self.max_seqlen)  -> (None, 5, self.max_seqlen, output_dim)

        embeddings_1 = embed_layer(story1_input) # (None, self.max_seqlen) -> (None, sel.max_seqlen, output_dim)
        embeddings_2 = embed_layer(story2_input)  # (None, self.max_seqlen) -> (None, sel.max_seqlen, output_dim)
        embeddings_3 = embed_layer(story3_input)  # (None, self.max_seqlen) -> (None, sel.max_seqlen, output_dim)
        embeddings_4 = embed_layer(story4_input)  # (None, self.max_seqlen) -> (None, sel.max_seqlen, output_dim)
        embeddings_op = embed_layer(option_input) # (None, self.max_seqlen) -> (None, sel.max_seqlen, output_dim)
        embeddings = concatenate([embeddings_1, embeddings_2, embeddings_3, embeddings_4, embeddings_op])

        birnn_outputs_1 = birnn_layer(embeddings_1) # (None, self.max_seqlen, output_dim) -> (None, self.max_seqlen, 2 * hidden_dim)
        birnn_outputs_2 = birnn_layer(embeddings_2) # (None, self.max_seqlen, output_dim) -> (None, self.max_seqlen, 2 * hidden_dim)
        birnn_outputs_3 = birnn_layer(embeddings_3) # (None, self.max_seqlen, output_dim) -> (None, self.max_seqlen, 2 * hidden_dim)
        birnn_outputs_4 = birnn_layer(embeddings_4) # (None, self.max_seqlen, output_dim) -> (None, self.max_seqlen, 2 * hidden_dim)
        birnn_outputs_op = birnn_layer(embeddings_op) # (None, self.max_seqlen, output_dim) -> (None, self.max_seqlen, 2 * hidden_dim)

        story_outputs_op = ending_dense_layer(birnn_outputs_op)  # -> (None, feature_dim)
        story_outputs_1 = multiply([story_outputs_op, dense_layer(birnn_outputs_1)]) # -> (None, feature_dim)
        story_outputs_2 = multiply([story_outputs_op, dense_layer(birnn_outputs_2)])
        story_outputs_3 = multiply([story_outputs_op, dense_layer(birnn_outputs_3)])
        story_outputs_4 = multiply([story_outputs_op, dense_layer(birnn_outputs_4)])

        story_features = concatenate([story_outputs_1, story_outputs_2, story_outputs_3, story_outputs_4])
        # story_features = Reshape((1, 4 * self.feature_dim))(fc_outputs)

        # TODO: multiply for each story sentences
        # story_features = multiply([story_outputs, story_outputs_op])  # TODO make it more exact like paper do

        # we only need the likelihood of being true ending so its squashed into scalar
        fc = Dense(1, activation='sigmoid', name='probability')(story_features)

        inputs = [story1_input, story2_input, story3_input, story4_input, option_input]
        model = Model(inputs=inputs, outputs=fc)

        # for printing the output of intermediate layer

        # self.embedding_model = Model(inputs=[story_inputs, option_input],
        #                         outputs=model.get_layer('embedding').output)
        self.embedding_model = Model(inputs=inputs,
                                     outputs=embeddings)

        # self.bilstm_model = Model(inputs=inputs,
        #                           outputs=model.get_layer('t_birirectionallstm').output)

        # some post said RMSprop is better for RNN task

        # rmsprop = optimizers.RMSprop()
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='mean_absolute_error', metrics=['accuracy'])
        # plot_model(model, Path.image_save_path)
        print(model.summary())
        self.model = model

    def train(self, inputs, outputs, save_output=True):
        if self.model == None:
            raise ValueError("self.model is None. run build_model() first.")
        hist = self.model.fit(inputs, outputs, epochs=self.epochs, batch_size=self.batchsize,
                              shuffle=True, validation_split=0.2, verbose=1, class_weight=self.class_weight)
        output_dict = {}
        if save_output:
            output_dict['inputs'] = inputs
            output_dict['embedding'] = self.embedding_model.predict(inputs)
            # output_dict['bilstm'] = self.bilstm_model.predict(inputs)
            output_dict['probability'] = self.model.predict(inputs)
        return output_dict

    def test(self, inputs, batchsize):
        if self.model == None:
            raise ValueError("self.model is None. run build_model() first.")
        prediction = self.model.predict(inputs, batch_size=batchsize)
        return prediction

    def calculate_accuracy(self, answer1, answer2, gt):
        result = np.argmax(np.concatenate((answer1, answer2), axis=1), axis=1) + 1
        acc = np.sum(result == gt)
        return acc

    def save_model(self, save_path):
        self.model.save(save_path)