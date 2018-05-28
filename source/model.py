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
        story_inputs = Input(shape=(4, self.max_seqlen))
        option_input = Input(shape=(1, self.max_seqlen))
        inputs = concatenate([story_inputs, option_input], axis=1)

        # TimeDistributed enables to apply Embedding function uniformly for each of sentence{1, 2, 3, 4, 5}
        if self.use_pretrained_embedding:
            # override the embed dimension size
            self.embed_dim = self.pretrained_embedding.shape[1]

            embed_layer = TimeDistributed(Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim,
                                                    input_length=self.max_seqlen, mask_zero=True,
                                                    weights=[self.pretrained_embedding], trainable=False)
                                          , name='embedding')
        else:
            embed_layer = TimeDistributed(Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim,
                                                    input_length=self.max_seqlen, mask_zero=True)
                                          , name='embedding')

        #TODO: make LSTM ignore <pad>
        birnn_layer = TimeDistributed(Bidirectional(LSTM(self.hidden_dim, input_shape=(5, self.max_seqlen, self.embed_dim), return_sequences=True)), name="t_birirectionallstm")
        flatten_story_layer = TimeDistributed(Flatten())
        flatten_ending_layer = TimeDistributed(Flatten())
        dense_layer = TimeDistributed(Dense(self.feature_dim, activation='relu'), name='dense')
        ending_dense_layer = TimeDistributed(Dense(self.n_stories * self.feature_dim, activation='relu'))

        embeddings = embed_layer(inputs) # (None, 5, self.max_seqlen)  -> (None, 5, self.max_seqlen, output_dim)

        birnn_outputs = birnn_layer(embeddings) # (None, 5, self.max_seqlen, output_dim) -> (None, 5, self.max_seqlen, 2 * hidden_dim)
        print("birnn_outputs.shape: ", birnn_outputs.shape)

        story_outputs = Lambda(lambda x: x[:, :4, :], output_shape=(4, self.max_seqlen, 2 * self.hidden_dim))(birnn_outputs)
        ending_outputs = Lambda(lambda x: x[:, 4:, :], output_shape=(1, self.max_seqlen, 2 * self.hidden_dim))(birnn_outputs)
        print("ending_outputs_before: ", ending_outputs.shape)
        # memo: bidirectional LSTM returns the vector with 2 * self.hidden row size.
        #       when applying lambda layer, if you forget 2 * in output_shape, the batchsize will be doubled instead,
        #       which will cause the error later line.
        story_outputs = flatten_story_layer(story_outputs) # -> (None, 4, self.max_seqlen * 2 * hidden_num)
        ending_outputs = flatten_ending_layer(ending_outputs) # -> (?)

        print("story_outputs.shape: ", story_outputs.shape)
        print("ending_outputs.shape: ", ending_outputs.shape)

        fc_outputs = dense_layer(story_outputs) # (None, 4, self.max_seqlen * 2 * hidden_num) -> (None, 4, feature_dim)
        ending_features = ending_dense_layer(ending_outputs) # (None, 1, hidden_num) -> (None, 1, 4 * feature_num)
        ending_features = Flatten()(ending_features) # TODO: check this flattening works as expected
        # story_features = Reshape((1, 4 * self.feature_dim))(fc_outputs)

        story_features = Flatten()(fc_outputs)
        story_features = multiply([story_features, ending_features])  # TODO make it more exact like paper do

        # we only need the likelihood of being true ending so its squashed into scalar
        fc = Dense(1, activation='sigmoid', name='probability')(story_features)

        model = Model(inputs=[story_inputs, option_input], outputs=fc)

        # for printing the output of intermediate layer
        self.embedding_model = Model(inputs=[story_inputs, option_input],
                                outputs=model.get_layer('embedding').output)
        self.bilstm_model = Model(inputs=[story_inputs, option_input],
                             outputs=model.get_layer('t_birirectionallstm').output)

        # some post said RMSprop is better for RNN task
        rmsprop = optimizers.RMSprop()
        model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
        # plot_model(model, Path.image_save_path)
        print(model.summary())
        self.model = model

    def train(self, inputs, outputs):
        if self.model == None:
            raise ValueError("self.model is None. run build_model() first.")
        hist = self.model.fit(inputs, outputs, epochs=self.epochs, batch_size=self.batchsize,
                              shuffle=True, validation_split=0.2, verbose=1, class_weight=self.class_weight)

        record_limit = 100
        output_dict = {}
        output_dict['inputs'] = inputs
        output_dict['embedding'] = self.embedding_model.predict(inputs)
        output_dict['bilstm'] = self.bilstm_model.predict(inputs)
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