from path import Path
import numpy as np
from keras.models import Model
# from keras.utils import plot_model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Flatten, Reshape, Lambda, TimeDistributed, concatenate, multiply
from keras import optimizers
from keras import backend as K
import keras

import tensorflow as tf

class Customtrainingreporter(keras.callbacks.Callback):
    """
    custom calll back class for model.fit(). not used currently
    ref: https://keras.io/callbacks/
    """

    def __init__(self, logger):
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info(len(self.gradients))

    def check_gradient(self):
        output = self.model.output
        variable_tensors = self.model.trainable_weights
        # print("variable_tensors: ", variable_tensors)
        gradients = K.gradients(output, variable_tensors)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        evaluated_gradients = sess.run(gradients, feed_dict={self.model.inputs[i]: d for i, d in enumerate(self.input_data)})
        evaluated_gradients = [np.sum(grad) for grad in evaluated_gradients]
        return evaluated_gradients


class Classifier():
    def __init__(self, model_name, max_seqlen, vocab_size, n_dummy, pretrained_embedding,
                 params_logger, train_logger
                 ):
        self.available_models = ['niko', 'scnn']
        self.model_name = model_name
        self.embed_dim = 128
        self.hidden_dim = 64
        self.feature_dim = 32
        self.model = None
        self.use_pretrained_embedding = True
        self.pretrained_embedding = pretrained_embedding
        self.params_logger = params_logger
        self.train_logger = train_logger

        self.batchsize = 64
        self.epochs = 50

        self.n_stories = 4
        self.n_options = 1

        self.max_seqlen = max_seqlen
        self.vocab_size = vocab_size
        self.n_dummy = n_dummy
        if self.n_dummy > 0:
            self.class_weight = {0: 1.,
                                 1: float(self.n_dummy)}
        else:
            self.class_weight = None

        params_string = """
        model: {}
        batchsize: {}
        epochs: {}
        max_seqlen: {}
        vocab_size: {}
        """.format(self.model_name, self.batchsize, self.epochs, self.max_seqlen, self.vocab_size)
        self.train_logger.info("")

        # if the loss for class 1 were (pred-1)**2, then class weight makes it ((pred-1)**2) * (self.n_dummy)/1.
        # it increases the loss w.r.t the balance of training data labels

    def build_model(self):
        if self.model_name == 'niko':
            self._build_niko()

        elif self.model_name == 'scnn':
            self._build_scnn()
        else:
            raise IOError("model should be picked from {}".format(self.available_models))
        self.train_logger.info("model: {} built.".format(self.model_name))

    def _build_niko(self):
        # TODO: make it faster -> DONE to some extent
        story1_input = Input(shape=(self.max_seqlen,), name="story1")
        story2_input = Input(shape=(self.max_seqlen,), name="story2")
        story3_input = Input(shape=(self.max_seqlen,), name="story3")
        story4_input = Input(shape=(self.max_seqlen,), name="story4")
        story_op_input = Input(shape=(self.max_seqlen,), name="story_op")

        inputs = [story1_input, story2_input, story3_input, story4_input, story_op_input]
        if not self.use_pretrained_embedding:
            raise ValueError("you have to specify the embeddings.")

        self.embed_dim = self.pretrained_embedding.shape[1]
        story_embed_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim,
                                input_length= 4 * self.max_seqlen, mask_zero=True,
                                weights=[self.pretrained_embedding], trainable=False,
                                name='story_embedding')

        op_embed_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim,
                                input_length= self.max_seqlen, mask_zero=True,
                                weights=[self.pretrained_embedding], trainable=False,
                                name='op_embedding')

        ave_layer = Lambda(lambda x: K.mean(x, axis=1, keepdims=False))
        dense_layer = Dense(50, activation='relu') # hard code units since its specified as 200-250.
        sf_layer = Dense(1, activation='sigmoid')

        C_input = concatenate([story1_input, story2_input, story3_input, story4_input]) # (None, self.max_seqlen) -> (None, self.max_seqlen * 4)

        embeddings = story_embed_layer(C_input)
        C_embeddings = ave_layer(embeddings) # -> (None, self.max_seqlen * 4, self.embed_dim) -> (None, self.embed_dim)
        op_embeddings = ave_layer(op_embed_layer(story_op_input))
        print("C_embeddings.shape: ", C_embeddings.shape)

        embeddings = concatenate([C_embeddings, op_embeddings])
        #TODO: they might have used relu in this embedding layer
        print("embeddings.shape: ", embeddings.shape)

        hidden_feature = dense_layer(embeddings)
        pred = sf_layer(hidden_feature)
        sgd = optimizers.SGD(lr=0.04)
        # add regularizer as specified in the paper

        model = Model(inputs=inputs, outputs=pred)
        model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

        self.embedding_model = Model(inputs = inputs, outputs = embeddings)
        print(model.summary())
        self.model = model

    def _build_scnn(self):
        # TODO: too dirty. make it faster somehow
        story1_input = Input(shape=(self.max_seqlen,), name="story1")
        story2_input = Input(shape=(self.max_seqlen,), name="story2")
        story3_input = Input(shape=(self.max_seqlen,), name="story3")
        story4_input = Input(shape=(self.max_seqlen,), name="story4")
        option_input = Input(shape=(self.max_seqlen,), name="story5")
        # inputs = concatenate([story1_input, story2_input, story3_input, story4_input, option_input], axis=1)

        # TimeDistributed enables to apply Embedding function uniformly for each of sentence{1, 2, 3, 4, 5}
        if self.use_pretrained_embedding:
            # override the embed dimension size
            self.embed_dim = self.pretrained_embedding.shape[1]
            embed_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim,
                                    input_length=self.max_seqlen, mask_zero=True,
                                    weights=[self.pretrained_embedding], trainable=False,
                                    name='embedding')
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
        birnn_layer = Bidirectional(LSTM(self.hidden_dim, input_shape=(5, self.max_seqlen, self.embed_dim),
                                         return_sequences=False), name='bidirectional_lstm')
        dense_layer = Dense(self.feature_dim, activation='relu', name='dense')
        ending_dense_layer = Dense(self.feature_dim, activation='relu', name='dense_ending')

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

        # TODO: use whole lstm outputs rather than the last output
        # TODO: is it good to use different layer for different sentences?

        story_outputs_op = ending_dense_layer(birnn_outputs_op)  # -> (None, feature_dim)
        story_outputs_1 = multiply([story_outputs_op, dense_layer(birnn_outputs_1)]) # -> (None, feature_dim)
        story_outputs_2 = multiply([story_outputs_op, dense_layer(birnn_outputs_2)])
        story_outputs_3 = multiply([story_outputs_op, dense_layer(birnn_outputs_3)])
        story_outputs_4 = multiply([story_outputs_op, dense_layer(birnn_outputs_4)])

        story_features = concatenate([story_outputs_1, story_outputs_2, story_outputs_3, story_outputs_4])
        # story_features = Reshape((1, 4 * self.feature_dim))(fc_outputs)

        # TODO implement self-weighting (mistrious though)

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

        rmsprop = optimizers.RMSprop()
        # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=rmsprop, loss='mean_squared_error', metrics=['accuracy'])
        # plot_model(model, Path.image_save_path)
        print(model.summary())
        self.model = model

    def train(self, inputs, outputs, save_output, save_path):
        if self.model == None:
            raise ValueError("self.model is None. run build_model() first.")

        modelsave_callback = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)

        hist = self.model.fit(inputs, outputs, epochs=self.epochs, batch_size=self.batchsize,
                              shuffle=True, validation_split=0.2, verbose=1,
                              class_weight=self.class_weight, callbacks=[modelsave_callback])
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
