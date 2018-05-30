from model import Classifier
from data_utils import Data, setup_logger
from keras import backend as K
import pickle
import logging
from path import Path
import os
import numpy as np
from datetime import datetime

datestring =  datetime.now().strftime('%m%d%Y_%H%M')
save_model_path = os.path.join(Path.model_save_path, datestring)
log_path = os.path.join(Path.log_path, datestring)

if not os.path.exists(log_path):
    os.mkdir(log_path)
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)

# == logger setup. ==
# logger for hyper parameters.
params_logger = setup_logger(logger_name='params_logger',
                             file_name=os.path.join(log_path, 'params.log'),
                             level=logging.INFO
                             )

# logger for training progress.
train_logger = setup_logger(logger_name='train_logger',
                            file_name=os.path.join(log_path, 'train.log'),
                            level=logging.INFO
                            )

def main():
    # data loading
    data = Data(logger=params_logger, data_limit=None)

    # classifier configuration and building
    classifier = Classifier(model_name='niko', max_seqlen=data.max_seqlen,
                            vocab_size=data.vocab_size,
                            n_dummy=data.n_dummy, pretrained_embedding=data.embedding_matrix,
                            params_logger=params_logger, train_logger=train_logger)
    classifier.build_model()

    inputs = [data.train_x[:, i, :] for i in range(data.train_x.shape[1])]
    answers = data.y

    # calculate chance rate for validation  (e.g. the max accuracy when the model always predicts only one label)
    val_answers = answers[-int(len(answers)*0.2):]
    val_1_labels_num = np.sum(val_answers==1)
    val_0_labels_num = np.sum(val_answers==0)
    chancerate = max(val_0_labels_num, val_1_labels_num)/(val_0_labels_num + val_1_labels_num)
    train_logger.info("""validation data has {} samples with 0 and {} samples with 1.\nvalidation chance rate: {}"""
                      .format(val_0_labels_num, val_1_labels_num, chancerate))

    # training phase (model kept every 5 epochs)
    save_path = os.path.join(save_model_path, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    output_dict = classifier.train(inputs, answers, save_output=False, save_path=save_path)
    data_dict = data.retrieve_data()

    # test phase
    # TODO: make code clean
    test_stories = [data.test_x[:, i, :] for i in range(data.test_x.shape[1])]
    test_inputs_with_1 = test_stories + [data.test_e1]
    test_inputs_with_2 = test_stories + [data.test_e2]
    answer_with_1 = classifier.test(test_inputs_with_1, batchsize=32)
    answer_with_2 = classifier.test(test_inputs_with_2, batchsize=32)

    # test accuracy
    acc = classifier.calculate_accuracy(answer_with_1, answer_with_2, gt=data.test_answers)

    # calculate chance rate for test
    # TODO make code clean
    test_answers = data.test_answers
    test_1_labels_num = np.sum(test_answers == 1)
    test_2_labels_num = np.sum(test_answers == 2)
    test_chancerate = max(test_1_labels_num, test_2_labels_num)/len(test_answers)
    train_logger.info("test acc: {}".format(acc/float(data.test_x.shape[0])))
    train_logger.info("""test data has {} samples with 0 and {} samples with 1.\ntest chance rate: {}"""
                      .format(test_1_labels_num, test_2_labels_num, test_chancerate))

    # model and output saving
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    save_intermed_output_path = os.path.join(save_model_path, "output_dict.pkl")
    save_data_path = os.path.join(save_model_path, "data.pkl")
    with open(save_intermed_output_path, 'wb') as w:
        pickle.dump(output_dict, w, protocol=4)
        train_logger.info("output_dict saved: {}".format(save_intermed_output_path))
    with open(save_data_path, 'wb') as w:
        pickle.dump(data_dict, w, protocol=4)
        train_logger.info("data_dict saved: {}".format(save_data_path))

if __name__ == "__main__":
    main()