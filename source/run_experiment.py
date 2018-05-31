import argparse
import pickle
import logging
import os
import numpy as np

from model import Classifier
from data_utils import Data, setup_logger
from path import Path
from datetime import datetime

# rename variable in PyCharm: Shift + F6
# TODO: accept experiment name
parser = argparse.ArgumentParser(description='input argument')
parser.add_argument('--exname', '-e', default=None, help='experiment name.')
args = parser.parse_args()

datestring =  datetime.now().strftime('%m%d%Y_%H%M')
if args.exname:
    datestring = datestring + "_" + str(args.exname)

save_model_path = os.path.join(Path.model_save_path, datestring)
save_log_path = os.path.join(Path.log_path, datestring)
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)
if not os.path.exists(save_log_path):
    os.mkdir(save_log_path)

# == logger setup. ==
# logger for hyper parameters.
params_logger = setup_logger(logger_name='params_logger',
                             file_name=os.path.join(save_log_path, 'params.log'),
                             level=logging.INFO
                             )

# logger for training progress.
train_logger = setup_logger(logger_name='train_logger',
                            file_name=os.path.join(save_log_path, 'train.log'),
                            level=logging.INFO
                            )

def main():
    # data loading
    data = Data(params_logger=params_logger, train_logger=train_logger,
                embedding_path=Path.skpgrm_path,
                data_limit=None,
                w2v_limit=None)

    # classifier configuration and building
    classifier = Classifier(model_name='scnn_CNN', max_seqlen=data.max_seqlen,
                            vocab_size=data.vocab_size,
                            n_dummy=data.n_dummy, pretrained_embedding=data.embedding_matrix,
                            params_logger=params_logger, train_logger=train_logger)
    classifier.build_model()

    # training phase (model kept every 5 epochs)
    inputs = [data.train_x[:, i, :] for i in range(data.train_x.shape[1])]
    answers = data.y

    save_weight_path = os.path.join(save_model_path, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    output_dict = classifier.train(inputs, answers, save_output=False,
                                   validation_split=data.validation_split, save_path=save_weight_path)
    data_dict = data.retrieve_data()

    # test phase
    # TODO: make code clean (wrap these process up as one method in Data class)
    test_stories = [data.test_x[:, i, :] for i in range(data.test_x.shape[1])]
    test_inputs_with_1 = test_stories + [data.test_e1]
    test_inputs_with_2 = test_stories + [data.test_e2]
    answer_with_1 = classifier.test(test_inputs_with_1, batchsize=32) # bsize for test does not matter
    answer_with_2 = classifier.test(test_inputs_with_2, batchsize=32)

    # test accuracy
    acc = classifier.calculate_accuracy(answer_with_1, answer_with_2, gt=data.test_answers)
    train_logger.info("test acc: {}".format(acc / float(data.test_x.shape[0])))

    # model and output saving
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