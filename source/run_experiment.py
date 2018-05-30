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

params_logger = setup_logger(logger_name='params_logger',
                             file_name=os.path.join(log_path, 'params.log'),
                             level=logging.INFO
                             )

train_logger = setup_logger(logger_name='train_logger',
                            file_name=os.path.join(log_path, 'train.log'),
                            level=logging.INFO
                            )

def main():
    data = Data(logger=params_logger, data_limit=None)
    print("data loaded.")

    classifier = Classifier(max_seqlen = data.max_seqlen, vocab_size = data.vocab_size,
                            n_dummy = data.n_dummy, pretrained_embedding = data.embedding_matrix,
                            params_logger=params_logger, train_logger=train_logger)

    classifier.build_scnn()
    print("model built.")

    # inputs = [data.train_x[:, :4, :], data.train_x[:, 4:, :]]
    inputs = [data.train_x[:, i, :] for i in range(data.train_x.shape[1])]
    small_inputs = [data.train_x[:100, i, :] for i in range(data.train_x.shape[1])]
    answers = data.y
    val_answers = answers[-int(len(answers)*0.2):]
    val_1_labels_num = np.sum(val_answers==1)
    val_0_labels_num = np.sum(val_answers==0)
    chancerate = max(val_0_labels_num, val_1_labels_num)/(val_0_labels_num + val_1_labels_num)
    train_logger.info("""validation data has {} samples with 0 and {} samples with 1.\nvalidation chance rate: {}"""
                      .format(val_0_labels_num, val_1_labels_num, chancerate))

    output_dict = classifier.train(inputs, answers, small_inputs)
    # gradients = classifier.check_gradient(inputs)
    # sum_gradients = [np.sum(grad) for grad in gradients]
    # print("gradients: ", sum_gradients)
    data_dict = data.retrieve_data()

    # test_inputs_with_1 = [data.test_x[:, :4, :], data.test_e1]
    # test_inputs_with_2 = [data.test_x[:, :4, :], data.test_e2]

    test_stories = [data.test_x[:, i, :] for i in range(data.test_x.shape[1])]
    test_inputs_with_1 = test_stories + [data.test_e1]
    test_inputs_with_2 = test_stories + [data.test_e2]

    answer_with_1 = classifier.test(test_inputs_with_1, batchsize=32)
    answer_with_2 = classifier.test(test_inputs_with_2, batchsize=32)

    acc = classifier.calculate_accuracy(answer_with_1, answer_with_2, gt=data.test_answers)
    train_logger.info("test acc: {}".format(acc/float(data.test_x.shape[0])))
    print("model saving...")

    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    save_path = os.path.join(save_model_path, "mymodel.h5")
    classifier.save_model(save_path=save_path)
    print("model saved.")

    save_intermed_output_path = os.path.join(save_model_path, "output_dict.pkl")
    save_data_path = os.path.join(save_model_path, "data.pkl")

    with open(save_intermed_output_path, 'wb') as w:
        pickle.dump(output_dict, w, protocol=4)
    with open(save_data_path, 'wb') as w:
        pickle.dump(data_dict, w, protocol=4)

if __name__ == "__main__":
    main()